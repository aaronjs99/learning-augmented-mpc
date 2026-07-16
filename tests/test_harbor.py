"""Tests for heterogeneous harbor dynamics and communication ablations."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import casadi as ca
import numpy as np
from PIL import Image

from scripts.harbor import (
    DEFAULT_HARBOR_CONFIG,
    HarborDisturbanceConfig,
    load_harbor_config,
    load_harbor_disturbance_config,
    run_harbor_simulation,
)
from scripts.harbor.experiments import sweep_network_robustness
from scripts.harbor.learning import run_distributed_harbor_lmpc
from scripts.harbor.mpc import (
    _estimate_control_effectiveness,
    load_harbor_mpc_config,
)
from scripts.harbor.plotting import (
    save_harbor_learning_progress,
    save_network_robustness_heatmap,
)


class HarborTests(unittest.TestCase):
    def test_yaml_loads_four_independent_platform_contracts(self) -> None:
        agents, _, communication = load_harbor_config()

        self.assertEqual(
            [agent.model.kind for agent in agents], ["ugv", "ugv", "usv", "rov"]
        )
        self.assertEqual([agent.model.state_dim for agent in agents], [5, 5, 6, 12])
        self.assertEqual([agent.model.control_dim for agent in agents], [2, 2, 2, 6])
        self.assertEqual([agent.model.pose_dim for agent in agents], [3, 3, 3, 6])
        self.assertEqual(
            [agent.model.variant for agent in agents],
            [
                "dynamic_skid_steer",
                "dynamic_skid_steer",
                "marine_3dof",
                "marine_6dof",
            ],
        )
        self.assertEqual(
            [agent.profile for agent in agents],
            [
                "srilab_roben_jackal",
                "srilab_inspector_gadget_husky",
                "clearpath_heron_full_payload",
                "bluerov2_heavy",
            ],
        )
        self.assertNotEqual(agents[0].model.mass, agents[1].model.mass)
        self.assertEqual(agents[-1].route.shape, (3, 6))
        self.assertEqual(communication.delay_steps, 1)
        self.assertEqual(communication.message_ttl_steps, 12)
        disturbance = load_harbor_disturbance_config()
        self.assertEqual(disturbance.water_current, [-0.1, 0.0, 0.03])
        self.assertEqual(disturbance.ugv_control_effectiveness, 0.92)
        self.assertEqual(disturbance.usv_control_effectiveness, 0.88)
        self.assertEqual(disturbance.rov_control_effectiveness, 0.88)
        self.assertEqual(disturbance.evaluation_hold_steps, 12)

    def test_local_effectiveness_estimator_recovers_every_platform_loss(self) -> None:
        agents, simulation, _ = load_harbor_config()
        config = replace(
            load_harbor_mpc_config(), effectiveness_estimator_gain=1.0
        )
        actual_effectiveness = 0.82
        for agent in agents:
            command = 0.45 * agent.model.control_scale()
            measured = agent.model.step(
                agent.start,
                actual_effectiveness * command,
                simulation.dt,
            )
            estimate = _estimate_control_effectiveness(
                agent.model,
                agent.start,
                command,
                measured,
                simulation.dt,
                1.0,
                config,
            )
            self.assertAlmostEqual(estimate, actual_effectiveness, places=10)

    def test_hidden_current_advects_only_marine_execution_plant(self) -> None:
        agents, simulation, communication = load_harbor_config()
        selected = [
            replace(
                agent,
                goal=np.asarray(agent.start[: agent.model.pose_dim]).copy(),
                waypoints=None,
            )
            for agent in (agents[0], agents[2])
        ]
        result = run_harbor_simulation(
            selected,
            replace(simulation, horizon=1, goal_hold_steps=2),
            replace(communication, enabled=False),
            disturbance=HarborDisturbanceConfig(water_current=(0.2, -0.1, 0.05)),
        )

        np.testing.assert_allclose(
            result.positions[selected[0].name][-1],
            result.positions[selected[0].name][0],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.positions[selected[1].name][-1]
            - result.positions[selected[1].name][0],
            simulation.dt * np.array([0.2, -0.1, 0.0]),
            atol=1e-12,
        )
        self.assertEqual(len(result.applied_controls[selected[1].name]), 1)

    def test_communication_improves_safety_without_pose_coupling(self) -> None:
        agents, simulation, communication = load_harbor_config()
        independent = run_harbor_simulation(
            agents, simulation, replace(communication, enabled=False)
        )
        coordinated = run_harbor_simulation(
            agents, simulation, replace(communication, enabled=True)
        )

        self.assertTrue(independent.all_goals_reached)
        self.assertTrue(coordinated.all_goals_reached)
        self.assertEqual(independent.pairwise_violation_count, 0)
        self.assertEqual(coordinated.pairwise_violation_count, 0)
        self.assertGreater(
            coordinated.min_pairwise_distance, independent.min_pairwise_distance
        )
        self.assertGreater(coordinated.messages_delivered, 0)

        rov_depth = coordinated.positions["underwater_rov"][:, 2]
        surface_height = coordinated.positions["surface_vessel"][:, 2]
        self.assertGreater(np.ptp(rov_depth), 1.0)
        self.assertLessEqual(np.max(rov_depth), -0.3)
        self.assertGreaterEqual(np.min(rov_depth), simulation.seabed_z)
        rov = next(agent for agent in agents if agent.model.kind == "rov")
        normalized_rov_control = (
            coordinated.controls[rov.name] / rov.model.control_scale()
        )
        rov_control_delta = np.diff(normalized_rov_control, axis=0)
        self.assertLess(np.max(np.abs(rov_control_delta)), 0.5)
        self.assertLess(
            coordinated.final_orientation_errors["underwater_rov"],
            simulation.orientation_tolerance,
        )
        np.testing.assert_allclose(surface_height, 0.0, atol=1e-10)
        for name in ("ground_rover_1", "ground_rover_2"):
            self.assertTrue(
                np.all(coordinated.positions[name][:, 1] >= simulation.shoreline_y)
            )
        self.assertTrue(
            np.all(
                coordinated.positions["surface_vessel"][:, 1]
                <= simulation.shoreline_y
            )
        )
        self.assertTrue(
            np.all(
                coordinated.positions["underwater_rov"][:, 1]
                <= simulation.shoreline_y
            )
        )

    def test_eta_priority_improves_completion_cost_over_reciprocal(self) -> None:
        agents, simulation, communication = load_harbor_config()
        reciprocal = run_harbor_simulation(
            agents,
            replace(simulation, coordination_policy="reciprocal"),
            communication,
        )
        eta_priority = run_harbor_simulation(
            agents,
            replace(simulation, coordination_policy="eta_priority"),
            communication,
        )

        self.assertEqual(reciprocal.pairwise_violation_count, 0)
        self.assertEqual(eta_priority.pairwise_violation_count, 0)
        reciprocal_cost = sum(
            step if step is not None else simulation.horizon + 1
            for step in reciprocal.first_goal_steps.values()
        )
        eta_cost = sum(
            step if step is not None else simulation.horizon + 1
            for step in eta_priority.first_goal_steps.values()
        )
        self.assertFalse(reciprocal.all_goals_reached)
        self.assertTrue(eta_priority.all_goals_reached)
        self.assertLess(eta_cost, reciprocal_cost)

    def test_block_guidance_reduces_updates_without_regression(self) -> None:
        agents, simulation, communication = load_harbor_config()
        every_step = run_harbor_simulation(
            agents,
            replace(simulation, guidance_update_interval_steps=1),
            communication,
        )
        blocked = run_harbor_simulation(
            agents,
            replace(simulation, guidance_update_interval_steps=2),
            communication,
        )

        self.assertEqual(every_step.pairwise_violation_count, 0)
        self.assertEqual(blocked.pairwise_violation_count, 0)
        blocked_cost = sum(blocked.first_goal_steps.values())
        every_step_cost = sum(
            step if step is not None else simulation.horizon + 1
            for step in every_step.first_goal_steps.values()
        )
        self.assertTrue(blocked.all_goals_reached)
        self.assertLessEqual(blocked_cost, every_step_cost)
        self.assertLess(
            blocked.guidance_update_count,
            0.6 * every_step.guidance_update_count,
        )

    def test_network_sweep_writes_nonblank_heatmap(self) -> None:
        agents, simulation, communication = load_harbor_config()
        records = sweep_network_robustness(
            agents,
            simulation,
            communication,
            delays=[0, 2],
            dropout_probabilities=[0.0, 0.5],
            seeds=[1, 2],
        )

        self.assertEqual(len(records), 4)
        self.assertTrue(
            all(0.0 <= record["safe_rate"] <= 1.0 for record in records)
        )
        baseline = next(
            record
            for record in records
            if record["delay_steps"] == 0
            and record["dropout_probability"] == 0.0
        )
        self.assertEqual(baseline["safe_rate"], 1.0)
        with TemporaryDirectory() as directory:
            path = save_network_robustness_heatmap(
                records, Path(directory) / "network.png"
            )
            image = Image.open(path).convert("RGB")
            self.assertGreater(image.width, 500)
            self.assertGreater(len(image.getcolors(maxcolors=1_000_000)), 10)

    def test_distributed_lmpc_is_safe_clean_and_rejects_cost_regression(self) -> None:
        agents, simulation, communication = load_harbor_config()
        mpc_config = replace(load_harbor_mpc_config(), learning_iterations=1)
        iterations = run_distributed_harbor_lmpc(
            agents, simulation, communication, mpc_config
        )

        guidance, mpc, lmpc = iterations
        self.assertTrue(mpc.admitted)
        self.assertFalse(lmpc.admitted)
        self.assertLess(mpc.completion_step_sum, guidance.completion_step_sum)
        self.assertGreater(lmpc.completion_step_sum, mpc.completion_step_sum)
        for record in (mpc, lmpc):
            self.assertTrue(record.result.all_goals_reached)
            self.assertEqual(record.result.pairwise_violation_count, 0)
            self.assertEqual(record.solver_fallbacks, 0)
            self.assertLess(record.max_collision_slack, 1e-9)

        with TemporaryDirectory() as directory:
            path = save_harbor_learning_progress(
                iterations,
                agents,
                simulation,
                Path(directory) / "progress.png",
            )
            image = Image.open(path).convert("RGB")
            self.assertGreater(image.width, 1000)
            self.assertGreater(len(image.getcolors(maxcolors=1_000_000)), 10)

    def test_symbolic_and_numeric_platform_steps_match(self) -> None:
        configs = [
            DEFAULT_HARBOR_CONFIG,
            DEFAULT_HARBOR_CONFIG.with_name("harbor_reduced.yaml"),
        ]
        for config in configs:
            agents, simulation, _ = load_harbor_config(config)
            for agent in (agents[0], agents[2], agents[3]):
                model = agent.model
                state = np.asarray(agent.start, dtype=float).copy()
                goal_delta = model.goal_position(agent.goal) - model.position(state)
                distance = np.linalg.norm(goal_delta)
                desired_velocity = (
                    np.zeros(3)
                    if distance == 0.0
                    else goal_delta / distance
                    * (
                        model.max_horizontal_speed
                        if model.kind == "rov"
                        else model.max_speed
                    )
                )
                control = model.guidance_control(
                    state,
                    desired_velocity,
                    simulation.dt,
                    desired_pose=agent.goal,
                )
                symbolic_state = ca.MX.sym("state", model.state_dim)
                symbolic_control = ca.MX.sym("control", model.control_dim)
                transition = ca.Function(
                    f"step_{model.variant}",
                    [symbolic_state, symbolic_control],
                    [
                        model.symbolic_step(
                            ca,
                            symbolic_state,
                            symbolic_control,
                            simulation.dt,
                        )
                    ],
                )
                predicted = np.asarray(transition(state, control)).reshape(-1)
                executed = model.step(state, control, simulation.dt)
                np.testing.assert_allclose(predicted, executed, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
