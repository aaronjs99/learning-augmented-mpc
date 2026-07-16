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
    load_harbor_fault_config,
    run_harbor_simulation,
)
from scripts.harbor.experiments import sweep_network_robustness
from scripts.harbor.learning import run_distributed_harbor_lmpc
from scripts.harbor.mpc import (
    HarborAgentOptimizer,
    _estimate_control_effectiveness,
    _estimate_diagonal_control_effectiveness,
    _identification_channel,
    _information_identification_channel,
    _least_excited_channel,
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
        self.assertEqual([agent.model.control_dim for agent in agents], [2, 2, 2, 8])
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
        for parameter in (
            "yaw_inertia",
            "max_speed",
            "mission_speed",
            "max_reverse_speed",
            "max_yaw_rate",
            "max_force",
            "max_yaw_moment",
            "effective_track",
            "drivetrain",
            "linear_drag",
            "quadratic_drag",
            "yaw_linear_drag",
            "yaw_quadratic_drag",
        ):
            self.assertNotEqual(
                getattr(agents[0].model, parameter),
                getattr(agents[1].model, parameter),
                f"RobEn and Inspector-Gadget must retain distinct {parameter}",
            )
        self.assertNotEqual(agents[0].radius, agents[1].radius)
        self.assertEqual(agents[0].model.drivetrain, "four_wheel_skid_steer")
        self.assertEqual(agents[1].model.drivetrain, "two_motor_skid_steer")
        self.assertEqual(agents[-1].route.shape, (3, 6))
        self.assertEqual(communication.delay_steps, 1)
        self.assertEqual(communication.message_ttl_steps, 12)
        disturbance = load_harbor_disturbance_config()
        self.assertEqual(disturbance.water_current, [-0.1, 0.0, 0.03])
        self.assertEqual(disturbance.ugv_control_effectiveness, 0.92)
        self.assertEqual(disturbance.usv_control_effectiveness, 0.88)
        self.assertEqual(disturbance.rov_control_effectiveness, 0.88)
        self.assertEqual(disturbance.evaluation_hold_steps, 12)

    def test_distinct_ugv_drive_sides_allocate_surge_and_yaw(self) -> None:
        agents, _, _ = load_harbor_config()
        for agent in agents[:2]:
            model = agent.model
            side_force = 0.4 * model.max_side_force
            np.testing.assert_allclose(
                model.generalized_wrench([side_force, side_force]),
                [2.0 * side_force, 0.0],
            )
            np.testing.assert_allclose(
                model.generalized_wrench([-side_force, side_force]),
                [0.0, model.effective_track * side_force],
            )

    def test_heron_waterjets_allocate_surge_and_yaw(self) -> None:
        agents, _, _ = load_harbor_config()
        model = agents[2].model
        jet_thrust = 0.4 * model.max_jet_thrust
        np.testing.assert_allclose(
            model.generalized_wrench([jet_thrust, jet_thrust]),
            [2.0 * jet_thrust, 0.0],
        )
        np.testing.assert_allclose(
            model.generalized_wrench([-jet_thrust, jet_thrust]),
            [0.0, model.waterjet_separation * jet_thrust],
        )

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

    def test_diagonal_estimator_recovers_independent_channel_losses(self) -> None:
        agents, simulation, _ = load_harbor_config()
        config = replace(
            load_harbor_mpc_config(),
            effectiveness_estimator_gain=1.0,
            effectiveness_excitation_threshold=0.001,
            effectiveness_estimator_mode="diagonal",
        )
        for agent in agents:
            truth = np.linspace(0.65, 0.95, agent.model.control_dim)
            estimate = np.ones(agent.model.control_dim)
            for channel in range(agent.model.control_dim):
                command = np.zeros(agent.model.control_dim)
                command[channel] = 0.02 * agent.model.control_scale()[channel]
                measured = agent.model.step(
                    agent.start,
                    truth * command,
                    simulation.dt,
                )
                estimate = _estimate_diagonal_control_effectiveness(
                    agent.model,
                    agent.start,
                    command,
                    measured,
                    simulation.dt,
                    estimate,
                    config,
                )
            np.testing.assert_allclose(estimate, truth, atol=1e-9)

    def test_bluerov2_heavy_allocation_is_full_rank_and_bounded(self) -> None:
        agents, _, _ = load_harbor_config()
        model = agents[-1].model
        self.assertEqual(np.linalg.matrix_rank(model.allocation_matrix), 6)
        self.assertEqual(model.allocation_matrix.shape, (6, 8))
        self.assertIsNotNone(model.thruster_allocation)
        requested = np.array([20.0, -15.0, 30.0, 4.0, -3.0, 5.0])
        command = model.allocate_wrench(requested)
        self.assertTrue(
            np.all(np.abs(command) <= model.thruster_limit_vector + 1e-12)
        )
        np.testing.assert_allclose(
            model.generalized_wrench(command), requested, atol=1e-10
        )

    def test_fault_config_applies_named_channel_vectors(self) -> None:
        agents, _, _ = load_harbor_config()
        disturbance = load_harbor_fault_config()
        expected = {
            "ground_rover_1": [0.68, 0.94],
            "ground_rover_2": [0.90, 0.70],
            "surface_vessel": [0.58, 0.93],
            "underwater_rov": [0.84, 0.95, 0.62, 0.96, 0.72, 0.90, 0.86, 0.68],
        }
        for agent in agents:
            np.testing.assert_allclose(
                disturbance.effectiveness(agent.model, agent.name),
                expected[agent.name],
            )

    def test_agent_fault_vector_must_match_platform_control_dimension(self) -> None:
        agents, _, _ = load_harbor_config()
        disturbance = HarborDisturbanceConfig(
            agent_control_effectiveness={"ground_rover_1": (0.8, 0.9, 1.0)}
        )
        with self.assertRaisesRegex(ValueError, "1 or 2 entries"):
            disturbance.effectiveness(agents[0].model, agents[0].name)

    def test_active_identification_selects_only_under_observed_channels(self) -> None:
        energy = np.array([0.08, 0.01, 0.03])
        self.assertEqual(_least_excited_channel(energy, 0.06), 1)
        self.assertIsNone(_least_excited_channel(energy, 0.01))

    def test_active_identification_rejects_zero_probe(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive when active"):
            replace(
                load_harbor_mpc_config(),
                active_identification=True,
                identification_probe_fraction=0.0,
            )

    def test_information_scheduler_targets_uncertain_identifiable_channel(self) -> None:
        config = replace(
            load_harbor_mpc_config(),
            identification_strategy="information",
            identification_min_probes_per_channel=1,
            identification_target_std=0.01,
        )
        information = np.diag([20.0, 0.5, 5.0])
        increments = np.zeros((3, 3, 3))
        increments[0, 0, 0] = 2.0
        increments[1, 1, 1] = 4.0
        increments[2, 2, 2] = 1.0
        channel = _information_identification_channel(
            information,
            increments,
            np.ones(3, dtype=int),
            np.zeros(3, dtype=int),
            config,
        )
        self.assertEqual(channel, 1)

    def test_information_scheduler_honors_probe_quota_and_rejections(self) -> None:
        config = replace(
            load_harbor_mpc_config(),
            identification_strategy="information",
            identification_min_probes_per_channel=2,
            identification_max_rejections=2,
        )
        increments = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
        channel = _information_identification_channel(
            np.zeros((3, 3)),
            increments,
            np.array([2, 0, 0]),
            np.array([0, 2, 0]),
            config,
        )
        self.assertEqual(channel, 2)

    def test_active_identification_skips_repeatedly_rejected_channel(self) -> None:
        config = replace(
            load_harbor_mpc_config(),
            active_identification=True,
            identification_min_probes_per_channel=1,
            identification_max_rejections=2,
        )
        channel = _identification_channel(
            np.zeros(3),
            np.zeros(3, dtype=int),
            np.array([2, 0, 0]),
            config,
        )
        self.assertEqual(channel, 1)

    def test_optimizer_enforces_selected_identification_probe(self) -> None:
        agents, simulation, _ = load_harbor_config()
        agent = agents[0]
        other = agents[2]
        config = replace(
            load_harbor_mpc_config(),
            prediction_horizon=3,
            active_identification=True,
        )
        optimizer = HarborAgentOptimizer(
            agent=agent,
            other_agents=[other],
            config=config,
            dt=simulation.dt,
            learning=False,
        )
        probe = np.array([0.0, 0.12 * agent.model.max_side_force])
        mask = np.array([0.0, 1.0])
        state = agent.start.copy()
        result = optimizer.solve(
            state=state,
            goal=agent.goal,
            obstacle_predictions={other.name: np.full((3, 3), 100.0)},
            safe_states=np.repeat(state[None, :], 3, axis=0),
            safe_costs=np.zeros(config.terminal_samples),
            warm_states=np.repeat(state[None, :], 4, axis=0),
            warm_controls=np.zeros((3, agent.model.control_dim)),
            previous_control=np.zeros(agent.model.control_dim),
            position_drift=np.zeros(3),
            control_effectiveness=np.ones(agent.model.control_dim),
            identification_probe=probe,
            identification_probe_mask=mask,
        )
        self.assertAlmostEqual(result.control[1], probe[1], places=8)

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
