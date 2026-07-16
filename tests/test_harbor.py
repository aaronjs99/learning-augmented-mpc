"""Tests for heterogeneous harbor dynamics and communication ablations."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from scripts.harbor import load_harbor_config, run_harbor_simulation
from scripts.harbor.experiments import sweep_network_robustness
from scripts.harbor.learning import run_distributed_harbor_lmpc
from scripts.harbor.mpc import load_harbor_mpc_config
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
        self.assertEqual([agent.model.state_dim for agent in agents], [4, 4, 4, 12])
        self.assertEqual([agent.model.control_dim for agent in agents], [2, 2, 2, 6])
        self.assertEqual([agent.model.pose_dim for agent in agents], [3, 3, 3, 6])
        self.assertEqual(agents[-1].route.shape, (3, 6))
        self.assertEqual(communication.delay_steps, 1)
        self.assertEqual(communication.message_ttl_steps, 12)

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
        self.assertGreater(independent.pairwise_violation_count, 0)
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
        rov_control_delta = np.diff(coordinated.controls["underwater_rov"], axis=0)
        self.assertLess(np.max(np.abs(rov_control_delta)), 0.9)
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
        every_step_cost = sum(every_step.first_goal_steps.values())
        self.assertLessEqual(blocked_cost, 1.02 * every_step_cost)
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
        self.assertTrue(all(record["safe_rate"] == 1.0 for record in records))
        with TemporaryDirectory() as directory:
            path = save_network_robustness_heatmap(
                records, Path(directory) / "network.png"
            )
            image = Image.open(path).convert("RGB")
            self.assertGreater(image.width, 500)
            self.assertGreater(len(image.getcolors(maxcolors=1_000_000)), 10)

    def test_distributed_lmpc_is_safe_clean_and_faster_than_guidance(self) -> None:
        agents, simulation, communication = load_harbor_config()
        mpc_config = replace(load_harbor_mpc_config(), learning_iterations=1)
        iterations = run_distributed_harbor_lmpc(
            agents, simulation, communication, mpc_config
        )

        guidance, mpc, lmpc = iterations
        self.assertTrue(mpc.admitted)
        self.assertTrue(lmpc.admitted)
        self.assertLess(mpc.completion_step_sum, guidance.completion_step_sum)
        self.assertLessEqual(lmpc.completion_step_sum, guidance.completion_step_sum)
        for record in (mpc, lmpc):
            self.assertTrue(record.result.all_goals_reached)
            self.assertEqual(record.result.pairwise_violation_count, 0)
            self.assertEqual(record.solver_fallbacks, 0)
            self.assertEqual(record.max_collision_slack, 0.0)

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


if __name__ == "__main__":
    unittest.main()
