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
from scripts.harbor.plotting import save_network_robustness_heatmap


class HarborTests(unittest.TestCase):
    def test_yaml_loads_three_independent_platform_contracts(self) -> None:
        agents, _, communication = load_harbor_config()

        self.assertEqual([agent.model.kind for agent in agents], ["ugv", "usv", "rov"])
        self.assertEqual([agent.model.state_dim for agent in agents], [4, 4, 6])
        self.assertEqual([agent.model.control_dim for agent in agents], [2, 2, 3])
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
        np.testing.assert_allclose(rov_depth, -0.6, atol=1e-10)
        np.testing.assert_allclose(surface_height, 0.0, atol=1e-10)
        self.assertTrue(
            np.all(
                coordinated.positions["ground_rover"][:, 1]
                >= simulation.shoreline_y
            )
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
        reciprocal_cost = sum(reciprocal.first_goal_steps.values())
        eta_cost = sum(eta_priority.first_goal_steps.values())
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
        self.assertEqual(
            sum(blocked.first_goal_steps.values()),
            sum(every_step.first_goal_steps.values()),
        )
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


if __name__ == "__main__":
    unittest.main()
