"""Tests for heterogeneous harbor dynamics and communication ablations."""

from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from scripts.harbor import load_harbor_config, run_harbor_simulation


class HarborTests(unittest.TestCase):
    def test_yaml_loads_three_independent_platform_contracts(self) -> None:
        agents, _, communication = load_harbor_config()

        self.assertEqual([agent.model.kind for agent in agents], ["ugv", "usv", "rov"])
        self.assertEqual([agent.model.state_dim for agent in agents], [4, 4, 6])
        self.assertEqual([agent.model.control_dim for agent in agents], [2, 2, 3])
        self.assertEqual(communication.delay_steps, 1)
        self.assertEqual(communication.message_ttl_steps, 3)

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
        np.testing.assert_allclose(rov_depth, -1.5, atol=1e-10)
        np.testing.assert_allclose(surface_height, 0.0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
