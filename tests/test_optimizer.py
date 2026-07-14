"""Construction tests for the reusable CasADi manta optimizer."""

from __future__ import annotations

from dataclasses import replace
import unittest

from scripts.mpc import MantaAgentOptimizer, MantaLMPCConfig
from scripts.simulation import StaticObstacle


class OptimizerTests(unittest.TestCase):
    def test_config_rejects_nonpositive_safety_collocation(self) -> None:
        with self.assertRaisesRegex(ValueError, "safety_constraint_substeps"):
            MantaLMPCConfig(safety_constraint_substeps=0)

    def test_config_rejects_negative_safety_filter_buffer(self) -> None:
        with self.assertRaisesRegex(ValueError, "safety_filter_buffer"):
            MantaLMPCConfig(safety_filter_buffer=-0.01)

    def test_optimizer_dimensions_follow_horizon_hull_and_agent_count(self) -> None:
        config = replace(MantaLMPCConfig(), prediction_horizon=3, k_hull=4)
        optimizer = MantaAgentOptimizer(
            config=config,
            num_obstacles=2,
            obstacle=StaticObstacle(center=(3.0, 3.0), radius=0.95),
        )

        self.assertEqual(optimizer.X_state.shape, (7, 4))
        self.assertEqual(optimizer.U.shape, (2, 3))
        self.assertEqual(optimizer.lambdas.shape, (4, 1))
        self.assertEqual(len(optimizer.pH_list), 2)
        self.assertEqual(optimizer.pH_list[0].shape, (3, 2))
        self.assertEqual(optimizer.ph_list[0].shape, (3, 1))

    def test_optimizer_supports_two_agent_problem_shape(self) -> None:
        optimizer = MantaAgentOptimizer(
            config=replace(MantaLMPCConfig(), prediction_horizon=2, k_hull=2),
            num_obstacles=1,
            obstacle=StaticObstacle(center=(3.0, 3.0), radius=0.95),
        )

        self.assertEqual(len(optimizer.pH_list), 1)
        self.assertEqual(optimizer.X_state.shape, (7, 3))


if __name__ == "__main__":
    unittest.main()
