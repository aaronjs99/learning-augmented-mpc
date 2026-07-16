"""Integration tests for YAML-backed APF safe-set construction."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.config import load_project_config
from scripts.dynamics import rk4_step_np
from scripts.learning import build_staggered_safe_sets
from scripts.metrics import validate_trajectory


class SafeSetIntegrationTests(unittest.TestCase):
    def test_crossover_seed_is_complete_safe_and_control_aligned(self) -> None:
        project = load_project_config(scenario_name="manta_crossover")
        safe_sets, safe_controls = build_staggered_safe_sets(
            project.scenario,
            dt=project.lmpc.dt,
            apf_config=project.apf,
            dynamics_config=project.dynamics,
        )

        self.assertEqual(set(safe_sets), {0, 1})
        for agent in safe_sets:
            self.assertEqual(len(safe_controls[agent]), len(safe_sets[agent]) - 1)

        validation = validate_trajectory(
            safe_sets,
            project.scenario.goals,
            project.scenario.safety_distance,
            project.scenario.obstacle.center,
            project.scenario.obstacle.radius,
            project.apf.goal_tolerance,
            statuses=None,
        )
        self.assertTrue(validation.valid)
        self.assertTrue(validation.usable_for_learning)

    def test_triangle_seed_is_compact_safe_and_dynamically_aligned(self) -> None:
        project = load_project_config(scenario_name="manta_triangle")
        safe_sets, safe_controls = build_staggered_safe_sets(
            project.scenario,
            dt=project.lmpc.dt,
            apf_config=project.apf,
            dynamics_config=project.dynamics,
        )

        self.assertLessEqual(len(safe_sets[0]) - 1, 200)
        for agent, trajectory in safe_sets.items():
            self.assertEqual(len(safe_controls[agent]), len(trajectory) - 1)
            for step, control in enumerate(safe_controls[agent]):
                expected = rk4_step_np(
                    trajectory[step],
                    control,
                    project.lmpc.dt,
                    project.dynamics,
                )
                np.testing.assert_allclose(
                    trajectory[step + 1], expected, rtol=1e-12, atol=1e-12
                )

        validation = validate_trajectory(
            safe_sets,
            project.scenario.goals,
            project.scenario.safety_distance,
            project.scenario.obstacle.center,
            project.scenario.obstacle.radius,
            project.apf.goal_tolerance,
            statuses=None,
        )
        self.assertTrue(validation.valid)
        self.assertGreaterEqual(
            validation.min_pairwise_distance,
            project.scenario.safety_distance,
        )


if __name__ == "__main__":
    unittest.main()
