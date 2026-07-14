"""Tests for reusable trajectory evaluation."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.metrics import cost_by_iteration, history_to_tensor, validate_trajectory


class EvaluationTests(unittest.TestCase):
    def test_history_padding_holds_final_state(self) -> None:
        history = {
            0: np.array([[0.0, 0.0], [1.0, 0.0]]),
            1: np.array([[3.0, 0.0]]),
        }

        agents, states = history_to_tensor(history)

        self.assertEqual(agents, [0, 1])
        self.assertEqual(states.shape, (2, 2, 2))
        np.testing.assert_array_equal(states[:, 1], [[3.0, 0.0], [3.0, 0.0]])

    def test_validation_requires_completion_and_safety(self) -> None:
        history = {
            0: np.array([[0.0, 0.0], [1.0, 0.0]]),
            1: np.array([[0.0, 2.0], [1.0, 2.0]]),
        }
        validation = validate_trajectory(
            history,
            goals=np.array([[1.0, 0.0], [1.0, 2.0]]),
            safety_distance=0.5,
            obstacle_center=(10.0, 10.0),
            obstacle_radius=1.0,
            goal_tolerance=0.1,
            statuses=[{0: "ok", 1: "fallback_apf"}],
        )

        self.assertTrue(validation.valid)
        self.assertTrue(validation.safe)
        self.assertFalse(validation.solver_clean)
        self.assertEqual(validation.fallback_count, 1)
        self.assertEqual(validation.to_dict()["usable_for_learning"], True)

    def test_collision_prevents_safe_set_admission(self) -> None:
        history = {
            0: np.array([[0.0, 0.0], [1.0, 0.0]]),
            1: np.array([[0.2, 0.0], [1.2, 0.0]]),
        }
        validation = validate_trajectory(
            history,
            goals=np.array([[1.0, 0.0], [1.2, 0.0]]),
            safety_distance=0.5,
            obstacle_center=(10.0, 10.0),
            obstacle_radius=1.0,
            goal_tolerance=0.1,
            statuses=None,
        )

        self.assertFalse(validation.safe)
        self.assertFalse(validation.usable_for_learning)
        self.assertEqual(validation.pairwise_violation_count, 2)

    def test_cost_uses_first_goal_hit_and_length_when_unreached(self) -> None:
        histories = [{
            0: np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            1: np.array([[0.0, 2.0], [0.5, 2.0], [1.0, 2.0]]),
        }]
        costs = cost_by_iteration(
            histories,
            goals=np.array([[1.0, 0.0], [3.0, 2.0]]),
            goal_tolerance=0.01,
        )

        self.assertEqual(costs, {0: [1], 1: [3]})


if __name__ == "__main__":
    unittest.main()
