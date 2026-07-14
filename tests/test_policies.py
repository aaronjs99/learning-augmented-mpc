"""Tests for decentralized learning policies."""

from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from scripts.learning.policies import priority_margins, warm_start_from_safe_set
from scripts.mpc import MantaLMPCConfig


class PolicyTests(unittest.TestCase):
    def test_priority_preserves_total_margin_budget(self) -> None:
        config = replace(
            MantaLMPCConfig(),
            hyperplane_safety_margin=0.4,
            priority_margin_scale=0.5,
        )
        current = {0: np.array([0.9, 0.0]), 1: np.array([0.0, 0.0])}
        goals = np.array([[1.0, 0.0], [10.0, 0.0]])
        safe_sets = {
            0: np.array([[0.9, 0.0], [1.0, 0.0]]),
            1: np.array([[0.0, 0.0], [10.0, 0.0]]),
        }

        margin_0, margin_1 = priority_margins(
            0, 1, safe_sets, 0, current, goals, config
        )

        self.assertLess(margin_0, margin_1)
        self.assertAlmostEqual(margin_0 + margin_1, 0.8)

    def test_disabled_priority_returns_equal_margins(self) -> None:
        config = replace(MantaLMPCConfig(), priority_hyperplanes=False)
        margin_0, margin_1 = priority_margins(
            0,
            1,
            {},
            0,
            {0: np.zeros(2), 1: np.ones(2)},
            np.zeros((2, 2)),
            config,
        )
        self.assertEqual((margin_0, margin_1), (0.3, 0.3))

    def test_warm_start_blends_and_holds_safe_memory(self) -> None:
        config = replace(
            MantaLMPCConfig(),
            prediction_horizon=3,
            warm_start_control=1.0,
            warm_start_control_blend=0.25,
        )
        states = np.arange(21, dtype=float).reshape(3, 7)
        controls = np.array([[2.0, 0.0], [0.0, 2.0]])

        warm_states, warm_controls = warm_start_from_safe_set(
            states, controls, 1, config
        )

        np.testing.assert_array_equal(warm_states[:, 0], states[1])
        np.testing.assert_array_equal(warm_states[:, -1], states[-1])
        np.testing.assert_allclose(warm_controls[:, 0], [0.75, 1.25])
        np.testing.assert_allclose(warm_controls[:, -1], [0.75, 1.25])

    def test_invalid_policy_config_fails_at_construction(self) -> None:
        with self.assertRaisesRegex(ValueError, "priority_metric"):
            MantaLMPCConfig(priority_metric="unknown")
        with self.assertRaisesRegex(ValueError, "warm_start_control_blend"):
            MantaLMPCConfig(warm_start_control_blend=1.1)


if __name__ == "__main__":
    unittest.main()
