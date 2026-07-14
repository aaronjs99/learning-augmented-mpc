"""Tests for variable-agent manta simulation and compatibility behavior."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.simulation import (
    MantaEnvConfig,
    MultiMantaRayEnv,
    ThreeMantaRayEnv,
    manta_rollout,
)


class SimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.starts = np.array(
            [
                [0.0, 0.0, 0.0, 0.1, 0.0, 0.1, 0.0],
                [2.0, 0.0, 3.14, 0.1, 0.0, 0.1, 0.0],
            ]
        )

    def test_multi_manta_environment_supports_two_agents(self) -> None:
        env = MultiMantaRayEnv(
            self.starts, self.starts.copy(), MantaEnvConfig(dt=0.2, horizon=2)
        )

        states, controls = manta_rollout(env)

        self.assertEqual(env.num_agents, 2)
        self.assertEqual(states.shape, (3, 2, 7))
        self.assertEqual(controls.shape, (2, 2, 2))

    def test_three_manta_wrapper_keeps_fixed_shape_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires starts/goals"):
            ThreeMantaRayEnv(
                self.starts,
                self.starts.copy(),
                MantaEnvConfig(dt=0.2, horizon=2),
            )

    def test_multi_manta_environment_rejects_mismatched_agent_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "same shape"):
            MultiMantaRayEnv(
                self.starts,
                self.starts[:1],
                MantaEnvConfig(dt=0.2, horizon=2),
            )

    def test_environment_config_rejects_invalid_horizon(self) -> None:
        with self.assertRaisesRegex(ValueError, "horizon"):
            MantaEnvConfig(horizon=0)


if __name__ == "__main__":
    unittest.main()
