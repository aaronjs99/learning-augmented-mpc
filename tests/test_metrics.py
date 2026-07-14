"""Tests for shared sampled and swept rollout metrics."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.metrics import compute_rollout_metrics, swept_pairwise_distances


class MetricsTests(unittest.TestCase):
    def test_rollout_metrics_detect_between_sample_crossing(self) -> None:
        states = np.array(
            [
                [[-1.0, 0.0], [1.0, 0.0]],
                [[1.0, 0.0], [-1.0, 0.0]],
            ]
        )
        metrics = compute_rollout_metrics(
            states,
            goals=states[-1],
            safety_distance=0.5,
            dt=0.2,
        )

        self.assertAlmostEqual(metrics.minimum_pairwise_distance, 0.0)
        self.assertEqual(metrics.collision_count, 1)

    def test_swept_distances_support_static_single_state_rollout(self) -> None:
        distances = swept_pairwise_distances(
            np.array([[[0.0, 0.0], [3.0, 4.0]]])
        )

        self.assertEqual(distances.shape, (1, 1))
        self.assertAlmostEqual(distances[0, 0], 5.0)


if __name__ == "__main__":
    unittest.main()
