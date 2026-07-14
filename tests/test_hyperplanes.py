"""Tests for SVM pairwise spatial hyperplane construction."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.learning import get_symmetric_hyperplanes_spatial


class HyperplaneTests(unittest.TestCase):
    def test_nearby_paths_are_inside_opposite_halfspaces(self) -> None:
        path_i = np.array([[0.0, 0.0], [0.0, 0.1]])
        path_j = np.array([[1.0, 0.0], [1.0, 0.1]])

        H_i, h_i, H_j, h_j = get_symmetric_hyperplanes_spatial(
            0,
            0,
            horizon=2,
            traj_i=path_i,
            traj_j=path_j,
            safety_margin_i=0.2,
            safety_margin_j=0.3,
        )

        residual_i = np.sum(H_i * path_i, axis=1) + h_i[:, 0]
        residual_j = np.sum(H_j * path_j, axis=1) + h_j[:, 0]
        self.assertTrue(np.all(residual_i <= 0.0))
        self.assertTrue(np.all(residual_j <= 0.0))
        np.testing.assert_allclose(H_i, -H_j)

    def test_far_paths_disable_pairwise_halfspaces(self) -> None:
        path_i = np.array([[0.0, 0.0]])
        path_j = np.array([[10.0, 0.0]])

        H_i, h_i, H_j, h_j = get_symmetric_hyperplanes_spatial(
            0, 0, horizon=3, traj_i=path_i, traj_j=path_j, ignore_distance=4.0
        )

        np.testing.assert_array_equal(H_i, np.zeros((3, 2)))
        np.testing.assert_array_equal(H_j, np.zeros((3, 2)))
        np.testing.assert_array_equal(h_i, np.full((3, 1), -100.0))
        np.testing.assert_array_equal(h_j, np.full((3, 1), -100.0))

    def test_coincident_references_produce_finite_planes(self) -> None:
        path = np.array([[1.0, 1.0]])
        outputs = get_symmetric_hyperplanes_spatial(
            0, 0, horizon=1, traj_i=path, traj_j=path
        )

        self.assertTrue(all(np.all(np.isfinite(array)) for array in outputs))


if __name__ == "__main__":
    unittest.main()
