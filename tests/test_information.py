"""Tests for bounded active observability selection."""

import numpy as np

from scripts.harbor.information import choose_information_waypoint, range_information
from scripts.harbor.localization import HarborRangeLocalization, RangeAidedSLAMConfig


def test_range_information_is_positive_semidefinite():
    matrix = range_information(np.array([0.0, 0.0]), np.array([[3.0, 0.0], [0.0, 3.0]]))
    assert np.all(np.linalg.eigvalsh(matrix) >= -1.0e-10)


def test_waypoint_respects_motion_bound_and_can_improve_geometry():
    current = np.array([0.0, 0.0])
    candidates = np.array([[0.5, 0.0], [0.0, 0.5], [2.0, 0.0]])
    chosen, gain = choose_information_waypoint(
        current, candidates, np.array([[3.0, 0.0], [3.0, 0.2]]), max_step=0.6
    )
    assert np.linalg.norm(chosen - current) <= 0.6 + 1.0e-9
    assert np.isfinite(gain)


def test_information_goal_is_inactive_when_not_configured():
    localization = HarborRangeLocalization(RangeAidedSLAMConfig(active_observability=False))
    goal = np.array([1.0, 2.0, 0.0])
    assert np.array_equal(localization.information_goal(type("A", (), {"name": "x"})(), np.zeros(3), goal), goal)
