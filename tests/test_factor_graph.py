"""Tests for the transparent robust fixed-lag estimator."""

import numpy as np

from scripts.harbor.factor_graph import (
    FactorGraphConfig,
    FixedLagRangeSLAM,
    GraphRange,
    huber_weight,
)


def test_huber_downweights_large_range_residuals():
    assert huber_weight(0.5, 2.5) == 1.0
    assert huber_weight(25.0, 2.5) == 0.1


def test_fixed_lag_window_and_range_rejection_telemetry():
    estimator = FixedLagRangeSLAM(
        FactorGraphConfig(lag=3, iterations=2),
        np.array([0.0, 0.0]),
        {"a": np.array([5.0, 0.0]), "b": np.array([0.0, 5.0])},
        {"a", "b"},
    )
    for _ in range(6):
        report = estimator.update(
            np.array([0.4, 0.0]),
            (GraphRange("a", 4.6), GraphRange("b", 999.0)),
        )
    assert len(estimator.poses) == 3
    assert report.rejected_ranges >= 1
    assert estimator.telemetry[-1]["window_size"] == 3


def test_full_rov_pose_has_attitude_factors_and_psd_covariance_proxy():
    estimator = FixedLagRangeSLAM(
        FactorGraphConfig(lag=4, iterations=3),
        np.zeros(6),
        {"a": np.array([4.0, 0.0, -2.0])},
        {"a"},
    )
    report = estimator.update(
        np.array([0.2, 0.0, -0.1]),
        (GraphRange("a", np.sqrt(4.0**2 + 2.0**2 + 2.0**2)),),
        attitude=np.array([0.0, 0.1, 0.0]),
    )
    assert estimator.pose.shape == (6,)
    assert report.window_size == 2
    assert np.all(np.isfinite(estimator.pose))
    assert np.all(np.linalg.eigvalsh(estimator.position_covariance) >= -1.0e-10)
