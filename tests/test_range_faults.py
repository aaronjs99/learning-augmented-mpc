"""Deterministic range fault model tests."""

import numpy as np

from scripts.harbor.localization import RangeAidedSLAMConfig, RangeBeacon, RangeSensor


def _config(**changes):
    values = dict(enabled=True, seed=17, range_std=0.01, maximum_range=20.0,
                  beacons=(RangeBeacon("a", (5.0, 0.0), (5.0, 0.0), True),))
    values.update(changes)
    return RangeAidedSLAMConfig(**values)


def test_nlos_bias_is_deterministic_and_recorded():
    sensor = RangeSensor(_config(nlos_probability=1.0, nlos_bias=2.0))
    measurement = sensor.measure(np.array([0.0, 0.0]), 0)[0]
    assert measurement.distance > 6.5
    assert sensor.telemetry[-1]["fault"] == "nlos"


def test_burst_dropout_is_bounded_and_recorded():
    sensor = RangeSensor(_config(burst_dropout_probability=1.0, burst_dropout_steps=2))
    assert sensor.measure(np.array([0.0, 0.0]), 0) == ()
    assert sensor.measure(np.array([0.0, 0.0]), 1) == ()
    assert sensor.measure(np.array([0.0, 0.0]), 2) == ()


def test_delayed_measurement_preserves_capture_step():
    sensor = RangeSensor(_config(measurement_delay_steps=2))
    assert sensor.measure(np.array([0.0, 0.0]), 0) == ()
    assert sensor.measure(np.array([0.0, 0.0]), 1) == ()
    delivered = sensor.measure(np.array([0.0, 0.0]), 2)
    assert len(delivered) == 1
    assert delivered[0].capture_step == 0
