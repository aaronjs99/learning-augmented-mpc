"""Modular range sensing, joint landmark EKF, and observability diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml

from .factor_graph import FactorGraphConfig, FixedLagRangeSLAM, GraphRange


@dataclass(frozen=True)
class RangeBeacon:
    """A simulated harbor transponder and its estimator prior."""

    name: str
    true_position: tuple[float, ...]
    initial_position: tuple[float, ...]
    fixed: bool = False

    def __post_init__(self) -> None:
        truth = tuple(float(value) for value in self.true_position)
        initial = tuple(float(value) for value in self.initial_position)
        if not self.name.strip() or len(truth) not in {2, 3} or len(initial) != len(truth):
            raise ValueError("range beacons require a name and matching 2D or 3D positions")
        if not np.all(np.isfinite(truth + initial)):
            raise ValueError("range beacon positions must be finite")
        object.__setattr__(self, "true_position", truth)
        object.__setattr__(self, "initial_position", initial)


@dataclass(frozen=True)
class RangeAidedSLAMConfig:
    """Sensor, process, and observability settings for range-aided estimation."""

    enabled: bool = False
    mode: str = "joint_landmark_ekf"
    fixed_lag: int = 8
    factor_iterations: int = 4
    robust_range_loss: str = "huber"
    range_std: float = 0.08
    range_bias: float = 0.0
    dropout_probability: float = 0.0
    nlos_probability: float = 0.0
    nlos_bias: float = 0.0
    outlier_probability: float = 0.0
    outlier_std: float = 1.0
    burst_dropout_probability: float = 0.0
    burst_dropout_steps: int = 0
    measurement_delay_steps: int = 0
    maximum_range: float = 20.0
    update_interval_steps: int = 1
    motion_std: tuple[float, ...] = (0.03, 0.03, 0.02)
    odometry_bias: tuple[float, ...] = (0.002, -0.001, 0.001)
    simulate_odometry_noise: bool = True
    initial_pose_std: float = 0.5
    initial_landmark_std: float = 2.0
    observability_window: int = 20
    observability_rank_tolerance: float = 1.0e-6
    defer_landmarks_until_pose_observable: bool = True
    landmark_pose_coupling_std: float = 0.5
    seed: int = 801
    beacons: tuple[RangeBeacon, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in {"known_anchor_ekf", "joint_landmark_ekf", "fixed_lag_slam"}:
            raise ValueError("range-aided mode must be known_anchor_ekf, joint_landmark_ekf, or fixed_lag_slam")
        if self.fixed_lag < 2 or self.factor_iterations < 1:
            raise ValueError("fixed-lag settings must be positive")
        if self.robust_range_loss not in {"huber", "none"}:
            raise ValueError("robust_range_loss must be huber or none")
        if self.range_std <= 0.0 or self.maximum_range <= 0.0:
            raise ValueError("range noise and maximum range must be positive")
        if not 0.0 <= self.dropout_probability <= 1.0:
            raise ValueError("range dropout probability must lie in [0, 1]")
        probabilities = (self.nlos_probability, self.outlier_probability,
                         self.burst_dropout_probability)
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("range fault probabilities must lie in [0, 1]")
        if self.nlos_bias < 0.0 or self.outlier_std <= 0.0:
            raise ValueError("NLOS bias must be nonnegative and outlier_std positive")
        if self.burst_dropout_steps < 0 or self.measurement_delay_steps < 0:
            raise ValueError("range delay settings must be nonnegative")
        if self.update_interval_steps <= 0 or self.observability_window <= 0:
            raise ValueError("range update interval and observability window must be positive")
        if self.initial_pose_std <= 0.0 or self.initial_landmark_std <= 0.0:
            raise ValueError("range-aided prior standard deviations must be positive")
        if self.landmark_pose_coupling_std <= 0.0:
            raise ValueError("landmark pose-coupling standard deviation must be positive")
        dimensions = {len(beacon.true_position) for beacon in self.beacons}
        if len(dimensions) > 1 or len({beacon.name for beacon in self.beacons}) != len(self.beacons):
            raise ValueError("range beacons must have unique names and one spatial dimension")
        if len(self.motion_std) not in {2, 3} or len(self.odometry_bias) not in {2, 3}:
            raise ValueError("motion_std and odometry_bias must cover 2D or 3D motion")
        if np.any(np.asarray(self.motion_std) < 0.0):
            raise ValueError("motion_std must be nonnegative")


@dataclass(frozen=True)
class RangeMeasurement:
    """One accepted scalar range observation."""

    beacon: str
    distance: float


@dataclass(frozen=True)
class ObservabilityReport:
    """Numerical rank diagnostics for the recent nonlinear measurement geometry."""

    state_dimension: int
    rank: int
    smallest_singular_value: float
    condition_number: float
    observable: bool


class RangeSensor:
    """Generate configurable ranges without exposing truth to the estimator."""

    def __init__(self, config: RangeAidedSLAMConfig):
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self._dropout_remaining = 0
        self.telemetry: list[dict] = []

    def measure(self, position: np.ndarray, step: int) -> tuple[RangeMeasurement, ...]:
        if not self.config.enabled or step % self.config.update_interval_steps:
            return ()
        if self._dropout_remaining:
            self._dropout_remaining -= 1
            self.telemetry.append({"step": step, "fault": "burst_dropout", "count": 0})
            return ()
        if self._rng.random() < self.config.burst_dropout_probability:
            self._dropout_remaining = self.config.burst_dropout_steps
            self.telemetry.append({"step": step, "fault": "burst_dropout", "count": 0})
            return ()
        position = np.asarray(position, dtype=float)
        measurements = []
        for beacon in self.config.beacons:
            truth = np.asarray(beacon.true_position, dtype=float)
            if position.size == 2 and truth.size == 3 and truth[2] == 0.0:
                truth = truth[:2]
            if truth.shape != position.shape:
                continue
            distance = float(np.linalg.norm(position - truth))
            if distance > self.config.maximum_range:
                continue
            if self._rng.random() < self.config.dropout_probability:
                self.telemetry.append({"step": step, "beacon": beacon.name, "fault": "dropout"})
                continue
            fault = "none"
            bias = self.config.range_bias
            if self._rng.random() < self.config.nlos_probability:
                bias += self.config.nlos_bias
                fault = "nlos"
            noise_std = self.config.range_std
            if self._rng.random() < self.config.outlier_probability:
                noise_std = self.config.outlier_std
                fault = "outlier"
            noisy = distance + bias + self._rng.normal(
                0.0, noise_std
            )
            measurements.append(RangeMeasurement(beacon.name, max(0.0, float(noisy))))
            self.telemetry.append({"step": step, "beacon": beacon.name, "fault": fault})
        return tuple(measurements)


class RangeAidedEKF:
    """Estimate platform position and, optionally, unknown beacon positions."""

    def __init__(self, config: RangeAidedSLAMConfig, initial_position: np.ndarray):
        self.config = config
        self.dimension = int(np.asarray(initial_position).size)
        if self.dimension not in {2, 3}:
            raise ValueError("range-aided EKF position must be 2D or 3D")
        compatible = [
            beacon
            for beacon in config.beacons
            if len(beacon.true_position) == self.dimension
            or (
                self.dimension == 2
                and len(beacon.true_position) == 3
                and beacon.true_position[2] == 0.0
            )
        ]
        self._beacons = {beacon.name: beacon for beacon in compatible}
        self._landmark_slices: dict[str, slice] = {}
        state = list(np.asarray(initial_position, dtype=float))
        for beacon in compatible:
            estimated = config.mode == "joint_landmark_ekf" and not beacon.fixed
            if estimated:
                start = len(state)
                self._landmark_slices[beacon.name] = slice(start, start + self.dimension)
                state.extend(beacon.initial_position[: self.dimension])
        self.state = np.asarray(state, dtype=float)
        variances = np.full(self.state.size, config.initial_pose_std**2)
        for landmark_slice in self._landmark_slices.values():
            variances[landmark_slice] = config.initial_landmark_std**2
        self.covariance = np.diag(variances)
        self._jacobian_window: list[np.ndarray] = []
        self._anchor_jacobian_window: list[np.ndarray] = []
        self.deferred_landmark_updates = 0
        self.landmark_only_updates = 0

    @property
    def position(self) -> np.ndarray:
        return self.state[: self.dimension].copy()

    @property
    def landmark_estimates(self) -> dict[str, np.ndarray]:
        return {
            name: self.state[index].copy()
            for name, index in self._landmark_slices.items()
        }

    def predict(self, odometry_displacement: np.ndarray) -> None:
        displacement = np.asarray(odometry_displacement, dtype=float)
        if displacement.shape != (self.dimension,):
            raise ValueError("odometry displacement has the wrong dimension")
        self.state[: self.dimension] += displacement
        configured = np.asarray(self.config.motion_std, dtype=float)
        motion_std = configured[: self.dimension]
        if motion_std.size != self.dimension:
            raise ValueError("motion_std must cover the estimator dimension")
        self.covariance[: self.dimension, : self.dimension] += np.diag(
            motion_std * motion_std
        )

    def update(self, measurements: tuple[RangeMeasurement, ...]) -> int:
        accepted = 0
        for measurement in measurements:
            beacon = self._beacons.get(measurement.beacon)
            if beacon is None:
                continue
            landmark_slice = self._landmark_slices.get(measurement.beacon)
            if (
                landmark_slice is not None
                and self.config.defer_landmarks_until_pose_observable
                and not self.pose_observable()
            ):
                self.deferred_landmark_updates += 1
                continue
            landmark = (
                self.state[landmark_slice]
                if landmark_slice is not None
                else np.asarray(beacon.true_position[: self.dimension], dtype=float)
            )
            delta = self.state[: self.dimension] - landmark
            predicted = float(np.linalg.norm(delta))
            if predicted <= 1.0e-9:
                continue
            direction = delta / predicted
            jacobian = np.zeros((1, self.state.size))
            jacobian[0, : self.dimension] = direction
            if landmark_slice is not None:
                jacobian[0, landmark_slice] = -direction
            innovation_variance = float(
                (
                    jacobian @ self.covariance @ jacobian.T
                    + self.config.range_std**2
                ).item()
            )
            gain = self.covariance @ jacobian.T / innovation_variance
            landmark_only = False
            if landmark_slice is not None:
                landmark_std = np.sqrt(
                    np.diag(self.covariance)[landmark_slice]
                )
                if float(np.max(landmark_std)) > self.config.landmark_pose_coupling_std:
                    gain[: self.dimension] = 0.0
                    self.landmark_only_updates += 1
                    landmark_only = True
            navigation_covariance = self.covariance[: self.dimension, :].copy()
            innovation = measurement.distance - self.config.range_bias - predicted
            self.state += gain[:, 0] * innovation
            identity = np.eye(self.state.size)
            residual = identity - gain @ jacobian
            noise = self.config.range_std**2
            self.covariance = (
                residual @ self.covariance @ residual.T + noise * gain @ gain.T
            )
            if landmark_only:
                self.covariance[: self.dimension, :] = navigation_covariance
                self.covariance[:, : self.dimension] = navigation_covariance.T
            self._jacobian_window.append(jacobian)
            self._jacobian_window = self._jacobian_window[-self.config.observability_window :]
            if landmark_slice is None:
                self._anchor_jacobian_window.append(direction.reshape(1, -1))
                self._anchor_jacobian_window = self._anchor_jacobian_window[
                    -self.config.observability_window :
                ]
            accepted += 1
        return accepted

    def pose_observable(self) -> bool:
        """Return whether recent fixed-anchor geometry spans platform position."""
        if not self._anchor_jacobian_window:
            return False
        matrix = np.vstack(self._anchor_jacobian_window)
        singular = np.linalg.svd(matrix, compute_uv=False)
        rank = int(
            np.count_nonzero(singular > self.config.observability_rank_tolerance)
        )
        return rank == self.dimension

    def observability(self) -> ObservabilityReport:
        if not self._jacobian_window:
            return ObservabilityReport(self.state.size, 0, 0.0, float("inf"), False)
        matrix = np.vstack(self._jacobian_window) / self.config.range_std
        singular = np.linalg.svd(matrix, compute_uv=False)
        tolerance = self.config.observability_rank_tolerance
        rank = int(np.count_nonzero(singular > tolerance))
        smallest = float(singular[-1]) if singular.size else 0.0
        condition = (
            float(singular[0] / smallest) if smallest > tolerance else float("inf")
        )
        return ObservabilityReport(
            self.state.size,
            rank,
            smallest,
            condition,
            rank == self.state.size,
        )


class HarborRangeLocalization:
    """Apply independent range-aided estimators behind one simulator boundary."""

    def __init__(self, config: RangeAidedSLAMConfig):
        self.config = config
        self.estimators: dict[str, RangeAidedEKF | FixedLagRangeSLAM] = {}
        self.sensors: dict[str, RangeSensor] = {}
        self._last_onboard_position: dict[str, np.ndarray] = {}
        self._odometry_rngs: dict[str, np.random.Generator] = {}
        self.reports: dict[str, list[ObservabilityReport]] = {}

    def estimate(self, *, agent, truth: np.ndarray, onboard: np.ndarray, step: int) -> np.ndarray:
        """Fuse one platform's odometry increment and ranges into its state estimate."""
        dimension = 3 if agent.model.kind == "rov" else 2
        truth_position = np.asarray(agent.model.position(truth), dtype=float)[:dimension]
        onboard_position = np.asarray(agent.model.position(onboard), dtype=float)[:dimension]
        if agent.name not in self.estimators:
            if self.config.mode == "fixed_lag_slam":
                dimension = 3 if agent.model.kind == "rov" else 2
                compatible = {
                    beacon.name: np.asarray(beacon.initial_position[:dimension], dtype=float)
                    for beacon in self.config.beacons
                    if len(beacon.initial_position) >= dimension
                }
                fixed = {beacon.name for beacon in self.config.beacons if beacon.fixed}
                self.estimators[agent.name] = FixedLagRangeSLAM(
                    FactorGraphConfig(
                        lag=self.config.fixed_lag,
                        iterations=self.config.factor_iterations,
                        range_std=self.config.range_std,
                        odometry_std=max(float(np.mean(self.config.motion_std[:dimension])), 1.0e-3),
                    ), onboard_position, compatible, fixed,
                )
            else:
                self.estimators[agent.name] = RangeAidedEKF(self.config, onboard_position)
            seed_offset = sum((index + 1) * ord(char) for index, char in enumerate(agent.name))
            self.sensors[agent.name] = RangeSensor(
                replace(self.config, seed=self.config.seed + seed_offset)
            )
            self._last_onboard_position[agent.name] = onboard_position.copy()
            self._odometry_rngs[agent.name] = np.random.default_rng(
                self.config.seed + seed_offset + 100000
            )
            self.reports[agent.name] = []
        estimator = self.estimators[agent.name]
        displacement = onboard_position - self._last_onboard_position[agent.name]
        if self.config.simulate_odometry_noise:
            std = np.asarray(self.config.motion_std[:dimension], dtype=float)
            bias = np.asarray(self.config.odometry_bias[:dimension], dtype=float)
            displacement = displacement + bias + self._odometry_rngs[agent.name].normal(
                0.0, std
            )
        if isinstance(estimator, FixedLagRangeSLAM):
            graph_measurements = tuple(
                GraphRange(m.beacon, m.distance, robust=self.config.robust_range_loss != "none")
                for m in self.sensors[agent.name].measure(truth_position, step)
            )
            estimator.update(displacement, graph_measurements)
            self._last_onboard_position[agent.name] = onboard_position.copy()
            self.reports[agent.name].append(
                ObservabilityReport(
                    estimator.pose.size, estimator.last_report.rank,
                    estimator.last_report.sigma_min, estimator.last_report.condition_number,
                    estimator.last_report.mode == "fully-observable",
                )
            )
            estimated = np.asarray(onboard, dtype=float).copy()
            estimated[:dimension] = estimator.position
            return estimated
        estimator.predict(displacement)
        measurements = self.sensors[agent.name].measure(truth_position, step)
        estimator.update(measurements)
        self._last_onboard_position[agent.name] = onboard_position.copy()
        self.reports[agent.name].append(estimator.observability())
        estimated = np.asarray(onboard, dtype=float).copy()
        estimated[:dimension] = estimator.position
        return estimated

    def belief(self, agent_name: str) -> dict:
        """Expose estimator uncertainty without exposing plant truth."""
        estimator = self.estimators.get(agent_name)
        if estimator is None:
            return {"position_covariance": np.eye(3), "observability": None}
        if isinstance(estimator, FixedLagRangeSLAM):
            covariance = np.eye(estimator.position.size) * max(
                estimator.last_report.condition_number, 1.0
            )
            report = estimator.last_report
        else:
            covariance = estimator.covariance[:estimator.dimension, :estimator.dimension].copy()
            report = estimator.observability()
        return {"position_covariance": covariance, "observability": report}


def load_range_aided_slam_config(path: str | Path) -> RangeAidedSLAMConfig:
    """Load nested range-SLAM settings while rejecting unknown fields."""
    with Path(path).open("r", encoding="utf-8") as stream:
        section = (yaml.safe_load(stream) or {}).get("range_aided_slam", {})
    beacon_fields = {"name", "true_position", "initial_position", "fixed"}
    beacons = []
    for raw in section.get("beacons", []):
        unknown = set(raw) - beacon_fields
        if unknown:
            raise ValueError(f"unknown range beacon field(s): {', '.join(sorted(unknown))}")
        beacons.append(RangeBeacon(**raw))
    values = {key: value for key, value in section.items() if key != "beacons"}
    known = set(RangeAidedSLAMConfig.__dataclass_fields__) - {"beacons"}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown range_aided_slam field(s): {', '.join(sorted(unknown))}")
    return RangeAidedSLAMConfig(**values, beacons=tuple(beacons))
