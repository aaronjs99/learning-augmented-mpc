"""Deterministic fixed-lag range SLAM with robust factor reweighting.

This module deliberately uses NumPy rather than a black-box graph package.  The
small solver is transparent, testable, and sufficient for the benchmark's
short windows.  It supports position-only UGV/USV states and six-DOF ROV pose
states (position, roll, pitch, yaw), while velocity is represented by odometry
between consecutive poses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class FactorGraphConfig:
    """Numerical settings for the fixed-lag smoother."""

    lag: int = 8
    iterations: int = 4
    huber_delta: float = 2.5
    range_std: float = 0.08
    odometry_std: float = 0.12
    attitude_std: float = 0.08
    depth_std: float = 0.06
    prior_std: float = 1.0
    damping: float = 1.0e-5
    rank_tolerance: float = 1.0e-7

    def __post_init__(self) -> None:
        if self.lag < 2 or self.iterations < 1:
            raise ValueError("fixed-lag windows require lag >= 2 and iterations >= 1")
        if min(self.huber_delta, self.range_std, self.odometry_std, self.attitude_std,
               self.depth_std, self.prior_std, self.damping) <= 0.0:
            raise ValueError("factor-graph scales must be positive")


@dataclass(frozen=True)
class GraphRange:
    """A range factor associated with a known or estimated beacon."""

    beacon: str
    distance: float
    robust: bool = True


@dataclass(frozen=True)
class FactorGraphReport:
    """Telemetry emitted after each graph update."""

    rank: int
    sigma_min: float
    condition_number: float
    mode: str
    residual_rms: float
    rejected_ranges: int
    window_size: int
    iterations: int


def huber_weight(residual: float, delta: float) -> float:
    """Return the IRLS weight for a standardized Huber residual."""
    magnitude = abs(float(residual))
    return 1.0 if magnitude <= delta else delta / max(magnitude, 1.0e-12)


def _numeric_jacobian(fun, x: np.ndarray) -> np.ndarray:
    base = np.asarray(fun(x), dtype=float)
    jac = np.zeros((base.size, x.size))
    for index in range(x.size):
        step = 1.0e-6 * max(1.0, abs(float(x[index])))
        perturbed = x.copy()
        perturbed[index] += step
        jac[:, index] = (np.asarray(fun(perturbed)) - base) / step
    return jac


class FixedLagRangeSLAM:
    """A robust fixed-lag smoother for one platform and its beacon map."""

    def __init__(self, config: FactorGraphConfig, initial_pose: np.ndarray,
                 beacons: dict[str, np.ndarray], fixed: set[str] | None = None):
        self.config = config
        pose = np.asarray(initial_pose, dtype=float).copy()
        if pose.size not in {2, 3, 6}:
            raise ValueError("pose must be 2D, 3D, or full 6-DOF")
        self.pose_dimension = int(pose.size)
        self._fixed = set(fixed or ())
        dimension = 3 if self.pose_dimension in {3, 6} else 2
        self.beacons = {name: np.asarray(value, dtype=float).copy()[:dimension]
                        for name, value in beacons.items()}
        self._landmarks = {name: value.copy() for name, value in self.beacons.items()
                           if name not in self._fixed}
        self.poses: list[np.ndarray] = [pose]
        self._odometry_history: list[np.ndarray] = []
        self._range_history: list[tuple[GraphRange, ...]] = [()]
        self._attitude_history: list[np.ndarray | None] = [
            pose[3:6].copy() if self.pose_dimension == 6 else None
        ]
        self._prior_pose = pose.copy()
        self._prior_information = np.eye(self.pose_dimension) / config.prior_std**2
        self.position_covariance = np.eye(dimension) * config.prior_std**2
        self.last_report = FactorGraphReport(0, 0.0, float("inf"),
                                             "gauge-underdetermined", 0.0, 0, 1, 0)
        self.telemetry: list[dict] = []

    @property
    def position(self) -> np.ndarray:
        return self.poses[-1][:3 if self.pose_dimension in {3, 6} else 2].copy()

    @property
    def pose(self) -> np.ndarray:
        return self.poses[-1].copy()

    def pose_observable(self) -> bool:
        """Match the EKF observability interface for study reporting."""
        return self.last_report.mode == "fully-observable"

    @property
    def landmark_estimates(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self._landmarks.items()}

    def _landmark_dimension(self) -> int:
        return 3 if self.pose_dimension in {3, 6} else 2

    def _residuals(self, vector: np.ndarray, odometry: list[np.ndarray],
                   ranges: list[tuple[int, GraphRange]], attitudes: list[np.ndarray] | None,
                   depths: list[float | None],
                   layout: dict[str, slice]) -> tuple[np.ndarray, list[bool]]:
        poses = [vector[i * self.pose_dimension:(i + 1) * self.pose_dimension]
                 for i in range(len(self.poses))]
        values = {name: vector[index] for name, index in layout.items()}
        residuals: list[float] = []
        robust_flags: list[bool] = []
        prior = self._prior_information ** 0.5 @ (poses[0] - self._prior_pose)
        residuals.extend(prior)
        for index, displacement in enumerate(odometry, start=1):
            residuals.extend((poses[index][:displacement.size] - poses[index - 1][:displacement.size]
                              - displacement) / self.config.odometry_std)
        for index, measurement in ranges:
            beacon = (values.get(measurement.beacon)
                      if measurement.beacon in values
                      else self.beacons.get(measurement.beacon))
            if beacon is None:
                continue
            position = poses[index][:self._landmark_dimension()]
            delta = position - beacon[:position.size]
            predicted = np.linalg.norm(delta)
            standardized = (predicted - measurement.distance) / self.config.range_std
            weight = huber_weight(standardized, self.config.huber_delta) if measurement.robust else 1.0
            residuals.append(standardized * np.sqrt(weight))
            robust_flags.append(weight >= 0.25)
        if attitudes is not None and self.pose_dimension == 6:
            for index, attitude in enumerate(attitudes):
                if index >= len(poses) or attitude is None:
                    continue
                residuals.extend((poses[index][3:6] - attitude) / self.config.attitude_std)
        if self.pose_dimension == 6:
            for index, depth in enumerate(depths):
                if depth is not None:
                    residuals.append((poses[index][2] - depth) / self.config.depth_std)
        for name, index in layout.items():
            residuals.extend((values[name] - self.beacons[name]) / self.config.prior_std)
        return np.asarray(residuals, dtype=float), robust_flags

    def update(self, odometry: np.ndarray, measurements: tuple[GraphRange, ...] = (),
               attitude: np.ndarray | None = None, depth: float | None = None) -> FactorGraphReport:
        """Append one pose, optimize the lag, and marginalize the oldest pose."""
        displacement = np.asarray(odometry, dtype=float)
        position_dim = self._landmark_dimension()
        if displacement.size != position_dim:
            raise ValueError("odometry dimension does not match platform position")
        next_pose = self.poses[-1].copy()
        next_pose[:position_dim] += displacement
        self.poses.append(next_pose)
        self._odometry_history.append(displacement.copy())
        self._range_history.append(tuple(measurements))
        self._attitude_history.append(
            np.asarray(attitude, dtype=float).copy()
            if attitude is not None and self.pose_dimension == 6 else None
        )
        if not hasattr(self, "_depth_history"):
            self._depth_history = [self.poses[0][2] if self.pose_dimension == 6 else None]
        self._depth_history.append(float(depth) if depth is not None else None)
        attitudes = self._attitude_history.copy()
        depths = self._depth_history.copy()
        odometries = self._odometry_history.copy()
        ranges = [
            (index, measurement)
            for index, row in enumerate(self._range_history)
            for measurement in row
        ]
        layout: dict[str, slice] = {}
        cursor = len(self.poses) * self.pose_dimension
        for name in self._landmarks:
            layout[name] = slice(cursor, cursor + self._landmark_dimension())
            cursor += self._landmark_dimension()
        vector = np.concatenate(self.poses + [self._landmarks[name] for name in self._landmarks])
        rejected = 0
        for _ in range(self.config.iterations):
            residual, flags = self._residuals(vector, odometries, ranges, attitudes, depths, layout)
            jacobian = _numeric_jacobian(
                lambda candidate: self._residuals(candidate, odometries, ranges, attitudes, depths, layout)[0], vector)
            normal = jacobian.T @ jacobian + self.config.damping * np.eye(vector.size)
            step = np.linalg.solve(normal, jacobian.T @ residual)
            vector -= step
            rejected = sum(not flag for flag in flags)
            if np.linalg.norm(step) < 1.0e-6:
                break
        for index in range(len(self.poses)):
            self.poses[index] = vector[index * self.pose_dimension:(index + 1) * self.pose_dimension].copy()
        for name, index in layout.items():
            self._landmarks[name] = vector[index].copy()
        if len(self.poses) > self.config.lag:
            self._prior_pose = self.poses[1].copy()
            self.poses.pop(0)
            self._odometry_history.pop(0)
            self._range_history.pop(0)
            self._attitude_history.pop(0)
            self._depth_history.pop(0)
        information = jacobian.T @ jacobian + self.config.damping * np.eye(vector.size)
        covariance = np.linalg.pinv(information, hermitian=True)
        last_pose_start = (len(self.poses) - 1) * self.pose_dimension
        self.position_covariance = covariance[
            last_pose_start:last_pose_start + self._landmark_dimension(),
            last_pose_start:last_pose_start + self._landmark_dimension(),
        ]
        pose_jacobian = jacobian[:, last_pose_start:last_pose_start + self._landmark_dimension()]
        singular = np.linalg.svd(pose_jacobian, compute_uv=False) if pose_jacobian.size else np.array([])
        rank = int(np.count_nonzero(singular > self.config.rank_tolerance))
        smallest = float(singular[-1]) if singular.size else 0.0
        condition = float(singular[0] / smallest) if smallest > self.config.rank_tolerance else float("inf")
        if rank == 0:
            mode = "gauge-underdetermined"
        elif rank < self._landmark_dimension():
            mode = "weakly-observable"
        elif smallest < 1.0:
            mode = "weakly-observable"
        else:
            mode = "fully-observable"
        report = FactorGraphReport(rank, smallest, condition, mode,
                                   float(np.sqrt(np.mean(residual * residual))) if residual.size else 0.0,
                                   rejected, len(self.poses), self.config.iterations)
        self.last_report = report
        self.telemetry.append({"rank": rank, "sigma_min": smallest, "condition_number": condition,
                               "mode": mode, "rejected_ranges": rejected,
                               "window_size": len(self.poses),
                               "position_covariance_trace": float(np.trace(self.position_covariance))})
        return report
