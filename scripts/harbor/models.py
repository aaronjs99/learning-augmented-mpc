"""Platform-specific dynamics behind a shared harbor guidance interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi]``."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class PlatformModel(ABC):
    """Minimal dynamics contract required by heterogeneous coordination."""

    kind: str
    state_dim: int
    control_dim: int
    pose_dim: int

    @abstractmethod
    def position(self, state: np.ndarray) -> np.ndarray:
        """Return world-frame ``[x, y, z]`` position."""

    @abstractmethod
    def velocity(self, state: np.ndarray) -> np.ndarray:
        """Return world-frame ``[vx, vy, vz]`` velocity."""

    @abstractmethod
    def goal_position(self, goal: np.ndarray) -> np.ndarray:
        """Map a platform pose goal to world-frame ``[x, y, z]``."""

    @abstractmethod
    def orientation_error(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Return wrapped angular pose error in the platform's DOF space."""

    @abstractmethod
    def guidance_control(
        self,
        state: np.ndarray,
        desired_velocity: np.ndarray,
        dt: float,
        desired_pose: np.ndarray | None = None,
    ) -> np.ndarray:
        """Map a desired world velocity to bounded platform controls."""

    @abstractmethod
    def step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        """Advance one platform state."""


@dataclass(frozen=True)
class UGVModel(PlatformModel):
    """Acceleration-controlled planar unicycle for a harbor ground vehicle."""

    max_speed: float = 1.0
    max_acceleration: float = 1.0
    max_yaw_rate: float = 1.2
    ground_z: float = 0.0
    kind: str = field(default="ugv", init=False)
    state_dim: int = field(default=4, init=False)
    control_dim: int = field(default=2, init=False)
    pose_dim: int = field(default=3, init=False)

    def __post_init__(self) -> None:
        if min(self.max_speed, self.max_acceleration, self.max_yaw_rate) <= 0.0:
            raise ValueError("UGV speed, acceleration, and yaw-rate bounds must be positive")

    def position(self, state: np.ndarray) -> np.ndarray:
        return np.array([state[0], state[1], self.ground_z], dtype=float)

    def velocity(self, state: np.ndarray) -> np.ndarray:
        return np.array(
            [state[3] * np.cos(state[2]), state[3] * np.sin(state[2]), 0.0]
        )

    def goal_position(self, goal: np.ndarray) -> np.ndarray:
        return np.array([goal[0], goal[1], self.ground_z], dtype=float)

    def orientation_error(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return np.array([wrap_angle(goal[2] - state[2])])

    def guidance_control(
        self,
        state: np.ndarray,
        desired_velocity: np.ndarray,
        dt: float,
        desired_pose: np.ndarray | None = None,
    ) -> np.ndarray:
        desired = np.asarray(desired_velocity, dtype=float)[:2]
        speed = min(float(np.linalg.norm(desired)), self.max_speed)
        heading = (
            float(desired_pose[2])
            if speed < 1e-9 and desired_pose is not None
            else state[2] if speed < 1e-9 else float(np.arctan2(desired[1], desired[0]))
        )
        return np.array(
            [
                np.clip((speed - state[3]) / dt, -self.max_acceleration, self.max_acceleration),
                np.clip(wrap_angle(heading - state[2]) / dt, -self.max_yaw_rate, self.max_yaw_rate),
            ]
        )

    def step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        value = np.asarray(state, dtype=float).copy()
        acceleration = np.clip(control[0], -self.max_acceleration, self.max_acceleration)
        yaw_rate = np.clip(control[1], -self.max_yaw_rate, self.max_yaw_rate)
        value[3] = np.clip(value[3] + dt * acceleration, 0.0, self.max_speed)
        value[2] = wrap_angle(value[2] + dt * yaw_rate)
        value[0] += dt * value[3] * np.cos(value[2])
        value[1] += dt * value[3] * np.sin(value[2])
        return value


@dataclass(frozen=True)
class USVModel(PlatformModel):
    """Planar surge-yaw surface-vessel model with linear drag."""

    max_speed: float = 1.0
    max_thrust: float = 1.5
    max_yaw_rate: float = 0.8
    drag: float = 0.25
    surface_z: float = 0.0
    kind: str = field(default="usv", init=False)
    state_dim: int = field(default=4, init=False)
    control_dim: int = field(default=2, init=False)
    pose_dim: int = field(default=3, init=False)

    def __post_init__(self) -> None:
        if min(self.max_speed, self.max_thrust, self.max_yaw_rate) <= 0.0:
            raise ValueError("USV speed, thrust, and yaw-rate bounds must be positive")
        if self.drag < 0.0:
            raise ValueError("USV drag must be nonnegative")

    def position(self, state: np.ndarray) -> np.ndarray:
        return np.array([state[0], state[1], self.surface_z], dtype=float)

    def velocity(self, state: np.ndarray) -> np.ndarray:
        return np.array(
            [state[3] * np.cos(state[2]), state[3] * np.sin(state[2]), 0.0]
        )

    def goal_position(self, goal: np.ndarray) -> np.ndarray:
        return np.array([goal[0], goal[1], self.surface_z], dtype=float)

    def orientation_error(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return np.array([wrap_angle(goal[2] - state[2])])

    def guidance_control(
        self,
        state: np.ndarray,
        desired_velocity: np.ndarray,
        dt: float,
        desired_pose: np.ndarray | None = None,
    ) -> np.ndarray:
        desired = np.asarray(desired_velocity, dtype=float)[:2]
        speed = min(float(np.linalg.norm(desired)), self.max_speed)
        heading = (
            float(desired_pose[2])
            if speed < 1e-9 and desired_pose is not None
            else state[2] if speed < 1e-9 else float(np.arctan2(desired[1], desired[0]))
        )
        thrust = (speed - state[3]) / dt + self.drag * state[3]
        return np.array(
            [
                np.clip(thrust, -self.max_thrust, self.max_thrust),
                np.clip(wrap_angle(heading - state[2]) / dt, -self.max_yaw_rate, self.max_yaw_rate),
            ]
        )

    def step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        value = np.asarray(state, dtype=float).copy()
        thrust = np.clip(control[0], -self.max_thrust, self.max_thrust)
        yaw_rate = np.clip(control[1], -self.max_yaw_rate, self.max_yaw_rate)
        value[3] = np.clip(
            value[3] + dt * (thrust - self.drag * value[3]), 0.0, self.max_speed
        )
        value[2] = wrap_angle(value[2] + dt * yaw_rate)
        value[0] += dt * value[3] * np.cos(value[2])
        value[1] += dt * value[3] * np.sin(value[2])
        return value


@dataclass(frozen=True)
class ROVModel(PlatformModel):
    """Untethered damped 6-DOF ROV with world-frame wrench control."""

    max_horizontal_speed: float = 0.8
    max_vertical_speed: float = 0.5
    max_force: float = 1.2
    max_angular_rate: float = 0.7
    max_torque: float = 1.0
    linear_drag: float = 0.3
    angular_drag: float = 0.4
    velocity_response_gain: float = 1.8
    attitude_response_gain: float = 1.0
    angular_rate_response_gain: float = 2.0
    control_smoothing: float = 0.4
    kind: str = field(default="rov", init=False)
    state_dim: int = field(default=12, init=False)
    control_dim: int = field(default=6, init=False)
    pose_dim: int = field(default=6, init=False)

    def __post_init__(self) -> None:
        positive = (
            self.max_horizontal_speed,
            self.max_vertical_speed,
            self.max_force,
            self.max_angular_rate,
            self.max_torque,
            self.velocity_response_gain,
            self.attitude_response_gain,
            self.angular_rate_response_gain,
            self.control_smoothing,
        )
        if min(positive) <= 0.0:
            raise ValueError("ROV speed, force, and yaw-rate bounds must be positive")
        if self.control_smoothing > 1.0:
            raise ValueError("ROV control_smoothing must be in (0, 1]")
        if self.linear_drag < 0.0 or self.angular_drag < 0.0:
            raise ValueError("ROV drag values must be nonnegative")

    def position(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state[:3], dtype=float).copy()

    def velocity(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state[6:9], dtype=float).copy()

    def goal_position(self, goal: np.ndarray) -> np.ndarray:
        return np.asarray(goal[:3], dtype=float).copy()

    def orientation_error(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return np.asarray([wrap_angle(goal[i] - state[i]) for i in range(3, 6)])

    def guidance_control(
        self,
        state: np.ndarray,
        desired_velocity: np.ndarray,
        dt: float,
        desired_pose: np.ndarray | None = None,
    ) -> np.ndarray:
        desired = np.asarray(desired_velocity, dtype=float)
        planar_speed = np.linalg.norm(desired[:2])
        if planar_speed > self.max_horizontal_speed:
            desired[:2] *= self.max_horizontal_speed / planar_speed
        desired[2] = np.clip(
            desired[2], -self.max_vertical_speed, self.max_vertical_speed
        )
        angular_error = (
            np.zeros(3)
            if desired_pose is None
            else self.orientation_error(state, desired_pose)
        )
        desired_rates = np.clip(
            self.attitude_response_gain * angular_error,
            -self.max_angular_rate,
            self.max_angular_rate,
        )
        forces = np.clip(
            self.velocity_response_gain * (desired - state[6:9])
            + self.linear_drag * state[6:9],
            -self.max_force,
            self.max_force,
        )
        torques = np.clip(
            self.angular_rate_response_gain * (desired_rates - state[9:12])
            + self.angular_drag * state[9:12],
            -self.max_torque,
            self.max_torque,
        )
        return np.concatenate((forces, torques))

    def step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        value = np.asarray(state, dtype=float).copy()
        forces = np.clip(control[:3], -self.max_force, self.max_force)
        torques = np.clip(control[3:6], -self.max_torque, self.max_torque)
        value[6:9] += dt * (forces - self.linear_drag * value[6:9])
        horizontal_speed = np.linalg.norm(value[6:8])
        if horizontal_speed > self.max_horizontal_speed:
            value[6:8] *= self.max_horizontal_speed / horizontal_speed
        value[8] = np.clip(
            value[8], -self.max_vertical_speed, self.max_vertical_speed
        )
        value[9:12] = np.clip(
            value[9:12] + dt * (torques - self.angular_drag * value[9:12]),
            -self.max_angular_rate,
            self.max_angular_rate,
        )
        value[:3] += dt * value[6:9]
        value[3:6] = [wrap_angle(angle) for angle in value[3:6] + dt * value[9:12]]
        return value


def make_platform_model(kind: str, parameters: dict[str, float]) -> PlatformModel:
    """Construct one platform model from YAML parameters."""
    models = {"ugv": UGVModel, "usv": USVModel, "rov": ROVModel}
    try:
        return models[kind.lower()](**parameters)
    except KeyError as exc:
        raise ValueError(f"unsupported harbor platform kind: {kind}") from exc
