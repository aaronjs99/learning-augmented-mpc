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

    @abstractmethod
    def position(self, state: np.ndarray) -> np.ndarray:
        """Return world-frame ``[x, y, z]`` position."""

    @abstractmethod
    def velocity(self, state: np.ndarray) -> np.ndarray:
        """Return world-frame ``[vx, vy, vz]`` velocity."""

    @abstractmethod
    def guidance_control(
        self, state: np.ndarray, desired_velocity: np.ndarray, dt: float
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

    def __post_init__(self) -> None:
        if min(self.max_speed, self.max_acceleration, self.max_yaw_rate) <= 0.0:
            raise ValueError("UGV speed, acceleration, and yaw-rate bounds must be positive")

    def position(self, state: np.ndarray) -> np.ndarray:
        return np.array([state[0], state[1], self.ground_z], dtype=float)

    def velocity(self, state: np.ndarray) -> np.ndarray:
        return np.array(
            [state[3] * np.cos(state[2]), state[3] * np.sin(state[2]), 0.0]
        )

    def guidance_control(
        self, state: np.ndarray, desired_velocity: np.ndarray, dt: float
    ) -> np.ndarray:
        desired = np.asarray(desired_velocity, dtype=float)[:2]
        speed = min(float(np.linalg.norm(desired)), self.max_speed)
        heading = state[2] if speed < 1e-9 else float(np.arctan2(desired[1], desired[0]))
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

    def guidance_control(
        self, state: np.ndarray, desired_velocity: np.ndarray, dt: float
    ) -> np.ndarray:
        desired = np.asarray(desired_velocity, dtype=float)[:2]
        speed = min(float(np.linalg.norm(desired)), self.max_speed)
        heading = state[2] if speed < 1e-9 else float(np.arctan2(desired[1], desired[0]))
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
    """Untethered underwater surge-heave-yaw model with linear drag."""

    max_surge_speed: float = 0.8
    max_heave_speed: float = 0.5
    max_surge_force: float = 1.2
    max_heave_force: float = 0.8
    max_yaw_rate: float = 0.7
    surge_drag: float = 0.3
    heave_drag: float = 0.4
    kind: str = field(default="rov", init=False)
    state_dim: int = field(default=6, init=False)
    control_dim: int = field(default=3, init=False)

    def __post_init__(self) -> None:
        positive = (
            self.max_surge_speed,
            self.max_heave_speed,
            self.max_surge_force,
            self.max_heave_force,
            self.max_yaw_rate,
        )
        if min(positive) <= 0.0:
            raise ValueError("ROV speed, force, and yaw-rate bounds must be positive")
        if self.surge_drag < 0.0 or self.heave_drag < 0.0:
            raise ValueError("ROV drag values must be nonnegative")

    def position(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state[:3], dtype=float).copy()

    def velocity(self, state: np.ndarray) -> np.ndarray:
        return np.array(
            [state[4] * np.cos(state[3]), state[4] * np.sin(state[3]), state[5]]
        )

    def guidance_control(
        self, state: np.ndarray, desired_velocity: np.ndarray, dt: float
    ) -> np.ndarray:
        desired = np.asarray(desired_velocity, dtype=float)
        planar = desired[:2]
        surge = min(float(np.linalg.norm(planar)), self.max_surge_speed)
        heading = state[3] if surge < 1e-9 else float(np.arctan2(planar[1], planar[0]))
        heave = np.clip(desired[2], -self.max_heave_speed, self.max_heave_speed)
        return np.array(
            [
                np.clip((surge - state[4]) / dt + self.surge_drag * state[4], -self.max_surge_force, self.max_surge_force),
                np.clip((heave - state[5]) / dt + self.heave_drag * state[5], -self.max_heave_force, self.max_heave_force),
                np.clip(wrap_angle(heading - state[3]) / dt, -self.max_yaw_rate, self.max_yaw_rate),
            ]
        )

    def step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        value = np.asarray(state, dtype=float).copy()
        surge_force = np.clip(control[0], -self.max_surge_force, self.max_surge_force)
        heave_force = np.clip(control[1], -self.max_heave_force, self.max_heave_force)
        yaw_rate = np.clip(control[2], -self.max_yaw_rate, self.max_yaw_rate)
        value[4] = np.clip(
            value[4] + dt * (surge_force - self.surge_drag * value[4]),
            0.0,
            self.max_surge_speed,
        )
        value[5] = np.clip(
            value[5] + dt * (heave_force - self.heave_drag * value[5]),
            -self.max_heave_speed,
            self.max_heave_speed,
        )
        value[3] = wrap_angle(value[3] + dt * yaw_rate)
        value[0] += dt * value[4] * np.cos(value[3])
        value[1] += dt * value[4] * np.sin(value[3])
        value[2] += dt * value[5]
        return value


def make_platform_model(kind: str, parameters: dict[str, float]) -> PlatformModel:
    """Construct one platform model from YAML parameters."""
    models = {"ugv": UGVModel, "usv": USVModel, "rov": ROVModel}
    try:
        return models[kind.lower()](**parameters)
    except KeyError as exc:
        raise ValueError(f"unsupported harbor platform kind: {kind}") from exc
