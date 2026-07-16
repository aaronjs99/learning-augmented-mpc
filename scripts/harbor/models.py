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
    variant: str

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

    @abstractmethod
    def symbolic_step(self, ca, state, control, dt: float):
        """Return the matching CasADi state transition."""

    @abstractmethod
    def symbolic_velocity(self, ca, state):
        """Return velocity coordinates used by the MPC tracking cost."""

    @abstractmethod
    def control_scale(self) -> np.ndarray:
        """Return characteristic control magnitudes for cost normalization."""


@dataclass(frozen=True)
class ReducedUGVModel(PlatformModel):
    """Acceleration-controlled planar unicycle for a harbor ground vehicle."""

    max_speed: float = 1.0
    max_acceleration: float = 1.0
    max_yaw_rate: float = 1.2
    ground_z: float = 0.0
    kind: str = field(default="ugv", init=False)
    variant: str = field(default="reduced_unicycle", init=False)
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

    def symbolic_step(self, ca, state, control, dt: float):
        speed = state[3] + dt * control[0]
        heading = state[2] + dt * control[1]
        return ca.vertcat(
            state[0] + dt * speed * ca.cos(heading),
            state[1] + dt * speed * ca.sin(heading),
            heading,
            speed,
        )

    def symbolic_velocity(self, ca, state):
        return state[3]

    def control_scale(self) -> np.ndarray:
        return np.ones(2)


@dataclass(frozen=True)
class ReducedUSVModel(PlatformModel):
    """Planar surge-yaw surface-vessel model with linear drag."""

    max_speed: float = 1.0
    max_thrust: float = 1.5
    max_yaw_rate: float = 0.8
    drag: float = 0.25
    surface_z: float = 0.0
    kind: str = field(default="usv", init=False)
    variant: str = field(default="reduced_surge_yaw", init=False)
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

    def symbolic_step(self, ca, state, control, dt: float):
        speed = state[3] + dt * (control[0] - self.drag * state[3])
        heading = state[2] + dt * control[1]
        return ca.vertcat(
            state[0] + dt * speed * ca.cos(heading),
            state[1] + dt * speed * ca.sin(heading),
            heading,
            speed,
        )

    def symbolic_velocity(self, ca, state):
        return state[3]

    def control_scale(self) -> np.ndarray:
        return np.ones(2)


@dataclass(frozen=True)
class ReducedROVModel(PlatformModel):
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
    variant: str = field(default="reduced_world_damped", init=False)
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

    def symbolic_step(self, ca, state, control, dt: float):
        velocity = state[6:9] + dt * (
            control[:3] - self.linear_drag * state[6:9]
        )
        angular_rate = state[9:12] + dt * (
            control[3:6] - self.angular_drag * state[9:12]
        )
        return ca.vertcat(
            state[:3] + dt * velocity,
            state[3:6] + dt * angular_rate,
            velocity,
            angular_rate,
        )

    def symbolic_velocity(self, ca, state):
        return state[6:9]

    def control_scale(self) -> np.ndarray:
        return np.ones(6)


@dataclass(frozen=True)
class UGVModel(PlatformModel):
    """Low-speed kinematic bicycle with acceleration and steering inputs."""

    max_speed: float = 1.0
    max_reverse_speed: float = 0.5
    max_acceleration: float = 1.0
    wheelbase: float = 0.8
    max_steering_angle: float = 0.55
    position_gain: float = 1.0
    pose_heading_gain: float = 2.0
    pose_goal_heading_gain: float = -0.6
    ground_z: float = 0.0
    kind: str = field(default="ugv", init=False)
    variant: str = field(default="kinematic_bicycle", init=False)
    state_dim: int = field(default=4, init=False)
    control_dim: int = field(default=2, init=False)
    pose_dim: int = field(default=3, init=False)

    def __post_init__(self) -> None:
        positive = (
            self.max_speed,
            self.max_reverse_speed,
            self.max_acceleration,
            self.wheelbase,
            self.max_steering_angle,
            self.position_gain,
            self.pose_heading_gain,
        )
        if min(positive) <= 0.0 or self.max_steering_angle >= np.pi / 2.0:
            raise ValueError("UGV bicycle parameters must be positive and nonsingular")

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
        if desired_pose is None:
            target = np.asarray(state[:3], dtype=float)
            target[:2] += desired * dt
        else:
            target = np.asarray(desired_pose, dtype=float)
        delta = target[:2] - state[:2]
        distance = float(np.linalg.norm(delta))
        requested_speed = float(np.linalg.norm(desired))
        heading_to_goal = (
            float(np.arctan2(desired[1], desired[0]))
            if requested_speed > 1e-9
            else state[2]
            if distance < 1e-9
            else float(np.arctan2(delta[1], delta[0]))
        )
        alpha = wrap_angle(heading_to_goal - state[2])
        beta = wrap_angle(target[2] - state[2] - alpha)
        speed = self.position_gain * distance * np.cos(alpha)
        speed = np.clip(speed, -self.max_reverse_speed, self.max_speed)
        orientation_error = abs(wrap_angle(target[2] - state[2]))
        speed_cap = max(requested_speed, min(0.2, distance) if orientation_error > 0.05 else 0.0)
        speed = np.clip(speed, -speed_cap, speed_cap)
        desired_yaw_rate = (
            self.pose_heading_gain * alpha + self.pose_goal_heading_gain * beta
        )
        reference_speed = np.copysign(max(abs(speed), 0.08), speed or 1.0)
        steering = np.arctan(self.wheelbase * desired_yaw_rate / reference_speed)
        return np.array(
            [
                np.clip(
                    (speed - state[3]) / dt,
                    -self.max_acceleration,
                    self.max_acceleration,
                ),
                np.clip(
                    steering,
                    -self.max_steering_angle,
                    self.max_steering_angle,
                ),
            ]
        )

    def step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        value = np.asarray(state, dtype=float).copy()
        acceleration = np.clip(
            control[0], -self.max_acceleration, self.max_acceleration
        )
        steering = np.clip(
            control[1], -self.max_steering_angle, self.max_steering_angle
        )
        value[3] = np.clip(
            value[3] + dt * acceleration,
            -self.max_reverse_speed,
            self.max_speed,
        )
        yaw_rate = value[3] * np.tan(steering) / self.wheelbase
        value[2] = value[2] + dt * yaw_rate
        value[0] += dt * value[3] * np.cos(value[2])
        value[1] += dt * value[3] * np.sin(value[2])
        return value

    def symbolic_step(self, ca, state, control, dt: float):
        speed = state[3] + dt * control[0]
        yaw_rate = speed * ca.tan(control[1]) / self.wheelbase
        heading = state[2] + dt * yaw_rate
        return ca.vertcat(
            state[0] + dt * speed * ca.cos(heading),
            state[1] + dt * speed * ca.sin(heading),
            heading,
            speed,
        )

    def symbolic_velocity(self, ca, state):
        return state[3]

    def control_scale(self) -> np.ndarray:
        return np.array([self.max_acceleration, self.max_steering_angle])


@dataclass(frozen=True)
class SkidSteerUGVModel(PlatformModel):
    """Planar skid-steer dynamics driven by left/right side forces."""

    mass: float = 17.0
    yaw_inertia: float = 0.65
    max_speed: float = 2.0
    mission_speed: float | None = None
    max_reverse_speed: float = 1.0
    max_yaw_rate: float = 1.5
    max_force: float = 120.0
    max_yaw_moment: float = 35.0
    effective_track: float = 0.5
    drivetrain: str = "skid_steer"
    linear_drag: float = 8.0
    quadratic_drag: float = 8.0
    yaw_linear_drag: float = 1.0
    yaw_quadratic_drag: float = 0.5
    velocity_response_gain: float = 2.0
    heading_gain: float = 2.0
    yaw_response_gain: float = 2.5
    ground_z: float = 0.0
    kind: str = field(default="ugv", init=False)
    variant: str = field(default="dynamic_skid_steer", init=False)
    state_dim: int = field(default=5, init=False)
    control_dim: int = field(default=2, init=False)
    pose_dim: int = field(default=3, init=False)

    def __post_init__(self) -> None:
        positive = (
            self.mass,
            self.yaw_inertia,
            self.max_speed,
            self.max_reverse_speed,
            self.max_yaw_rate,
            self.max_force,
            self.max_yaw_moment,
            self.effective_track,
            self.velocity_response_gain,
            self.heading_gain,
            self.yaw_response_gain,
        )
        if min(positive) <= 0.0:
            raise ValueError("skid-steer UGV parameters must be positive")
        if self.mission_speed is not None and not 0.0 < self.mission_speed <= self.max_speed:
            raise ValueError("UGV mission_speed must be in (0, max_speed]")
        if min(
            self.linear_drag,
            self.quadratic_drag,
            self.yaw_linear_drag,
            self.yaw_quadratic_drag,
        ) < 0.0:
            raise ValueError("skid-steer UGV damping must be nonnegative")
        if not self.drivetrain.strip():
            raise ValueError("skid-steer UGV drivetrain must be named")
        if (
            self.max_yaw_moment
            > self.effective_track * self.max_side_force + 1e-12
        ):
            raise ValueError(
                "UGV max_yaw_moment exceeds the left/right drive-side authority"
            )

    @property
    def max_side_force(self) -> float:
        """Return the symmetric force bound for either drive side."""
        return self.max_force / 2.0

    def generalized_wrench(self, control: np.ndarray) -> np.ndarray:
        """Map left/right drive-side forces to surge force and yaw moment."""
        left, right = np.clip(
            np.asarray(control, dtype=float),
            -self.max_side_force,
            self.max_side_force,
        )
        return np.array(
            [left + right, 0.5 * self.effective_track * (right - left)]
        )

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
        requested_speed = min(float(np.linalg.norm(desired)), self.max_speed)
        if requested_speed > 1e-9:
            desired_heading = float(np.arctan2(desired[1], desired[0]))
        elif desired_pose is not None:
            desired_heading = float(desired_pose[2])
        else:
            desired_heading = float(state[2])
        heading_error = wrap_angle(desired_heading - state[2])
        signed_speed = requested_speed * max(0.0, np.cos(heading_error))
        desired_yaw_rate = np.clip(
            self.heading_gain * heading_error,
            -self.max_yaw_rate,
            self.max_yaw_rate,
        )
        force = (
            self.mass * self.velocity_response_gain * (signed_speed - state[3])
            + self.linear_drag * state[3]
            + self.quadratic_drag * abs(state[3]) * state[3]
        )
        moment = (
            self.yaw_inertia * self.yaw_response_gain * (desired_yaw_rate - state[4])
            + self.yaw_linear_drag * state[4]
            + self.yaw_quadratic_drag * abs(state[4]) * state[4]
        )
        force = np.clip(force, -self.max_force, self.max_force)
        moment = np.clip(moment, -self.max_yaw_moment, self.max_yaw_moment)
        left = 0.5 * force - moment / self.effective_track
        right = 0.5 * force + moment / self.effective_track
        return np.clip(
            np.array([left, right]), -self.max_side_force, self.max_side_force
        )

    def step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        value = np.asarray(state, dtype=float).copy()
        force, moment = self.generalized_wrench(control)
        speed = value[3] + dt * (
            force
            - self.linear_drag * value[3]
            - self.quadratic_drag * abs(value[3]) * value[3]
        ) / self.mass
        yaw_rate = value[4] + dt * (
            moment
            - self.yaw_linear_drag * value[4]
            - self.yaw_quadratic_drag * abs(value[4]) * value[4]
        ) / self.yaw_inertia
        value[3] = np.clip(speed, -self.max_reverse_speed, self.max_speed)
        value[4] = np.clip(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)
        value[2] += dt * value[4]
        value[0] += dt * value[3] * np.cos(value[2])
        value[1] += dt * value[3] * np.sin(value[2])
        return value

    def symbolic_step(self, ca, state, control, dt: float):
        force = control[0] + control[1]
        moment = 0.5 * self.effective_track * (control[1] - control[0])
        speed = state[3] + dt * (
            force
            - self.linear_drag * state[3]
            - self.quadratic_drag * ca.fabs(state[3]) * state[3]
        ) / self.mass
        yaw_rate = state[4] + dt * (
            moment
            - self.yaw_linear_drag * state[4]
            - self.yaw_quadratic_drag * ca.fabs(state[4]) * state[4]
        ) / self.yaw_inertia
        heading = state[2] + dt * yaw_rate
        return ca.vertcat(
            state[0] + dt * speed * ca.cos(heading),
            state[1] + dt * speed * ca.sin(heading),
            heading,
            speed,
            yaw_rate,
        )

    def symbolic_velocity(self, ca, state):
        return state[3:5]

    def control_scale(self) -> np.ndarray:
        return np.full(2, self.max_side_force)


@dataclass(frozen=True)
class USVModel(PlatformModel):
    """Underactuated 3-DOF marine model with twin waterjet inputs."""

    max_speed: float = 0.9
    mission_speed: float | None = None
    max_sway_speed: float = 0.45
    max_yaw_rate: float = 0.8
    max_thrust: float = 40.0
    max_yaw_moment: float = 12.0
    waterjet_separation: float = 0.74
    mass_diagonal: tuple[float, float, float] = (35.0, 45.0, 8.0)
    linear_damping: tuple[float, float, float] = (8.0, 12.0, 4.0)
    quadratic_damping: tuple[float, float, float] = (4.0, 8.0, 2.0)
    velocity_response_gain: float = 1.6
    heading_gain: float = 1.5
    yaw_response_gain: float = 2.0
    surface_z: float = 0.0
    kind: str = field(default="usv", init=False)
    variant: str = field(default="marine_3dof", init=False)
    state_dim: int = field(default=6, init=False)
    control_dim: int = field(default=2, init=False)
    pose_dim: int = field(default=3, init=False)

    def __post_init__(self) -> None:
        _freeze_vectors(self, "mass_diagonal", "linear_damping", "quadratic_damping")
        positive = (
            self.max_speed,
            self.max_sway_speed,
            self.max_yaw_rate,
            self.max_thrust,
            self.max_yaw_moment,
            self.waterjet_separation,
            self.velocity_response_gain,
            self.heading_gain,
            self.yaw_response_gain,
            *self.mass_diagonal,
        )
        if min(positive) <= 0.0:
            raise ValueError("USV marine parameters must be positive")
        if self.mission_speed is not None and not 0.0 < self.mission_speed <= self.max_speed:
            raise ValueError("USV mission_speed must be in (0, max_speed]")
        if min(self.linear_damping) < 0.0 or min(self.quadratic_damping) < 0.0:
            raise ValueError("USV damping must be nonnegative")
        if (
            self.max_yaw_moment
            > self.waterjet_separation * self.max_jet_thrust + 1e-12
        ):
            raise ValueError("USV max_yaw_moment exceeds twin-waterjet authority")

    @property
    def max_jet_thrust(self) -> float:
        """Return the symmetric thrust bound for either waterjet."""
        return self.max_thrust / 2.0

    def generalized_wrench(self, control: np.ndarray) -> np.ndarray:
        """Map port/starboard waterjet thrust to surge force and yaw moment."""
        port, starboard = np.clip(
            np.asarray(control, dtype=float),
            -self.max_jet_thrust,
            self.max_jet_thrust,
        )
        return np.array(
            [port + starboard, 0.5 * self.waterjet_separation * (starboard - port)]
        )

    def position(self, state: np.ndarray) -> np.ndarray:
        return np.array([state[0], state[1], self.surface_z], dtype=float)

    def velocity(self, state: np.ndarray) -> np.ndarray:
        yaw = state[2]
        return np.array(
            [
                np.cos(yaw) * state[3] - np.sin(yaw) * state[4],
                np.sin(yaw) * state[3] + np.cos(yaw) * state[4],
                0.0,
            ]
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
        desired_yaw_rate = np.clip(
            self.heading_gain * wrap_angle(heading - state[2]),
            -self.max_yaw_rate,
            self.max_yaw_rate,
        )
        nu = np.asarray(state[3:6], dtype=float)
        coriolis = _marine_coriolis_product(nu, self.mass_diagonal)
        damping = _damping_product(
            nu, self.linear_damping, self.quadratic_damping
        )
        thrust = (
            self.mass_diagonal[0] * self.velocity_response_gain * (speed - nu[0])
            + coriolis[0]
            + damping[0]
        )
        moment = (
            self.mass_diagonal[2]
            * self.yaw_response_gain
            * (desired_yaw_rate - nu[2])
            + coriolis[2]
            + damping[2]
        )
        thrust = np.clip(thrust, -self.max_thrust, self.max_thrust)
        moment = np.clip(moment, -self.max_yaw_moment, self.max_yaw_moment)
        port = 0.5 * thrust - moment / self.waterjet_separation
        starboard = 0.5 * thrust + moment / self.waterjet_separation
        return np.clip(
            np.array([port, starboard]), -self.max_jet_thrust, self.max_jet_thrust
        )

    def step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        value = np.asarray(state, dtype=float).copy()
        nu = value[3:6]
        thrust, moment = self.generalized_wrench(control)
        tau = np.array([thrust, 0.0, moment])
        rhs = tau - _marine_coriolis_product(
            nu, self.mass_diagonal
        ) - _damping_product(nu, self.linear_damping, self.quadratic_damping)
        nu = nu + dt * rhs / np.asarray(self.mass_diagonal)
        nu[0] = np.clip(nu[0], 0.0, self.max_speed)
        nu[1] = np.clip(nu[1], -self.max_sway_speed, self.max_sway_speed)
        nu[2] = np.clip(nu[2], -self.max_yaw_rate, self.max_yaw_rate)
        yaw = value[2]
        value[0] += dt * (np.cos(yaw) * nu[0] - np.sin(yaw) * nu[1])
        value[1] += dt * (np.sin(yaw) * nu[0] + np.cos(yaw) * nu[1])
        value[2] = yaw + dt * nu[2]
        value[3:6] = nu
        return value

    def symbolic_step(self, ca, state, control, dt: float):
        nu = state[3:6]
        mass = ca.DM(self.mass_diagonal)
        coriolis = _symbolic_marine_coriolis_product(ca, nu, mass)
        damping = _symbolic_damping_product(
            ca, nu, self.linear_damping, self.quadratic_damping
        )
        thrust = control[0] + control[1]
        moment = 0.5 * self.waterjet_separation * (control[1] - control[0])
        tau = ca.vertcat(thrust, 0.0, moment)
        next_nu = nu + dt * ca.rdivide(tau - coriolis - damping, mass)
        yaw = state[2]
        return ca.vertcat(
            state[0]
            + dt * (ca.cos(yaw) * next_nu[0] - ca.sin(yaw) * next_nu[1]),
            state[1]
            + dt * (ca.sin(yaw) * next_nu[0] + ca.cos(yaw) * next_nu[1]),
            yaw + dt * next_nu[2],
            next_nu,
        )

    def symbolic_velocity(self, ca, state):
        return state[3:6]

    def control_scale(self) -> np.ndarray:
        return np.full(2, self.max_jet_thrust)


@dataclass(frozen=True)
class ROVModel(PlatformModel):
    """Body-frame 6-DOF marine craft with eight-thruster allocation."""

    max_horizontal_speed: float = 0.75
    mission_speed: float | None = None
    max_vertical_speed: float = 0.4
    max_angular_rate: float = 0.7
    max_force: float = 30.0
    max_torque: float = 8.0
    force_limits: tuple[float, float, float] | None = None
    torque_limits: tuple[float, float, float] | None = None
    thruster_limits: tuple[float, ...] | None = None
    thruster_allocation: tuple[tuple[float, ...], ...] | None = None
    mass_diagonal: tuple[float, ...] = (18.0, 18.0, 22.0, 1.2, 1.4, 1.5)
    linear_damping: tuple[float, ...] = (8.0, 10.0, 12.0, 1.2, 1.4, 1.5)
    quadratic_damping: tuple[float, ...] = (12.0, 16.0, 18.0, 0.6, 0.8, 0.8)
    weight: float = 176.58
    buoyancy: float = 176.58
    center_of_gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    center_of_buoyancy: tuple[float, float, float] = (0.0, 0.0, 0.03)
    velocity_response_gain: float = 1.8
    attitude_response_gain: float = 1.0
    angular_rate_response_gain: float = 2.0
    control_smoothing: float = 0.6
    kind: str = field(default="rov", init=False)
    variant: str = field(default="marine_6dof", init=False)
    state_dim: int = field(default=12, init=False)
    control_dim: int = field(default=8, init=False)
    pose_dim: int = field(default=6, init=False)

    def __post_init__(self) -> None:
        _freeze_vectors(
            self,
            "mass_diagonal",
            "linear_damping",
            "quadratic_damping",
            "center_of_gravity",
            "center_of_buoyancy",
        )
        if self.force_limits is not None:
            _freeze_vectors(self, "force_limits")
        if self.torque_limits is not None:
            _freeze_vectors(self, "torque_limits")
        if self.thruster_limits is not None:
            _freeze_vectors(self, "thruster_limits")
        if self.thruster_allocation is not None:
            object.__setattr__(
                self,
                "thruster_allocation",
                tuple(
                    tuple(float(value) for value in row)
                    for row in self.thruster_allocation
                ),
            )
        if not (
            len(self.mass_diagonal)
            == len(self.linear_damping)
            == len(self.quadratic_damping)
            == 6
        ):
            raise ValueError("ROV mass and damping vectors must have six entries")
        if len(self.center_of_gravity) != 3 or len(self.center_of_buoyancy) != 3:
            raise ValueError("ROV mass and buoyancy centers must have three entries")
        if len(self.force_limit_vector) != 3 or len(self.torque_limit_vector) != 3:
            raise ValueError("ROV force and torque limits must have three entries")
        if len(self.thruster_limit_vector) != 8:
            raise ValueError("ROV thruster_limits must have eight entries")
        if self.allocation_matrix.shape != (6, self.control_dim):
            raise ValueError("ROV thruster_allocation must have shape 6 x 8")
        if np.linalg.matrix_rank(self.allocation_matrix) != 6:
            raise ValueError("ROV thruster_allocation must have full row rank")
        positive = (
            self.max_horizontal_speed,
            self.max_vertical_speed,
            self.max_angular_rate,
            self.max_force,
            self.max_torque,
            self.weight,
            self.buoyancy,
            self.velocity_response_gain,
            self.attitude_response_gain,
            self.angular_rate_response_gain,
            self.control_smoothing,
            *self.mass_diagonal,
            *self.force_limit_vector,
            *self.torque_limit_vector,
            *self.thruster_limit_vector,
        )
        if min(positive) <= 0.0 or self.control_smoothing > 1.0:
            raise ValueError("ROV marine parameters must be positive and bounded")
        if (
            self.mission_speed is not None
            and not 0.0 < self.mission_speed <= self.max_horizontal_speed
        ):
            raise ValueError("ROV mission_speed must be in (0, max_horizontal_speed]")
        if min(self.linear_damping) < 0.0 or min(self.quadratic_damping) < 0.0:
            raise ValueError("ROV damping must be nonnegative")

    @property
    def force_limit_vector(self) -> np.ndarray:
        """Return axis-specific body-force limits, or the scalar fallback."""
        return np.asarray(
            self.force_limits
            if self.force_limits is not None
            else (self.max_force,) * 3,
            dtype=float,
        )

    @property
    def torque_limit_vector(self) -> np.ndarray:
        """Return axis-specific body-moment limits, or the scalar fallback."""
        return np.asarray(
            self.torque_limits
            if self.torque_limits is not None
            else (self.max_torque,) * 3,
            dtype=float,
        )

    @property
    def thruster_limit_vector(self) -> np.ndarray:
        """Return the eight symmetric physical thruster-force bounds."""
        if self.thruster_limits is not None:
            return np.asarray(self.thruster_limits, dtype=float)
        horizontal = self.force_limit_vector[0] / (2.0 * np.sqrt(2.0))
        vertical = self.force_limit_vector[2] / 4.0
        return np.array([horizontal] * 4 + [vertical] * 4)

    @property
    def allocation_matrix(self) -> np.ndarray:
        """Return the calibrated BlueROV2 Heavy 6x8 thruster allocation."""
        if self.thruster_allocation is not None:
            return np.asarray(self.thruster_allocation, dtype=float)
        horizontal_total = float(np.sum(self.thruster_limit_vector[:4]))
        vertical_total = float(np.sum(self.thruster_limit_vector[4:]))
        x_gain = self.force_limit_vector[0] / horizontal_total
        y_gain = self.force_limit_vector[1] / horizontal_total
        z_gain = self.force_limit_vector[2] / vertical_total
        roll_gain = self.torque_limit_vector[0] / vertical_total
        pitch_gain = self.torque_limit_vector[1] / vertical_total
        yaw_gain = self.torque_limit_vector[2] / horizontal_total
        return np.array(
            [
                [-x_gain, -x_gain, x_gain, x_gain, 0.0, 0.0, 0.0, 0.0],
                [y_gain, -y_gain, y_gain, -y_gain, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, z_gain, z_gain, z_gain, z_gain],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    roll_gain,
                    -roll_gain,
                    roll_gain,
                    -roll_gain,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -pitch_gain,
                    -pitch_gain,
                    pitch_gain,
                    pitch_gain,
                ],
                [yaw_gain, -yaw_gain, -yaw_gain, yaw_gain, 0.0, 0.0, 0.0, 0.0],
            ]
        )

    def generalized_wrench(self, control: np.ndarray) -> np.ndarray:
        """Map eight bounded T200 forces to body force and moment."""
        bounded = np.clip(
            np.asarray(control, dtype=float),
            -self.thruster_limit_vector,
            self.thruster_limit_vector,
        )
        return self.allocation_matrix @ bounded

    def allocate_wrench(self, wrench: np.ndarray) -> np.ndarray:
        """Allocate a desired body wrench and uniformly desaturate thrusters."""
        command = np.linalg.pinv(self.allocation_matrix) @ np.asarray(
            wrench, dtype=float
        )
        ratio = float(
            np.max(np.abs(command) / self.thruster_limit_vector, initial=0.0)
        )
        if ratio > 1.0:
            command /= ratio
        return command

    def position(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state[:3], dtype=float).copy()

    def velocity(self, state: np.ndarray) -> np.ndarray:
        return _rotation_matrix(state[3:6]) @ np.asarray(state[6:9], dtype=float)

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
        desired_world = np.asarray(desired_velocity, dtype=float).copy()
        planar_speed = np.linalg.norm(desired_world[:2])
        if planar_speed > self.max_horizontal_speed:
            desired_world[:2] *= self.max_horizontal_speed / planar_speed
        desired_world[2] = np.clip(
            desired_world[2], -self.max_vertical_speed, self.max_vertical_speed
        )
        rotation = _rotation_matrix(state[3:6])
        desired_body = rotation.T @ desired_world
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
        nu = np.asarray(state[6:12], dtype=float)
        desired_nu = np.concatenate((desired_body, desired_rates))
        acceleration = np.concatenate(
            (
                self.velocity_response_gain * (desired_nu[:3] - nu[:3]),
                self.angular_rate_response_gain * (desired_nu[3:] - nu[3:]),
            )
        )
        tau = (
            np.asarray(self.mass_diagonal) * acceleration
            + _marine_coriolis_product(nu, self.mass_diagonal)
            + _damping_product(nu, self.linear_damping, self.quadratic_damping)
            + _restoring_vector(
                state[3:6],
                self.weight,
                self.buoyancy,
                self.center_of_gravity,
                self.center_of_buoyancy,
            )
        )
        tau[:3] = np.clip(
            tau[:3], -self.force_limit_vector, self.force_limit_vector
        )
        tau[3:] = np.clip(
            tau[3:], -self.torque_limit_vector, self.torque_limit_vector
        )
        return self.allocate_wrench(tau)

    def step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        value = np.asarray(state, dtype=float).copy()
        tau = self.generalized_wrench(control)
        nu = value[6:12]
        rhs = (
            tau
            - _marine_coriolis_product(nu, self.mass_diagonal)
            - _damping_product(nu, self.linear_damping, self.quadratic_damping)
            - _restoring_vector(
                value[3:6],
                self.weight,
                self.buoyancy,
                self.center_of_gravity,
                self.center_of_buoyancy,
            )
        )
        nu = nu + dt * rhs / np.asarray(self.mass_diagonal)
        horizontal_speed = np.linalg.norm(nu[:2])
        if horizontal_speed > self.max_horizontal_speed:
            nu[:2] *= self.max_horizontal_speed / horizontal_speed
        nu[2] = np.clip(nu[2], -self.max_vertical_speed, self.max_vertical_speed)
        nu[3:] = np.clip(
            nu[3:], -self.max_angular_rate, self.max_angular_rate
        )
        value[:3] += dt * (_rotation_matrix(value[3:6]) @ nu[:3])
        value[3:6] = value[3:6] + dt * (
            _euler_rate_matrix(value[3:6]) @ nu[3:]
        )
        value[6:12] = nu
        return value

    def symbolic_step(self, ca, state, control, dt: float):
        nu = state[6:12]
        mass = ca.DM(self.mass_diagonal)
        coriolis = _symbolic_marine_coriolis_product(ca, nu, mass)
        damping = _symbolic_damping_product(
            ca, nu, self.linear_damping, self.quadratic_damping
        )
        restoring = _symbolic_restoring_vector(ca, state[3:6], self)
        tau = ca.DM(self.allocation_matrix) @ control
        next_nu = nu + dt * ca.rdivide(tau - coriolis - damping - restoring, mass)
        rotation = _symbolic_rotation_matrix(ca, state[3:6])
        euler_rates = _symbolic_euler_rate_matrix(ca, state[3:6]) @ next_nu[3:]
        return ca.vertcat(
            state[:3] + dt * (rotation @ next_nu[:3]),
            state[3:6] + dt * euler_rates,
            next_nu,
        )

    def symbolic_velocity(self, ca, state):
        return state[6:12]

    def control_scale(self) -> np.ndarray:
        return self.thruster_limit_vector


def make_platform_model(kind: str, parameters: dict[str, float]) -> PlatformModel:
    """Construct one platform model from YAML parameters."""
    parameters = dict(parameters)
    requested = str(parameters.pop("model", "physical")).lower()
    models = {
        ("ugv", "physical"): UGVModel,
        ("ugv", "kinematic_bicycle"): UGVModel,
        ("ugv", "dynamic_skid_steer"): SkidSteerUGVModel,
        ("ugv", "reduced"): ReducedUGVModel,
        ("ugv", "reduced_unicycle"): ReducedUGVModel,
        ("usv", "physical"): USVModel,
        ("usv", "marine_3dof"): USVModel,
        ("usv", "reduced"): ReducedUSVModel,
        ("usv", "reduced_surge_yaw"): ReducedUSVModel,
        ("rov", "physical"): ROVModel,
        ("rov", "marine_6dof"): ROVModel,
        ("rov", "reduced"): ReducedROVModel,
        ("rov", "reduced_world_damped"): ReducedROVModel,
    }
    try:
        return models[(kind.lower(), requested)](**parameters)
    except KeyError as exc:
        raise ValueError(
            f"unsupported harbor platform model: {kind}/{requested}"
        ) from exc


def _freeze_vectors(instance, *names: str) -> None:
    for name in names:
        object.__setattr__(
            instance,
            name,
            tuple(float(value) for value in getattr(instance, name)),
        )


def _rotation_matrix(angles: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(angles, dtype=float)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _euler_rate_matrix(angles: np.ndarray) -> np.ndarray:
    roll, pitch, _ = np.asarray(angles, dtype=float)
    cr, sr = np.cos(roll), np.sin(roll)
    cp = np.cos(pitch)
    cp = np.copysign(max(abs(cp), 1e-6), cp)
    tp = np.sin(pitch) / cp
    return np.array(
        [[1.0, sr * tp, cr * tp], [0.0, cr, -sr], [0.0, sr / cp, cr / cp]]
    )


def _marine_coriolis_product(
    velocity: np.ndarray, mass_diagonal: tuple[float, ...]
) -> np.ndarray:
    nu = np.asarray(velocity, dtype=float)
    mass = np.asarray(mass_diagonal, dtype=float)
    if len(nu) == 3:
        surge, sway, yaw_rate = nu
        return np.array(
            [
                -mass[1] * sway * yaw_rate,
                mass[0] * surge * yaw_rate,
                (mass[1] - mass[0]) * surge * sway,
            ]
        )
    linear, angular = nu[:3], nu[3:]
    linear_momentum = mass[:3] * linear
    angular_momentum = mass[3:] * angular
    return np.concatenate(
        (
            np.cross(angular, linear_momentum),
            np.cross(linear, linear_momentum)
            + np.cross(angular, angular_momentum),
        )
    )


def _damping_product(
    velocity: np.ndarray,
    linear: tuple[float, ...],
    quadratic: tuple[float, ...],
) -> np.ndarray:
    nu = np.asarray(velocity, dtype=float)
    return np.asarray(linear) * nu + np.asarray(quadratic) * np.abs(nu) * nu


def _restoring_vector(
    angles: np.ndarray,
    weight: float,
    buoyancy: float,
    center_of_gravity: tuple[float, float, float],
    center_of_buoyancy: tuple[float, float, float],
) -> np.ndarray:
    rotation = _rotation_matrix(angles)
    gravity = rotation.T @ np.array([0.0, 0.0, -weight])
    lift = rotation.T @ np.array([0.0, 0.0, buoyancy])
    external = np.concatenate(
        (
            gravity + lift,
            np.cross(center_of_gravity, gravity)
            + np.cross(center_of_buoyancy, lift),
        )
    )
    return -external


def _symbolic_marine_coriolis_product(ca, velocity, mass):
    if int(velocity.numel()) == 3:
        return ca.vertcat(
            -mass[1] * velocity[1] * velocity[2],
            mass[0] * velocity[0] * velocity[2],
            (mass[1] - mass[0]) * velocity[0] * velocity[1],
        )
    linear, angular = velocity[:3], velocity[3:]
    linear_momentum = ca.times(mass[:3], linear)
    angular_momentum = ca.times(mass[3:], angular)
    return ca.vertcat(
        ca.cross(angular, linear_momentum),
        ca.cross(linear, linear_momentum)
        + ca.cross(angular, angular_momentum),
    )


def _symbolic_damping_product(ca, velocity, linear, quadratic):
    return ca.times(ca.DM(linear), velocity) + ca.times(
        ca.DM(quadratic), ca.times(ca.fabs(velocity), velocity)
    )


def _symbolic_rotation_matrix(ca, angles):
    roll, pitch, yaw = angles[0], angles[1], angles[2]
    cr, sr = ca.cos(roll), ca.sin(roll)
    cp, sp = ca.cos(pitch), ca.sin(pitch)
    cy, sy = ca.cos(yaw), ca.sin(yaw)
    return ca.vertcat(
        ca.horzcat(cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        ca.horzcat(sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        ca.horzcat(-sp, cp * sr, cp * cr),
    )


def _symbolic_euler_rate_matrix(ca, angles):
    roll, pitch = angles[0], angles[1]
    cr, sr = ca.cos(roll), ca.sin(roll)
    cp = ca.cos(pitch)
    tp = ca.tan(pitch)
    return ca.vertcat(
        ca.horzcat(1.0, sr * tp, cr * tp),
        ca.horzcat(0.0, cr, -sr),
        ca.horzcat(0.0, sr / cp, cr / cp),
    )


def _symbolic_restoring_vector(ca, angles, model: ROVModel):
    rotation = _symbolic_rotation_matrix(ca, angles)
    gravity = rotation.T @ ca.DM([0.0, 0.0, -model.weight])
    lift = rotation.T @ ca.DM([0.0, 0.0, model.buoyancy])
    moment = ca.cross(ca.DM(model.center_of_gravity), gravity) + ca.cross(
        ca.DM(model.center_of_buoyancy), lift
    )
    return -ca.vertcat(gravity + lift, moment)
