"""Artificial-potential-field initializer and backup control for manta runs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from scripts.dynamics import MantaDynamicsConfig, rk4_step_np, wrap_angle
from scripts.simulation import StaticObstacle


@dataclass(frozen=True)
class APFConfig:
    """APF tuning loaded from ``config/manta.yaml``."""

    heading_gain: float = 1.5
    influence_radius: float = 4.0
    obstacle_padding: float = 0.2
    repulsion_gain: float = 1.5
    goal_tolerance: float = 0.35
    max_steps: int = 400
    base_mu_gain: float = 0.8
    base_mu_min: float = 0.5
    base_mu_max: float = 1.5

    def __post_init__(self) -> None:
        """Validate APF tuning independently of a simulation run."""
        if self.max_steps <= 0:
            raise ValueError("apf.max_steps must be positive")
        if self.influence_radius <= 0.0:
            raise ValueError("apf.influence_radius must be positive")
        if self.goal_tolerance <= 0.0:
            raise ValueError("apf.goal_tolerance must be positive")
        if self.heading_gain < 0.0 or self.repulsion_gain < 0.0:
            raise ValueError("apf heading and repulsion gains must be nonnegative")
        if self.obstacle_padding < 0.0:
            raise ValueError("apf.obstacle_padding must be nonnegative")
        if self.base_mu_gain <= 0.0:
            raise ValueError("apf.base_mu_gain must be positive")
        if not 0.0 <= self.base_mu_min <= self.base_mu_max:
            raise ValueError("apf base_mu bounds must satisfy 0 <= min <= max")


def compute_apf_control(
    current_state: np.ndarray,
    goal_state: np.ndarray,
    *,
    obstacle: StaticObstacle,
    extra_obstacles: Sequence[StaticObstacle] | None = None,
    apf_config: APFConfig = APFConfig(),
    dynamics_config: MantaDynamicsConfig = MantaDynamicsConfig(),
) -> np.ndarray:
    """Return one obstacle-aware APF control for the current manta state."""
    x, y, theta = np.asarray(current_state, dtype=float)[:3]

    goal_delta = np.asarray(goal_state[:2], dtype=float) - np.array([x, y])
    goal_dist = float(np.linalg.norm(goal_delta))
    attraction = goal_delta / (goal_dist + 1e-5)
    base_mu = np.clip(
        apf_config.base_mu_gain * goal_dist,
        apf_config.base_mu_min,
        apf_config.base_mu_max,
    )

    repulsion = np.zeros(2, dtype=float)
    for static_obstacle in (obstacle, *(extra_obstacles or ())):
        obs = np.asarray(static_obstacle.center, dtype=float)
        obs_delta = np.array([x, y]) - obs
        obs_dist = float(np.linalg.norm(obs_delta))
        if obs_dist < apf_config.influence_radius:
            surface_dist = obs_dist - (
                static_obstacle.radius + apf_config.obstacle_padding
            )
            surface_dist = max(surface_dist, 0.05)
            denominator = max(
                apf_config.influence_radius - static_obstacle.radius, 0.05
            )
            strength = apf_config.repulsion_gain * (
                1.0 / surface_dist - 1.0 / denominator
            )
            repulsion += (obs_delta / (obs_dist + 1e-9)) * strength

    desired_velocity = attraction + repulsion
    target_theta = np.arctan2(desired_velocity[1], desired_velocity[0])
    theta_error = float(wrap_angle(target_theta - theta))

    delta_mu = apf_config.heading_gain * theta_error
    return np.array(
        [
            np.clip(base_mu + delta_mu, dynamics_config.mu_min, dynamics_config.mu_max),
            np.clip(base_mu - delta_mu, dynamics_config.mu_min, dynamics_config.mu_max),
        ],
        dtype=float,
    )


# Backward-compatible name for callers that treat the APF step as a backup.
backup_apf_control = compute_apf_control


def simulate_manta_autopilot(
    start_state: np.ndarray,
    goal_state: np.ndarray,
    *,
    dt: float = 0.2,
    obstacle: StaticObstacle = StaticObstacle(center=(3.1, 2.9), radius=0.95),
    extra_obstacles: Sequence[StaticObstacle] | None = None,
    apf_config: APFConfig = APFConfig(),
    dynamics_config: MantaDynamicsConfig = MantaDynamicsConfig(),
) -> np.ndarray:
    """Generate one APF trajectory from ``start_state`` toward ``goal_state``."""
    history, _ = simulate_manta_autopilot_with_controls(
        start_state,
        goal_state,
        dt=dt,
        obstacle=obstacle,
        extra_obstacles=extra_obstacles,
        apf_config=apf_config,
        dynamics_config=dynamics_config,
    )
    return history


def simulate_manta_autopilot_with_controls(
    start_state: np.ndarray,
    goal_state: np.ndarray,
    *,
    dt: float = 0.2,
    obstacle: StaticObstacle = StaticObstacle(center=(3.1, 2.9), radius=0.95),
    extra_obstacles: Sequence[StaticObstacle] | None = None,
    apf_config: APFConfig = APFConfig(),
    dynamics_config: MantaDynamicsConfig = MantaDynamicsConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one APF trajectory and the controls that produced it."""
    history = [np.asarray(start_state, dtype=float).copy()]
    controls: list[np.ndarray] = []
    current_state = history[0].copy()

    for _ in range(apf_config.max_steps):
        control = compute_apf_control(
            current_state=current_state,
            goal_state=goal_state,
            obstacle=obstacle,
            extra_obstacles=extra_obstacles,
            apf_config=apf_config,
            dynamics_config=dynamics_config,
        )

        controls.append(control.copy())
        current_state = rk4_step_np(current_state, control, dt, dynamics_config)
        history.append(current_state.copy())

        if (
            np.linalg.norm(current_state[:2] - goal_state[:2])
            < apf_config.goal_tolerance
        ):
            break

    return np.asarray(history, dtype=float), np.asarray(controls, dtype=float)
