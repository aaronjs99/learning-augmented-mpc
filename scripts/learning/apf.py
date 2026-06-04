"""Artificial-potential-field initializer for iteration-0 manta safe sets."""

from __future__ import annotations

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


def simulate_manta_autopilot(
    start_state: np.ndarray,
    goal_state: np.ndarray,
    *,
    dt: float = 0.2,
    obstacle: StaticObstacle = StaticObstacle(center=(3.1, 2.9), radius=0.95),
    apf_config: APFConfig = APFConfig(),
    dynamics_config: MantaDynamicsConfig = MantaDynamicsConfig(),
) -> np.ndarray:
    """Generate one solo trajectory with obstacle-aware APF steering."""
    history = [np.asarray(start_state, dtype=float).copy()]
    current_state = history[0].copy()
    obs = np.asarray(obstacle.center, dtype=float)

    for _ in range(apf_config.max_steps):
        x, y, theta = current_state[:3]

        goal_delta = np.asarray(goal_state[:2], dtype=float) - np.array([x, y])
        goal_dist = float(np.linalg.norm(goal_delta))
        attraction = goal_delta / (goal_dist + 1e-5)
        base_mu = np.clip(
            apf_config.base_mu_gain * goal_dist,
            apf_config.base_mu_min,
            apf_config.base_mu_max,
        )

        obs_delta = np.array([x, y]) - obs
        obs_dist = float(np.linalg.norm(obs_delta))
        repulsion = np.zeros(2, dtype=float)
        if obs_dist < apf_config.influence_radius:
            surface_dist = obs_dist - (obstacle.radius + apf_config.obstacle_padding)
            surface_dist = max(surface_dist, 0.05)
            denominator = apf_config.influence_radius - obstacle.radius
            strength = apf_config.repulsion_gain * (
                1.0 / surface_dist - 1.0 / denominator
            )
            repulsion = (obs_delta / (obs_dist + 1e-9)) * strength

        desired_velocity = attraction + repulsion
        target_theta = np.arctan2(desired_velocity[1], desired_velocity[0])
        theta_error = float(wrap_angle(target_theta - theta))

        delta_mu = apf_config.heading_gain * theta_error
        control = np.array(
            [
                np.clip(
                    base_mu + delta_mu, dynamics_config.mu_min, dynamics_config.mu_max
                ),
                np.clip(
                    base_mu - delta_mu, dynamics_config.mu_min, dynamics_config.mu_max
                ),
            ],
            dtype=float,
        )

        current_state = rk4_step_np(current_state, control, dt, dynamics_config)
        history.append(current_state.copy())

        if (
            np.linalg.norm(current_state[:2] - goal_state[:2])
            < apf_config.goal_tolerance
        ):
            break

    return np.asarray(history, dtype=float)
