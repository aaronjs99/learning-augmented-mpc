"""Manta-inspired CPG dynamics shared by APF, simulation, and LMPC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MantaDynamicsConfig:
    """Constants for the 7-state manta/CPG model loaded from YAML.

    State is ``[x, y, theta, p_L, q_L, p_R, q_R]`` and control is
    ``[mu_L, mu_R]``.
    """

    c_v: float = 0.3
    c_tau: float = 0.2
    wing_span: float = 1.0
    oscillator_frequency: float = 4.0
    rk4_substeps: int = 4
    mu_min: float = 0.0
    mu_max: float = 2.5
    oscillator_bound: float = 6.0


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap an angle to ``[-pi, pi]``."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def state_derivative_np(
    state: np.ndarray,
    control: np.ndarray,
    config: MantaDynamicsConfig = MantaDynamicsConfig(),
) -> np.ndarray:
    """Return the continuous-time derivative for one manta state."""
    x, y, theta, p_l, q_l, p_r, q_r = np.asarray(state, dtype=float)
    del x, y
    mu_l, mu_r = np.asarray(control, dtype=float)

    p_l_dot = (mu_l - p_l**2 - q_l**2) * p_l - config.oscillator_frequency * q_l
    q_l_dot = (mu_l - p_l**2 - q_l**2) * q_l + config.oscillator_frequency * p_l
    p_r_dot = (mu_r - p_r**2 - q_r**2) * p_r - config.oscillator_frequency * q_r
    q_r_dot = (mu_r - p_r**2 - q_r**2) * q_r + config.oscillator_frequency * p_r

    v = config.c_v * (mu_l + mu_r)
    theta_dot = (config.c_tau / config.wing_span) * (mu_l - mu_r)

    return np.array(
        [
            v * np.cos(theta),
            v * np.sin(theta),
            theta_dot,
            p_l_dot,
            q_l_dot,
            p_r_dot,
            q_r_dot,
        ],
        dtype=float,
    )


def rk4_step_np(
    state: np.ndarray,
    control: np.ndarray,
    dt: float,
    config: MantaDynamicsConfig = MantaDynamicsConfig(),
) -> np.ndarray:
    """Advance one manta state with RK4 integration."""
    x_curr = np.asarray(state, dtype=float).copy()
    u = np.clip(np.asarray(control, dtype=float), config.mu_min, config.mu_max)
    dt_sub = dt / config.rk4_substeps

    for _ in range(config.rk4_substeps):
        k1 = state_derivative_np(x_curr, u, config)
        k2 = state_derivative_np(x_curr + 0.5 * dt_sub * k1, u, config)
        k3 = state_derivative_np(x_curr + 0.5 * dt_sub * k2, u, config)
        k4 = state_derivative_np(x_curr + dt_sub * k3, u, config)
        x_curr = x_curr + (dt_sub / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    x_curr[2] = float(wrap_angle(x_curr[2]))
    return x_curr


def state_derivative_ca(
    state: Any,
    control: Any,
    config: MantaDynamicsConfig = MantaDynamicsConfig(),
) -> Any:
    """Return the CasADi symbolic derivative for one manta state."""
    import casadi as ca

    theta = state[2]
    p_l, q_l = state[3], state[4]
    p_r, q_r = state[5], state[6]
    mu_l, mu_r = control[0], control[1]

    p_l_dot = (mu_l - p_l**2 - q_l**2) * p_l - config.oscillator_frequency * q_l
    q_l_dot = (mu_l - p_l**2 - q_l**2) * q_l + config.oscillator_frequency * p_l
    p_r_dot = (mu_r - p_r**2 - q_r**2) * p_r - config.oscillator_frequency * q_r
    q_r_dot = (mu_r - p_r**2 - q_r**2) * q_r + config.oscillator_frequency * p_r

    v = config.c_v * (mu_l + mu_r)
    theta_dot = (config.c_tau / config.wing_span) * (mu_l - mu_r)

    return ca.vertcat(
        v * ca.cos(theta),
        v * ca.sin(theta),
        theta_dot,
        p_l_dot,
        q_l_dot,
        p_r_dot,
        q_r_dot,
    )


def rk4_step_ca(
    state: Any,
    control: Any,
    dt: float,
    config: MantaDynamicsConfig = MantaDynamicsConfig(),
) -> Any:
    """Advance one CasADi symbolic manta state with RK4 integration."""
    x_curr = state
    dt_sub = dt / config.rk4_substeps

    for _ in range(config.rk4_substeps):
        k1 = state_derivative_ca(x_curr, control, config)
        k2 = state_derivative_ca(x_curr + 0.5 * dt_sub * k1, control, config)
        k3 = state_derivative_ca(x_curr + 0.5 * dt_sub * k2, control, config)
        k4 = state_derivative_ca(x_curr + dt_sub * k3, control, config)
        x_curr = x_curr + (dt_sub / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return x_curr
