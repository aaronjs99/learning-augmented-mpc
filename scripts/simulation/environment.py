"""Multi-agent manta simulation environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.dynamics import MantaDynamicsConfig, rk4_step_np


@dataclass(frozen=True)
class MantaEnvConfig:
    """Configuration for the active multi-agent manta environment."""

    dt: float = 0.2
    horizon: int = 300
    dynamics: MantaDynamicsConfig = MantaDynamicsConfig()

    def __post_init__(self) -> None:
        """Validate integration settings before allocating rollout arrays."""
        if self.dt <= 0.0:
            raise ValueError("manta environment dt must be positive")
        if self.horizon <= 0:
            raise ValueError("manta environment horizon must be positive")


class MultiMantaRayEnv:
    """Stateful multi-agent manta/CPG simulation environment.

    State uses shape ``(A, 7)`` with rows
    ``[x, y, theta, p_L, q_L, p_R, q_R]``. Controls use shape ``(A, 2)`` as
    ``[mu_L, mu_R]`` and advance with the shared RK4 manta dynamics.
    """

    def __init__(
        self, starts: np.ndarray, goals: np.ndarray, config: MantaEnvConfig
    ) -> None:
        """Initialize environment with fixed starts/goals and simulation config."""
        self.starts = _as_manta_array(starts)
        self.goals = _as_manta_array(goals)
        if self.goals.shape != self.starts.shape:
            raise ValueError(
                "manta starts and goals must have the same shape, got "
                f"{self.starts.shape} and {self.goals.shape}"
            )
        self.config = config
        self._state = self.starts.copy()
        self._step_idx = 0

    @property
    def state(self) -> np.ndarray:
        """Return current state array with shape ``(A, 7)``."""
        return self._state.copy()

    @property
    def num_agents(self) -> int:
        """Return the number of independently controlled manta agents."""
        return self.starts.shape[0]

    @property
    def step_index(self) -> int:
        """Return current simulation step index."""
        return self._step_idx

    def reset(self) -> np.ndarray:
        """Reset state to starts and return the reset state."""
        self._state = self.starts.copy()
        self._step_idx = 0
        return self.state

    def step(self, control: np.ndarray | None = None) -> np.ndarray:
        """Advance one step using control, or zero-control if omitted."""
        u = (
            np.zeros((self.num_agents, 2), dtype=float)
            if control is None
            else _as_control_array(control, self.num_agents)
        )
        next_state = np.zeros_like(self._state)
        for agent_index in range(self.num_agents):
            next_state[agent_index] = rk4_step_np(
                self._state[agent_index],
                u[agent_index],
                self.config.dt,
                self.config.dynamics,
            )
        self._state = next_state
        self._step_idx += 1
        return self.state


class ThreeMantaRayEnv(MultiMantaRayEnv):
    """Compatibility wrapper that requires exactly three manta agents."""

    def __init__(
        self, starts: np.ndarray, goals: np.ndarray, config: MantaEnvConfig
    ) -> None:
        """Initialize the fixed-three-agent compatibility environment."""
        if np.asarray(starts).shape != (3, 7) or np.asarray(goals).shape != (3, 7):
            raise ValueError("ThreeMantaRayEnv requires starts/goals shaped (3, 7)")
        super().__init__(starts, goals, config)


def manta_rollout(
    env: MultiMantaRayEnv,
    controls: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out the manta environment and return state/control histories."""
    horizon = env.config.horizon
    control_shape = (horizon, env.num_agents, 2)
    control_hist = np.zeros(control_shape, dtype=float)
    if controls is not None:
        controls = np.asarray(controls, dtype=float)
        if controls.shape != control_shape:
            raise ValueError(
                f"controls must have shape {control_shape}, got {controls.shape}"
            )
        control_hist[:] = controls

    state_hist = np.zeros((horizon + 1, env.num_agents, 7), dtype=float)
    state_hist[0] = env.reset()
    for k in range(horizon):
        state_hist[k + 1] = env.step(control_hist[k])
    return state_hist, control_hist


def _as_manta_array(arr: np.ndarray) -> np.ndarray:
    """Validate and cast an array to shape ``(A>=1, 7)`` float."""
    out = np.asarray(arr, dtype=float)
    if out.ndim != 2 or out.shape[0] < 1 or out.shape[1] != 7:
        raise ValueError(f"expected shape (A>=1, 7), got {out.shape}")
    return out


def _as_control_array(arr: np.ndarray, num_agents: int) -> np.ndarray:
    """Validate and cast an array to shape ``(A, 2)`` float."""
    out = np.asarray(arr, dtype=float)
    if out.shape != (num_agents, 2):
        raise ValueError(f"expected shape ({num_agents}, 2), got {out.shape}")
    return out
