"""Minimal 3-agent single-integrator simulation environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnvConfig:
    """Configuration for the 3-agent single-integrator environment."""

    dt: float = 0.1
    horizon: int = 80


class ThreeAgentSingleIntegratorEnv:
    """Stateful 3-agent single-integrator environment.

    State uses shape ``(3, 2)`` where each row is agent ``[x, y]``.
    Controls use shape ``(3, 2)`` and integrate as ``x_{k+1} = x_k + dt * u_k``.
    """

    def __init__(
        self, starts: np.ndarray, goals: np.ndarray, config: EnvConfig
    ) -> None:
        """Initialize environment with fixed starts/goals and simulation config."""
        self.starts = _as_agent_array(starts)
        self.goals = _as_agent_array(goals)
        self.config = config
        self._state = self.starts.copy()
        self._step_idx = 0

    @property
    def state(self) -> np.ndarray:
        """Return current state array with shape ``(3, 2)``."""
        return self._state.copy()

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
            np.zeros((3, 2), dtype=float)
            if control is None
            else _as_agent_array(control)
        )
        self._state = self._state + self.config.dt * u
        self._step_idx += 1
        return self.state


def rollout(
    env: ThreeAgentSingleIntegratorEnv,
    controls: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out full horizon and return states/controls histories.

    Returns:
        ``states``: shape ``(T+1, 3, 2)``
        ``controls``: shape ``(T, 3, 2)``
    """
    horizon = env.config.horizon
    control_hist = np.zeros((horizon, 3, 2), dtype=float)
    if controls is not None:
        controls = np.asarray(controls, dtype=float)
        if controls.shape != (horizon, 3, 2):
            raise ValueError(
                f"controls must have shape {(horizon, 3, 2)}, got {controls.shape}"
            )
        control_hist[:] = controls

    state_hist = np.zeros((horizon + 1, 3, 2), dtype=float)
    state_hist[0] = env.reset()
    for k in range(horizon):
        state_hist[k + 1] = env.step(control_hist[k])
    return state_hist, control_hist


def _as_agent_array(arr: np.ndarray) -> np.ndarray:
    """Validate and cast array to shape ``(3, 2)`` float."""
    out = np.asarray(arr, dtype=float)
    if out.shape != (3, 2):
        raise ValueError(f"expected shape (3, 2), got {out.shape}")
    return out
