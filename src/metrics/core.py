"""Run-level metrics for 3-agent trajectory rollouts."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class RolloutMetrics:
    """Summary metrics computed from state and optional control histories."""

    total_cost_proxy: float
    minimum_pairwise_distance: float
    collision_count: int
    time_to_goal: float
    control_effort: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        """Serialize metrics to plain dictionary for JSON output."""
        return asdict(self)


def pairwise_distances(states: np.ndarray) -> np.ndarray:
    """Compute pairwise distances over time.

    Args:
        states: array of shape ``(T+1, 3, 2)``.

    Returns:
        Array of shape ``(T+1, 3)`` for pairs ``(0,1), (0,2), (1,2)``.
    """
    s = np.asarray(states, dtype=float)
    if s.ndim != 3 or s.shape[1:] != (3, 2):
        raise ValueError(f"states must have shape (T+1, 3, 2), got {s.shape}")

    d01 = np.linalg.norm(s[:, 0] - s[:, 1], axis=1)
    d02 = np.linalg.norm(s[:, 0] - s[:, 2], axis=1)
    d12 = np.linalg.norm(s[:, 1] - s[:, 2], axis=1)
    return np.stack((d01, d02, d12), axis=1)


def compute_rollout_metrics(
    states: np.ndarray,
    goals: np.ndarray,
    safety_distance: float,
    dt: float,
    controls: np.ndarray | None = None,
    goal_tolerance: float = 0.1,
) -> RolloutMetrics:
    """Compute minimal rollout metrics used by baseline and LMPC comparisons."""
    s = np.asarray(states, dtype=float)
    g = np.asarray(goals, dtype=float)
    if s.ndim != 3 or s.shape[1:] != (3, 2):
        raise ValueError(f"states must have shape (T+1, 3, 2), got {s.shape}")
    if g.shape != (3, 2):
        raise ValueError(f"goals must have shape (3, 2), got {g.shape}")

    dist_pairs = pairwise_distances(s)
    min_pair = float(np.min(dist_pairs))
    collisions = int(np.sum(dist_pairs < safety_distance))

    goal_error = np.linalg.norm(s - g[None, :, :], axis=2)
    total_cost_proxy = float(np.sum(goal_error**2))

    reached = np.all(goal_error <= goal_tolerance, axis=1)
    first_hit = int(np.argmax(reached)) if np.any(reached) else s.shape[0] - 1
    time_to_goal = float(first_hit * dt)

    control_effort: float | None = None
    if controls is not None:
        u = np.asarray(controls, dtype=float)
        if u.shape != (s.shape[0] - 1, 3, 2):
            raise ValueError(f"controls must have shape {(s.shape[0] - 1, 3, 2)}, got {u.shape}")
        control_effort = float(np.sum(u**2))

    return RolloutMetrics(
        total_cost_proxy=total_cost_proxy,
        minimum_pairwise_distance=min_pair,
        collision_count=collisions,
        time_to_goal=time_to_goal,
        control_effort=control_effort,
    )
