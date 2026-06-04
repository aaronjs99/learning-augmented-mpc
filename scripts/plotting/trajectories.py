"""Central plotting utilities for 3-agent rollouts."""

from __future__ import annotations

from ._backend import configure_matplotlib

import numpy as np

configure_matplotlib()
from matplotlib import pyplot as plt

PAIR_LABELS = ("d(0,1)", "d(0,2)", "d(1,2)")
AGENT_COLORS = ("tab:blue", "tab:orange", "tab:green")


def plot_trajectories(
    states: np.ndarray, goals: np.ndarray, title: str, out_path: str
) -> None:
    """Save a 2D trajectory plot for all agents with start and goal markers."""
    s = np.asarray(states, dtype=float)
    g = np.asarray(goals, dtype=float)
    if s.ndim != 3 or s.shape[1] != 3 or s.shape[2] < 2:
        raise ValueError(f"states must have shape (T+1, 3, D>=2), got {s.shape}")
    if g.ndim != 2 or g.shape[0] != 3 or g.shape[1] < 2:
        raise ValueError(f"goals must have shape (3, D>=2), got {g.shape}")
    pos = s[:, :, :2]
    goal_pos = g[:, :2]

    fig, ax = plt.subplots(figsize=(6, 6))
    for i, color in enumerate(AGENT_COLORS):
        ax.plot(
            pos[:, i, 0], pos[:, i, 1], color=color, linewidth=2, label=f"agent {i}"
        )
        ax.scatter(pos[0, i, 0], pos[0, i, 1], color=color, marker="o", s=50)
        ax.scatter(goal_pos[i, 0], goal_pos[i, 1], color=color, marker="x", s=70)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pairwise_distances(
    distances: np.ndarray,
    dt: float,
    safety_distance: float,
    title: str,
    out_path: str,
) -> None:
    """Save pairwise-distance time series with safety threshold."""
    d = np.asarray(distances, dtype=float)
    t = np.arange(d.shape[0], dtype=float) * dt

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, label in enumerate(PAIR_LABELS):
        ax.plot(t, d[:, i], linewidth=2, label=label)

    ax.axhline(
        safety_distance, color="red", linestyle="--", linewidth=1.5, label="safety"
    )
    ax.set_title(title)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("distance")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
