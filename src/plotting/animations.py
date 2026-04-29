"""Animation utilities for 3-agent rollout diagnostics."""

from __future__ import annotations

from pathlib import Path

from ._backend import configure_matplotlib

configure_matplotlib()
import numpy as np
from matplotlib import animation, patches, pyplot as plt

from .trajectories import AGENT_COLORS


def save_rollout_animation(
    states: np.ndarray,
    goals: np.ndarray,
    safety_distance: float,
    dt: float,
    out_path: str,
    fps: int = 10,
) -> None:
    """Save a GIF animation with agent positions, trails, goals, and safety radii."""
    s = np.asarray(states, dtype=float)
    g = np.asarray(goals, dtype=float)
    if s.ndim != 3 or s.shape[1:] != (3, 2):
        raise ValueError(f"states must have shape (T+1, 3, 2), got {s.shape}")
    if g.shape != (3, 2):
        raise ValueError(f"goals must have shape (3, 2), got {g.shape}")

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    _set_equal_limits(ax, s, g, safety_distance)

    trails = []
    points = []
    safety_circles = []
    for agent_index, color in enumerate(AGENT_COLORS):
        ax.scatter(
            s[0, agent_index, 0], s[0, agent_index, 1], color=color, marker="o", s=45
        )
        ax.scatter(g[agent_index, 0], g[agent_index, 1], color=color, marker="x", s=75)
        (trail,) = ax.plot(
            [], [], color=color, linewidth=2, alpha=0.75, label=f"agent {agent_index}"
        )
        (point,) = ax.plot([], [], color=color, marker="o", markersize=7)
        circle = patches.Circle(
            s[0, agent_index],
            safety_distance / 2.0,
            color=color,
            alpha=0.12,
            label="safety radius" if agent_index == 0 else None,
        )
        ax.add_patch(circle)
        trails.append(trail)
        points.append(point)
        safety_circles.append(circle)

    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")
    ax.set_title("Baseline decentralized MPC")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    def update(frame: int) -> list[object]:
        artists: list[object] = [time_text]
        for agent_index in range(3):
            trails[agent_index].set_data(
                s[: frame + 1, agent_index, 0], s[: frame + 1, agent_index, 1]
            )
            points[agent_index].set_data(
                [s[frame, agent_index, 0]], [s[frame, agent_index, 1]]
            )
            safety_circles[agent_index].center = s[frame, agent_index]
            artists.extend(
                (trails[agent_index], points[agent_index], safety_circles[agent_index])
            )
        time_text.set_text(f"t = {frame * dt:.1f}s")
        return artists

    anim = animation.FuncAnimation(
        fig, update, frames=s.shape[0], interval=1000 / fps, blit=True
    )
    anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=120)
    plt.close(fig)


def _set_equal_limits(
    ax: plt.Axes, states: np.ndarray, goals: np.ndarray, safety_distance: float
) -> None:
    points = np.vstack((states.reshape(-1, 2), goals))
    margin = max(0.25, safety_distance)
    mins = np.min(points, axis=0) - margin
    maxs = np.max(points, axis=0) + margin
    span = np.maximum(maxs - mins, 1.0)
    center = 0.5 * (mins + maxs)
    half_width = 0.5 * float(np.max(span))
    ax.set_xlim(center[0] - half_width, center[0] + half_width)
    ax.set_ylim(center[1] - half_width, center[1] + half_width)
    ax.set_aspect("equal", adjustable="box")
