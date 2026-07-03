"""Animation utilities for multi-agent rollout diagnostics."""

from __future__ import annotations

from pathlib import Path

from ._backend import configure_matplotlib

configure_matplotlib()
import numpy as np
from matplotlib import animation, patches, pyplot as plt

from .trajectories import agent_color


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
    if s.ndim != 3 or s.shape[1] < 2 or s.shape[2] < 2:
        raise ValueError(f"states must have shape (T+1, A>=2, D>=2), got {s.shape}")
    if g.ndim != 2 or g.shape[0] != s.shape[1] or g.shape[1] < 2:
        expected = (s.shape[1], "D>=2")
        raise ValueError(f"goals must have shape {expected}, got {g.shape}")
    pos = s[:, :, :2]
    goal_pos = g[:, :2]
    num_agents = s.shape[1]

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    _set_equal_limits(ax, pos, goal_pos, safety_distance)

    trails = []
    points = []
    safety_circles = []
    for agent_index in range(num_agents):
        color = agent_color(agent_index)
        ax.scatter(
            pos[0, agent_index, 0],
            pos[0, agent_index, 1],
            color=color,
            marker="o",
            s=45,
        )
        ax.scatter(
            goal_pos[agent_index, 0],
            goal_pos[agent_index, 1],
            color=color,
            marker="x",
            s=75,
        )
        (trail,) = ax.plot(
            [], [], color=color, linewidth=2, alpha=0.75, label=f"agent {agent_index}"
        )
        (point,) = ax.plot([], [], color=color, marker="o", markersize=7)
        circle = patches.Circle(
            pos[0, agent_index],
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
        for agent_index in range(num_agents):
            trails[agent_index].set_data(
                pos[: frame + 1, agent_index, 0], pos[: frame + 1, agent_index, 1]
            )
            points[agent_index].set_data(
                [pos[frame, agent_index, 0]], [pos[frame, agent_index, 1]]
            )
            safety_circles[agent_index].center = pos[frame, agent_index]
            artists.extend(
                (trails[agent_index], points[agent_index], safety_circles[agent_index])
            )
        time_text.set_text(f"t = {frame * dt:.1f}s")
        return artists

    anim = animation.FuncAnimation(
        fig, update, frames=pos.shape[0], interval=1000 / fps, blit=True
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
