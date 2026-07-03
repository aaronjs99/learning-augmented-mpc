"""Central plotting utilities for multi-agent rollouts."""

from __future__ import annotations

from ._backend import configure_matplotlib

import numpy as np

configure_matplotlib()
from matplotlib import pyplot as plt

from scripts.simulation import StaticObstacle

from .diagnostics import (
    add_diagnostic_box,
    add_highlighted_segments,
    add_status_markers,
    compute_diagnostics,
    legend_label,
    tensor_to_position_history,
)

AGENT_COLORS = ("tab:blue", "tab:red", "tab:green", "magenta", "cyan", "tab:purple")


def agent_color(agent: int) -> str:
    """Return a stable plotting color for an agent index."""
    return AGENT_COLORS[agent % len(AGENT_COLORS)]


def plot_trajectories(
    states: np.ndarray,
    goals: np.ndarray,
    title: str,
    out_path: str,
    *,
    goal_tolerance: float | None = None,
    obstacle: StaticObstacle | None = None,
    obstacle_padding: float = 0.0,
    safety_distance: float | None = None,
    statuses: list[dict[int, str]] | None = None,
) -> None:
    """Save a 2D trajectory plot for all agents with start and goal markers."""
    s = np.asarray(states, dtype=float)
    g = np.asarray(goals, dtype=float)
    if s.ndim != 3 or s.shape[1] < 2 or s.shape[2] < 2:
        raise ValueError(f"states must have shape (T+1, A>=2, D>=2), got {s.shape}")
    if g.ndim != 2 or g.shape[0] != s.shape[1] or g.shape[1] < 2:
        expected = (s.shape[1], "D>=2")
        raise ValueError(f"goals must have shape {expected}, got {g.shape}")
    pos = s[:, :, :2]
    goal_pos = g[:, :2]
    positions = tensor_to_position_history(s)
    num_agents = s.shape[1]

    fig, ax = plt.subplots(figsize=(6, 6))
    labels_used: set[str] = set()
    if obstacle is not None:
        if obstacle_padding > 0.0:
            ax.add_patch(
                plt.Circle(
                    obstacle.center,
                    obstacle.radius + obstacle_padding,
                    color="lightgray",
                    alpha=0.25,
                    label="APF padding",
                    zorder=0,
                )
            )
        ax.add_patch(
            plt.Circle(
                obstacle.center,
                obstacle.radius,
                facecolor="gray",
                edgecolor="black",
                linewidth=1.0,
                alpha=0.25,
                label="inflated obstacle constraint",
                zorder=1,
            )
        )
        if (
            obstacle.physical_radius is not None
            and obstacle.physical_radius < obstacle.radius
        ):
            ax.add_patch(
                plt.Circle(
                    obstacle.center,
                    obstacle.physical_radius,
                    color="dimgray",
                    alpha=0.75,
                    label="physical obstacle",
                    zorder=2,
                )
            )
    for i in range(num_agents):
        color = agent_color(i)
        stride = max(1, len(pos[:, i]) // 45)
        ax.scatter(
            pos[::stride, i, 0],
            pos[::stride, i, 1],
            marker=".",
            s=10,
            color=color,
            alpha=0.18,
            label=legend_label("Safe-set samples", labels_used),
            zorder=1.5,
        )
        ax.plot(
            pos[:, i, 0],
            pos[:, i, 1],
            color=color,
            linewidth=2,
            label=legend_label(f"agent {i}", labels_used),
            zorder=2,
        )
        add_highlighted_segments(
            ax,
            positions,
            i,
            obstacle=obstacle,
            obstacle_padding=obstacle_padding,
            safety_distance=safety_distance,
            linewidth=3.6,
            labels_used=labels_used,
        )
        ax.scatter(
            pos[0, i, 0],
            pos[0, i, 1],
            color=color,
            marker="^",
            edgecolors="black",
            linewidths=0.8,
            s=56,
            label=legend_label("Starts", labels_used),
            zorder=6,
        )
        ax.scatter(
            goal_pos[i, 0],
            goal_pos[i, 1],
            color=color,
            marker="*",
            s=92,
            zorder=6,
        )
        if goal_tolerance is not None:
            ax.add_patch(
                plt.Circle(
                    goal_pos[i],
                    goal_tolerance,
                    fill=False,
                    edgecolor=color,
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.8,
                )
            )

    add_status_markers(ax, positions, statuses, labels_used=labels_used)
    diagnostics = compute_diagnostics(
        positions, obstacle, safety_distance, obstacle_padding
    )
    add_diagnostic_box(ax, diagnostics, safety_distance=safety_distance)
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
    pair_labels = _pair_labels(_agent_count_from_pair_count(d.shape[1]))
    for i, label in enumerate(pair_labels):
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


def _pair_labels(num_agents: int) -> list[str]:
    return [
        f"d({i},{j})"
        for i in range(num_agents)
        for j in range(i + 1, num_agents)
    ]


def _agent_count_from_pair_count(pair_count: int) -> int:
    for num_agents in range(2, 64):
        if num_agents * (num_agents - 1) // 2 == pair_count:
            return num_agents
    raise ValueError(f"could not infer agent count from {pair_count} pair distances")
