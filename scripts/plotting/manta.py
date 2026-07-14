"""Plots and animations for manta LMPC runs."""

from __future__ import annotations

from pathlib import Path

from ._backend import configure_matplotlib

configure_matplotlib()
import numpy as np
from matplotlib import animation, patches, pyplot as plt

from scripts.metrics import cost_by_iteration
from scripts.simulation import StaticObstacle

from .diagnostics import (
    add_diagnostic_box,
    add_highlighted_segments,
    add_status_markers,
    compute_diagnostics,
    legend_label,
    to_position_history,
)
from .trajectories import agent_color


def plot_learning_progression(
    histories: list[dict[int, np.ndarray]],
    goals: np.ndarray,
    obstacle: StaticObstacle,
    out_path: str,
    *,
    goal_tolerance: float | None = None,
    obstacle_padding: float = 0.0,
    safety_distance: float | None = None,
    statuses_by_iteration: list[list[dict[int, str]]] | None = None,
) -> None:
    """Save the APF and LMPC trajectory progression plot."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    g = np.asarray(goals, dtype=float)
    num_agents = g.shape[0]

    fig, ax = plt.subplots(figsize=(10, 8))
    _set_workspace_limits(ax)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_title(f"{num_agents}-Agent Manta LMPC Learning Progression")
    _add_obstacle_layers(ax, obstacle, obstacle_padding)

    labels_used: set[str] = set()
    final_idx = len(histories) - 1
    for iteration, run in enumerate(histories):
        positions = to_position_history(run)
        for agent in range(num_agents):
            color = agent_color(agent)
            traj = np.asarray(run[agent], dtype=float)
            _add_safe_set_points(
                ax,
                traj,
                color,
                labels_used,
                is_final=iteration == final_idx,
            )
            if iteration == final_idx:
                ax.plot(
                    traj[:, 0],
                    traj[:, 1],
                    color=color,
                    linewidth=2.5,
                    label=legend_label(f"Agent {agent} final", labels_used),
                    zorder=3,
                )
            else:
                alpha = 0.15 + 0.35 * (iteration / max(1, final_idx))
                ax.plot(
                    traj[:, 0],
                    traj[:, 1],
                    "--",
                    color=color,
                    linewidth=1.5,
                    alpha=alpha,
                    zorder=2,
                )
            add_highlighted_segments(
                ax,
                positions,
                agent,
                obstacle=obstacle,
                obstacle_padding=obstacle_padding,
                safety_distance=safety_distance,
                linewidth=4.0 if iteration == final_idx else 2.8,
                labels_used=labels_used,
            )

        _add_iteration_label(ax, positions, iteration, final_idx)
        statuses = _statuses_for_history(iteration, statuses_by_iteration)
        add_status_markers(ax, positions, statuses, labels_used=labels_used)

    for agent in range(num_agents):
        color = agent_color(agent)
        start = np.asarray(histories[0][agent], dtype=float)[0]
        ax.scatter(
            start[0],
            start[1],
            marker="^",
            color=color,
            edgecolors="black",
            s=62,
            linewidths=0.8,
            label=legend_label("Starts", labels_used),
            zorder=6,
        )
        ax.plot(g[agent, 0], g[agent, 1], marker="*", color=color, markersize=15)
        if goal_tolerance is not None:
            ax.add_patch(_goal_tolerance_patch(g[agent], goal_tolerance, color))

    final_positions = to_position_history(histories[-1])
    diagnostics = compute_diagnostics(
        final_positions, obstacle, safety_distance, obstacle_padding
    )
    add_diagnostic_box(ax, diagnostics, safety_distance=safety_distance)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    fig.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_cost_decrease(
    histories: list[dict[int, np.ndarray]],
    goals: np.ndarray,
    out_path: str,
    *,
    goal_tolerance: float = 0.5,
) -> None:
    """Save cost-vs-iteration plot using first-hit time as cost."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    costs = cost_by_iteration(histories, goals, goal_tolerance=goal_tolerance)
    iterations = np.arange(len(histories), dtype=float)
    num_agents = len(costs)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f"LMPC Learning: Agent Cost Proxy (tol={goal_tolerance:g})")
    ax.set_xlabel("Iteration Number (0 = APF Baseline)")
    ax.set_ylabel("First-Hit Step Proxy")
    ax.grid(True, linestyle="--", alpha=0.7)

    for agent in sorted(costs):
        color = agent_color(agent)
        jitter = (agent - (num_agents - 1) / 2.0) * 0.05
        values = costs[agent]
        ax.plot(
            iterations + jitter,
            values,
            marker="o",
            markersize=8,
            linewidth=2.5,
            color=color,
            label=f"Agent {agent}",
        )
        for x_val, y_val in zip(iterations, values):
            ax.annotate(
                str(y_val),
                (x_val + jitter, y_val),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=10,
                color=color,
            )

    ax.set_xticks(iterations)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_manta_animation(
    final_history: dict[int, np.ndarray],
    goals: np.ndarray,
    obstacle: StaticObstacle,
    dt: float,
    out_path: str,
    *,
    fps: int = 20,
    goal_tolerance: float | None = None,
    obstacle_padding: float = 0.0,
    safety_distance: float | None = None,
    statuses: list[dict[int, str]] | None = None,
) -> None:
    """Save the final-iteration trajectory and wing-motion GIF."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    g = np.asarray(goals, dtype=float)
    num_agents = g.shape[0]
    histories = {
        agent: np.asarray(final_history[agent], dtype=float)
        for agent in range(num_agents)
    }
    max_frames = max(len(histories[agent]) for agent in range(num_agents))

    fig = plt.figure(figsize=(10, 14))
    gs = fig.add_gridspec(
        num_agents + 1, 2, height_ratios=[3] + [1] * num_agents
    )

    ax_top = fig.add_subplot(gs[0, :])
    _set_workspace_limits(ax_top)
    ax_top.grid(True, linestyle="--", alpha=0.6)
    ax_top.set_title("LMPC Final Iteration: Dynamic Avoidance")
    _add_obstacle_layers(ax_top, obstacle, obstacle_padding)
    labels_used: set[str] = set()

    lines = []
    dots = []
    for agent in range(num_agents):
        color = agent_color(agent)
        start = histories[agent][0]
        ax_top.scatter(
            start[0],
            start[1],
            marker="^",
            color=color,
            edgecolors="black",
            s=52,
            linewidths=0.8,
            label=legend_label("Starts", labels_used),
            zorder=6,
        )
        (line,) = ax_top.plot([], [], color=color, linewidth=2, alpha=0.8)
        (dot,) = ax_top.plot(
            [], [], marker="o", color=color, markersize=8, label=f"Agent {agent}"
        )
        ax_top.plot(g[agent, 0], g[agent, 1], marker="*", color=color, markersize=15)
        if goal_tolerance is not None:
            ax_top.add_patch(_goal_tolerance_patch(g[agent], goal_tolerance, color))
        lines.append(line)
        dots.append(dot)
    diagnostics = compute_diagnostics(
        {agent: traj[:, :2] for agent, traj in histories.items()},
        obstacle,
        safety_distance,
        obstacle_padding,
    )
    add_diagnostic_box(ax_top, diagnostics, safety_distance=safety_distance)
    add_status_markers(
        ax_top,
        {agent: traj[:, :2] for agent, traj in histories.items()},
        statuses,
        labels_used=labels_used,
    )
    ax_top.legend(loc="upper right")

    wing_lines = {}
    for agent in range(num_agents):
        color = agent_color(agent)
        ax_rear = fig.add_subplot(gs[agent + 1, 0])
        ax_rear.set_xlim(-1.2, 1.2)
        ax_rear.set_ylim(-0.8, 0.8)
        ax_rear.set_title(f"Agent {agent}: Rear Elevation")
        ax_rear.grid(True)
        (rear_l,) = ax_rear.plot([], [], color=color, linewidth=4)
        (rear_r,) = ax_rear.plot([], [], color=color, linewidth=4)
        ax_rear.plot([0], [0], "ko")

        ax_side = fig.add_subplot(gs[agent + 1, 1])
        ax_side.set_xlim(-1.0, 1.0)
        ax_side.set_ylim(-0.8, 0.8)
        ax_side.set_title(f"Agent {agent}: Side Profile")
        ax_side.grid(True)
        ax_side.plot([-0.6, 0.4], [0, 0], "k-", linewidth=2)
        ax_side.plot([0.4], [0], "ko")
        (side_w,) = ax_side.plot([], [], color=color, linewidth=4)
        wing_lines[agent] = (rear_l, rear_r, side_w)

    def update(frame: int) -> list[object]:
        t = frame * dt
        artists: list[object] = []
        for agent in range(num_agents):
            traj = histories[agent]
            idx = min(frame, len(traj) - 1)
            lines[agent].set_data(traj[: idx + 1, 0], traj[: idx + 1, 1])
            dots[agent].set_data([traj[idx, 0]], [traj[idx, 1]])
            artists.extend([lines[agent], dots[agent]])

            flap_l = (traj[idx, 3] * 0.6) * np.sin(t)
            flap_r = (traj[idx, 5] * 0.6) * np.sin(t)
            rear_l, rear_r, side_w = wing_lines[agent]
            rear_l.set_data([0, -np.cos(flap_l)], [0, np.sin(flap_l)])
            rear_r.set_data([0, np.cos(flap_r)], [0, np.sin(flap_r)])
            side_w.set_data([0, -0.4], [0, np.sin(flap_l)])
            artists.extend([rear_l, rear_r, side_w])
        return artists

    anim = animation.FuncAnimation(
        fig, update, frames=max_frames, interval=1000 / fps, blit=True, repeat=True
    )
    fig.tight_layout()
    anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=120)
    plt.close(fig)


def _add_obstacle_layers(
    ax: plt.Axes, obstacle: StaticObstacle, obstacle_padding: float
) -> None:
    if obstacle_padding > 0.0:
        ax.add_patch(_obstacle_padding_patch(obstacle, obstacle_padding))
    ax.add_patch(_inflated_obstacle_patch(obstacle))
    if obstacle.physical_radius is not None and obstacle.physical_radius < obstacle.radius:
        ax.add_patch(_physical_obstacle_patch(obstacle))


def _inflated_obstacle_patch(obstacle: StaticObstacle) -> patches.Circle:
    return patches.Circle(
        obstacle.center,
        obstacle.radius,
        facecolor="gray",
        edgecolor="black",
        linewidth=1.0,
        alpha=0.25,
        label="Inflated obstacle constraint",
        zorder=1,
    )


def _physical_obstacle_patch(obstacle: StaticObstacle) -> patches.Circle:
    return patches.Circle(
        obstacle.center,
        obstacle.physical_radius,
        color="dimgray",
        alpha=0.75,
        label="Physical obstacle",
        zorder=2,
    )


def _obstacle_padding_patch(
    obstacle: StaticObstacle, obstacle_padding: float
) -> patches.Circle:
    return patches.Circle(
        obstacle.center,
        obstacle.radius + obstacle_padding,
        color="lightgray",
        alpha=0.25,
        label="APF Padding",
        zorder=0,
    )


def _add_safe_set_points(
    ax: plt.Axes,
    traj: np.ndarray,
    color: str,
    labels_used: set[str],
    *,
    is_final: bool,
) -> None:
    stride = max(1, len(traj) // 45)
    ax.scatter(
        traj[::stride, 0],
        traj[::stride, 1],
        marker=".",
        s=12 if is_final else 8,
        color=color,
        alpha=0.22 if is_final else 0.10,
        label=legend_label("Safe-set samples", labels_used),
        zorder=1.5,
    )


def _add_iteration_label(
    ax: plt.Axes,
    positions: dict[int, np.ndarray],
    iteration: int,
    final_idx: int,
) -> None:
    if iteration == final_idx:
        return
    if 0 not in positions or len(positions[0]) == 0:
        return
    traj = positions[0]
    idx = min(len(traj) - 1, max(0, len(traj) // 2))
    label = "APF" if iteration == 0 else f"Iter {iteration}"
    ax.text(
        traj[idx, 0] + 0.06,
        traj[idx, 1] + 0.06,
        label,
        fontsize=8,
        color="0.25",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.72,
        },
        zorder=8,
    )


def _statuses_for_history(
    iteration: int, statuses_by_iteration: list[list[dict[int, str]]] | None
) -> list[dict[int, str]] | None:
    if iteration == 0 or statuses_by_iteration is None:
        return None
    status_idx = iteration - 1
    if status_idx >= len(statuses_by_iteration):
        return None
    return statuses_by_iteration[status_idx]


def _goal_tolerance_patch(
    goal: np.ndarray, goal_tolerance: float, color: str
) -> patches.Circle:
    return patches.Circle(
        goal[:2],
        goal_tolerance,
        fill=False,
        edgecolor=color,
        linestyle=":",
        linewidth=1.2,
        alpha=0.8,
    )


def _set_workspace_limits(ax: plt.Axes) -> None:
    ax.set_xlim(-1, 7)
    ax.set_ylim(-1, 7)
    ax.set_aspect("equal", adjustable="box")
