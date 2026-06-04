"""Plots and animations for manta LMPC runs."""

from __future__ import annotations

from pathlib import Path

from ._backend import configure_matplotlib

configure_matplotlib()
import numpy as np
from matplotlib import animation, patches, pyplot as plt

from scripts.learning.runner import cost_by_iteration
from scripts.simulation import StaticObstacle

AGENT_COLORS = ("tab:blue", "tab:red", "tab:green")


def plot_learning_progression(
    histories: list[dict[int, np.ndarray]],
    goals: np.ndarray,
    obstacle: StaticObstacle,
    out_path: str,
) -> None:
    """Save the APF and LMPC trajectory progression plot."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    g = np.asarray(goals, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 8))
    _set_workspace_limits(ax)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_title("3-Agent Manta LMPC Learning Progression")
    ax.add_patch(_obstacle_patch(obstacle, alpha=0.45))

    final_idx = len(histories) - 1
    for iteration, run in enumerate(histories):
        for agent, color in enumerate(AGENT_COLORS):
            traj = np.asarray(run[agent], dtype=float)
            if iteration == final_idx:
                ax.plot(
                    traj[:, 0],
                    traj[:, 1],
                    color=color,
                    linewidth=2.5,
                    label=f"Agent {agent} final",
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
                )

    for agent, color in enumerate(AGENT_COLORS):
        ax.plot(g[agent, 0], g[agent, 1], marker="*", color=color, markersize=15)

    ax.legend(loc="upper right")
    fig.tight_layout()
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

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("LMPC Learning: Agent Cost per Iteration")
    ax.set_xlabel("Iteration Number (0 = APF Baseline)")
    ax.set_ylabel("Total Cost (Time Steps to Goal)")
    ax.grid(True, linestyle="--", alpha=0.7)

    for agent, color in enumerate(AGENT_COLORS):
        jitter = (agent - 1) * 0.05
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
) -> None:
    """Save the final-iteration trajectory and wing-motion GIF."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    g = np.asarray(goals, dtype=float)
    histories = {
        agent: np.asarray(final_history[agent], dtype=float) for agent in range(3)
    }
    max_frames = max(len(histories[agent]) for agent in range(3))

    fig = plt.figure(figsize=(10, 14))
    gs = fig.add_gridspec(4, 2, height_ratios=[3, 1, 1, 1])

    ax_top = fig.add_subplot(gs[0, :])
    _set_workspace_limits(ax_top)
    ax_top.grid(True, linestyle="--", alpha=0.6)
    ax_top.set_title("LMPC Final Iteration: Dynamic Avoidance")
    ax_top.add_patch(_obstacle_patch(obstacle, alpha=0.45))

    lines = []
    dots = []
    for agent, color in enumerate(AGENT_COLORS):
        (line,) = ax_top.plot([], [], color=color, linewidth=2, alpha=0.8)
        (dot,) = ax_top.plot(
            [], [], marker="o", color=color, markersize=8, label=f"Agent {agent}"
        )
        ax_top.plot(g[agent, 0], g[agent, 1], marker="*", color=color, markersize=15)
        lines.append(line)
        dots.append(dot)
    ax_top.legend(loc="upper right")

    wing_lines = {}
    for agent, color in enumerate(AGENT_COLORS):
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
        for agent in range(3):
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


def _obstacle_patch(obstacle: StaticObstacle, alpha: float) -> patches.Circle:
    return patches.Circle(
        obstacle.center,
        obstacle.radius,
        color="gray",
        alpha=alpha,
        label="Static Obstacle",
    )


def _set_workspace_limits(ax: plt.Axes) -> None:
    ax.set_xlim(-1, 7)
    ax.set_ylim(-1, 7)
    ax.set_aspect("equal", adjustable="box")
