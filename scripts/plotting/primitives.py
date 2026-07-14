"""Shared visual primitives for static and animated manta diagnostics."""

from __future__ import annotations

from pathlib import Path

from ._backend import configure_matplotlib

configure_matplotlib()
from matplotlib import patches, pyplot as plt

from scripts.simulation import StaticObstacle


AGENT_COLORS = (
    "tab:blue",
    "tab:red",
    "tab:green",
    "magenta",
    "cyan",
    "tab:purple",
)


def agent_color(agent: int) -> str:
    """Return a stable plotting color for any nonnegative agent index."""
    if agent < 0:
        raise ValueError("agent index must be nonnegative")
    return AGENT_COLORS[agent % len(AGENT_COLORS)]


def add_obstacle_layers(
    ax: plt.Axes,
    obstacle: StaticObstacle | None,
    obstacle_padding: float,
) -> None:
    """Draw APF padding, inflated constraint, and physical obstacle layers."""
    if obstacle is None:
        return
    if obstacle_padding < 0.0:
        raise ValueError("obstacle_padding must be nonnegative")
    if obstacle_padding > 0.0:
        ax.add_patch(
            patches.Circle(
                obstacle.center,
                obstacle.radius + obstacle_padding,
                color="lightgray",
                alpha=0.25,
                label="APF padding",
                zorder=0,
            )
        )
    ax.add_patch(
        patches.Circle(
            obstacle.center,
            obstacle.radius,
            facecolor="gray",
            edgecolor="black",
            linewidth=1.0,
            alpha=0.25,
            label="Inflated obstacle constraint",
            zorder=1,
        )
    )
    if obstacle.physical_radius is not None and obstacle.physical_radius < obstacle.radius:
        ax.add_patch(
            patches.Circle(
                obstacle.center,
                obstacle.physical_radius,
                color="dimgray",
                alpha=0.75,
                label="Physical obstacle",
                zorder=2,
            )
        )


def goal_tolerance_patch(
    goal: object,
    goal_tolerance: float,
    color: str,
    *,
    linewidth: float = 1.0,
) -> patches.Circle:
    """Return the shared dotted goal-tolerance boundary."""
    if goal_tolerance <= 0.0:
        raise ValueError("goal_tolerance must be positive")
    return patches.Circle(
        goal,
        goal_tolerance,
        fill=False,
        edgecolor=color,
        linestyle=":",
        linewidth=linewidth,
        alpha=0.8,
    )


def set_workspace_limits(ax: plt.Axes) -> None:
    """Apply the configured scenario workspace used by current experiments."""
    ax.set_xlim(-1, 7)
    ax.set_ylim(-1, 7)
    ax.set_aspect("equal", adjustable="box")


def prepare_output_path(path: str | Path) -> Path:
    """Create an artifact's parent directory and return its normalized path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path
