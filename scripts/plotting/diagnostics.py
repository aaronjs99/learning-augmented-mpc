"""Diagnostic overlays shared by static trajectory plots."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from scripts.metrics import segment_pairwise_distances, segment_point_distances
from scripts.simulation import StaticObstacle

OBSTACLE_VIOLATION_COLOR = "crimson"
PAIRWISE_VIOLATION_COLOR = "black"
PADDING_COLOR = "darkorange"


@dataclass(frozen=True)
class TrajectoryDiagnostics:
    """Minimum-distance diagnostics for one multi-agent trajectory set."""

    min_pairwise_distance: float
    min_obstacle_clearance: float
    pairwise_violation_count: int
    obstacle_violation_count: int
    padding_entry_count: int


def to_position_history(run: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Return per-agent ``(x, y)`` arrays from one saved iteration history."""
    return {
        agent: np.asarray(states, dtype=float)[:, :2]
        for agent, states in run.items()
    }


def tensor_to_position_history(states: np.ndarray) -> dict[int, np.ndarray]:
    """Return per-agent ``(x, y)`` arrays from ``(T, agents, state_dim)`` states."""
    s = np.asarray(states, dtype=float)
    return {agent: s[:, agent, :2] for agent in range(s.shape[1])}


def compute_diagnostics(
    positions: dict[int, np.ndarray],
    obstacle: StaticObstacle | None,
    safety_distance: float | None,
    obstacle_padding: float = 0.0,
) -> TrajectoryDiagnostics:
    """Compute minimum pairwise distance and obstacle-clearance diagnostics."""
    min_pairwise = np.inf
    pairwise_violations = 0
    if safety_distance is not None:
        agents = sorted(positions)
        for i, agent_i in enumerate(agents):
            pos_i = positions[agent_i]
            for agent_j in agents[i + 1 :]:
                pos_j = positions[agent_j]
                count = min(len(pos_i), len(pos_j))
                if count == 0:
                    continue
                distances = segment_pairwise_distances(
                    pos_i[:count], pos_j[:count]
                )
                min_pairwise = min(min_pairwise, float(np.min(distances)))
                pairwise_violations += int(np.sum(distances < safety_distance))

    min_clearance = np.inf
    obstacle_violations = 0
    padding_entries = 0
    if obstacle is not None:
        center = np.asarray(obstacle.center, dtype=float)
        for pos in positions.values():
            if len(pos) == 0:
                continue
            clearances = segment_point_distances(pos, center) - obstacle.radius
            min_clearance = min(min_clearance, float(np.min(clearances)))
            obstacle_violations += int(np.sum(clearances < 0.0))
            if obstacle_padding > 0.0:
                padding_entries += int(
                    np.sum((clearances >= 0.0) & (clearances < obstacle_padding))
                )

    return TrajectoryDiagnostics(
        min_pairwise_distance=_finite_or_nan(min_pairwise),
        min_obstacle_clearance=_finite_or_nan(min_clearance),
        pairwise_violation_count=pairwise_violations,
        obstacle_violation_count=obstacle_violations,
        padding_entry_count=padding_entries,
    )


def add_highlighted_segments(
    ax: plt.Axes,
    positions: dict[int, np.ndarray],
    agent: int,
    *,
    obstacle: StaticObstacle | None,
    obstacle_padding: float,
    safety_distance: float | None,
    linewidth: float,
    labels_used: set[str],
) -> None:
    """Overlay colored diagnostic segments on one agent trajectory."""
    traj = np.asarray(positions[agent], dtype=float)
    if len(traj) < 2:
        return

    obstacle_mask, padding_mask = _obstacle_segment_masks(
        traj, obstacle, obstacle_padding
    )
    pairwise_mask = _pairwise_segment_mask(positions, agent, safety_distance)
    pairwise_mask = pairwise_mask & ~obstacle_mask
    padding_mask = padding_mask & ~obstacle_mask & ~pairwise_mask

    _add_segment_collection(
        ax,
        traj,
        obstacle_mask,
        OBSTACLE_VIOLATION_COLOR,
        linewidth,
        "Obstacle violation",
        labels_used,
    )
    _add_segment_collection(
        ax,
        traj,
        pairwise_mask,
        PAIRWISE_VIOLATION_COLOR,
        linewidth,
        "Agent spacing violation",
        labels_used,
    )
    _add_segment_collection(
        ax,
        traj,
        padding_mask,
        PADDING_COLOR,
        linewidth,
        "Inside APF padding",
        labels_used,
    )


def add_diagnostic_box(
    ax: plt.Axes,
    diagnostics: TrajectoryDiagnostics,
    *,
    safety_distance: float | None,
) -> None:
    """Add compact minimum-distance diagnostics to a plot."""
    pair_text = "n/a"
    if not np.isnan(diagnostics.min_pairwise_distance):
        pair_text = f"{diagnostics.min_pairwise_distance:.2f}"
        if safety_distance is not None:
            pair_text += f" / {safety_distance:.2f}"

    obstacle_text = "n/a"
    if not np.isnan(diagnostics.min_obstacle_clearance):
        obstacle_text = f"{diagnostics.min_obstacle_clearance:.2f}"

    text = (
        f"min pairwise: {pair_text}\n"
        f"min obstacle clearance: {obstacle_text}\n"
        f"violations: pair {diagnostics.pairwise_violation_count}, "
        f"obs {diagnostics.obstacle_violation_count}"
    )
    ax.text(
        0.02,
        0.02,
        text,
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.82,
        },
        zorder=10,
    )


def add_status_markers(
    ax: plt.Axes,
    positions: dict[int, np.ndarray],
    statuses: list[dict[int, str]] | None,
    *,
    labels_used: set[str],
) -> None:
    """Mark hold, solver fallback, and safety-filter locations."""
    if not statuses:
        return

    hold_seen: set[int] = set()
    fallback_count = 0
    for step, step_statuses in enumerate(statuses):
        for agent, status in step_statuses.items():
            if agent not in positions:
                continue
            traj = positions[agent]
            idx = min(step + 1, len(traj) - 1)
            if idx < 0:
                continue
            x_val, y_val = traj[idx]
            if status == "hold" and agent not in hold_seen:
                _scatter_once(
                    ax,
                    x_val,
                    y_val,
                    marker="s",
                    color="white",
                    edgecolor="black",
                    size=42,
                    label="Hold starts",
                    labels_used=labels_used,
                )
                hold_seen.add(agent)
            elif status in {"fallback", "fallback_apf"} and fallback_count < 40:
                _scatter_once(
                    ax,
                    x_val,
                    y_val,
                    marker="x",
                    color="black",
                    edgecolor="black",
                    size=34,
                    label="Solver fallback",
                    labels_used=labels_used,
                )
                fallback_count += 1
            elif status.startswith("safety_filter"):
                _scatter_once(
                    ax,
                    x_val,
                    y_val,
                    marker="P",
                    color="darkorange",
                    edgecolor="black",
                    size=44,
                    label="Safety filter",
                    labels_used=labels_used,
                )


def legend_label(label: str, labels_used: set[str]) -> str:
    """Return a legend label once, then suppress repeats."""
    if label in labels_used:
        return "_nolegend_"
    labels_used.add(label)
    return label


def _obstacle_segment_masks(
    traj: np.ndarray, obstacle: StaticObstacle | None, obstacle_padding: float
) -> tuple[np.ndarray, np.ndarray]:
    count = max(0, len(traj) - 1)
    if obstacle is None or count == 0:
        return np.zeros(count, dtype=bool), np.zeros(count, dtype=bool)

    center = np.asarray(obstacle.center, dtype=float)
    segment_clearance = segment_point_distances(traj, center) - obstacle.radius
    obstacle_mask = segment_clearance < 0.0
    padding_mask = np.zeros(count, dtype=bool)
    if obstacle_padding > 0.0:
        padding_mask = (segment_clearance >= 0.0) & (
            segment_clearance < obstacle_padding
        )
    return obstacle_mask, padding_mask


def _pairwise_segment_mask(
    positions: dict[int, np.ndarray],
    agent: int,
    safety_distance: float | None,
) -> np.ndarray:
    traj = np.asarray(positions[agent], dtype=float)
    count = max(0, len(traj) - 1)
    segment_mask = np.zeros(count, dtype=bool)
    if safety_distance is None:
        return np.zeros(count, dtype=bool)

    for other, other_traj in positions.items():
        if other == agent:
            continue
        shared = min(len(traj), len(other_traj))
        if shared == 0:
            continue
        if shared < 2:
            continue
        distances = segment_pairwise_distances(
            traj[:shared], np.asarray(other_traj, dtype=float)[:shared]
        )
        segment_mask[: len(distances)] |= distances < safety_distance
    return segment_mask


def _add_segment_collection(
    ax: plt.Axes,
    traj: np.ndarray,
    mask: np.ndarray,
    color: str,
    linewidth: float,
    label: str,
    labels_used: set[str],
) -> None:
    if not np.any(mask):
        return
    segments = np.stack((traj[:-1], traj[1:]), axis=1)[mask]
    collection = LineCollection(
        segments,
        colors=color,
        linewidths=linewidth,
        alpha=0.95,
        label=legend_label(label, labels_used),
        zorder=5,
    )
    ax.add_collection(collection)


def _scatter_once(
    ax: plt.Axes,
    x_val: float,
    y_val: float,
    *,
    marker: str,
    color: str,
    edgecolor: str,
    size: float,
    label: str,
    labels_used: set[str],
) -> None:
    scatter_kwargs = {
        "marker": marker,
        "color": color,
        "s": size,
        "linewidths": 1.2,
        "label": legend_label(label, labels_used),
        "zorder": 7,
    }
    if marker not in {"x", "+", "1", "2", "3", "4", "|", "_"}:
        scatter_kwargs["edgecolors"] = edgecolor
    ax.scatter([x_val], [y_val], **scatter_kwargs)


def _finite_or_nan(value: float) -> float:
    return float(value) if np.isfinite(value) else float("nan")
