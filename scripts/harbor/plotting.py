"""Static and animated diagnostics for heterogeneous harbor rollouts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.plotting._backend import configure_matplotlib

from .simulation import HarborAgent, HarborResult

configure_matplotlib()

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


_COLORS = {
    "ugv": "#2ca02c",
    "usv": "#1f77b4",
    "rov": "#9467bd",
}
_LABELS = {"ugv": "UGV", "usv": "USV", "rov": "ROV"}


def save_harbor_comparison(
    results: dict[str, HarborResult],
    agents: list[HarborAgent],
    simulation_config,
    path: str | Path,
) -> Path:
    """Save top-down independent-versus-coordinated trajectory panels."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(results), figsize=(6.4 * len(results), 6.0))
    axes = np.atleast_1d(axes)
    for axis, (label, result) in zip(axes, results.items(), strict=True):
        _draw_harbor_map(axis, simulation_config)
        _draw_paths(axis, result, agents)
        axis.set_title(
            f"{label}\nviolations={result.pairwise_violation_count}, "
            f"min distance={result.min_pairwise_distance:.3f} m"
        )
    fig.suptitle("Untethered Heterogeneous Harbor Coordination", fontsize=16)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def save_harbor_animation(
    result: HarborResult,
    agents: list[HarborAgent],
    simulation_config,
    path: str | Path,
    *,
    fps: int = 15,
) -> Path:
    """Save a top-down GIF with an accompanying platform-depth panel."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, (map_axis, depth_axis) = plt.subplots(
        1, 2, figsize=(11.0, 5.2), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )
    models = {agent.name: agent.model for agent in agents}
    radii = {agent.name: agent.radius for agent in agents}
    goals = {agent.name: agent.goal for agent in agents}
    names = [agent.name for agent in agents]
    frames = max(len(result.positions[name]) for name in names)

    def update(frame: int):
        map_axis.clear()
        depth_axis.clear()
        _draw_harbor_map(map_axis, simulation_config)
        for name in names:
            kind = models[name].kind
            color = _COLORS[kind]
            positions = result.positions[name]
            index = min(frame, len(positions) - 1)
            shown = positions[: index + 1]
            map_axis.plot(shown[:, 0], shown[:, 1], color=color, linewidth=2)
            map_axis.scatter(
                shown[-1, 0], shown[-1, 1], color=color, s=90, label=_LABELS[kind]
            )
            map_axis.scatter(
                goals[name][0], goals[name][1], color=color, marker="*", s=150
            )
            footprint = plt.Circle(
                shown[-1, :2], radii[name], facecolor=color, alpha=0.12, edgecolor=color
            )
            map_axis.add_patch(footprint)
            depth_axis.plot(
                np.arange(index + 1), shown[:, 2], color=color, linewidth=2
            )
            depth_axis.scatter(index, shown[-1, 2], color=color, s=50)
        map_axis.set_title(f"ETA-priority coordination, step {frame}")
        map_axis.set_xlabel("x [m]")
        map_axis.set_ylabel("y [m]")
        map_axis.grid(True, alpha=0.3)
        map_axis.set_aspect("equal", adjustable="box")
        map_axis.set_xlim(*simulation_config.world_x_bounds)
        map_axis.set_ylim(*simulation_config.world_y_bounds)
        map_axis.legend(loc="upper left")
        depth_axis.axhline(0.0, color="#4aa3df", linewidth=1.2, alpha=0.7)
        depth_axis.axhline(
            simulation_config.seabed_z,
            color="#665544",
            linewidth=2.0,
            alpha=0.8,
        )
        depth_axis.set_title("Operating Layer / Depth")
        depth_axis.set_xlabel("step")
        depth_axis.set_ylabel("z [m]")
        depth_axis.set_xlim(0, frames - 1)
        depth_axis.set_ylim(simulation_config.seabed_z - 0.2, 0.35)
        depth_axis.grid(True, alpha=0.3)
        return ()

    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps)
    animation.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return output


def save_network_robustness_heatmap(
    records: list[dict[str, float | int]], path: str | Path
) -> Path:
    """Save safety-rate and completion-cost heatmaps for a network sweep."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    delays = sorted({int(record["delay_steps"]) for record in records})
    dropouts = sorted(
        {float(record["dropout_probability"]) for record in records}
    )
    safety = np.empty((len(dropouts), len(delays)))
    cost = np.empty_like(safety)
    completion = np.empty_like(safety)
    lookup = {
        (int(record["delay_steps"]), float(record["dropout_probability"])): record
        for record in records
    }
    for row, dropout in enumerate(dropouts):
        for column, delay in enumerate(delays):
            record = lookup[(delay, dropout)]
            safety[row, column] = float(record["safe_rate"])
            cost[row, column] = float(record["mean_completion_step_sum"])
            completion[row, column] = float(record["completion_rate"])

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.8))
    images = [
        axes[0].imshow(safety, vmin=0.0, vmax=1.0, cmap="RdYlGn"),
        axes[1].imshow(cost, cmap="viridis_r"),
    ]
    titles = ["Safe Trial Rate", "Mean Completion Step Sum"]
    formats = [lambda value: f"{value:.0%}", lambda value: f"{value:.0f}"]
    for axis, values, image, title, formatter in zip(
        axes, (safety, cost), images, titles, formats, strict=True
    ):
        axis.set_title(title)
        axis.set_xticks(range(len(delays)), labels=delays)
        axis.set_yticks(range(len(dropouts)), labels=[f"{v:.0%}" for v in dropouts])
        axis.set_xlabel("communication delay [steps]")
        axis.set_ylabel("packet dropout probability")
        for row in range(len(dropouts)):
            for column in range(len(delays)):
                label = formatter(values[row, column])
                if axis is axes[1]:
                    flags = []
                    if safety[row, column] < 1.0:
                        flags.append(f"safe {safety[row, column]:.0%}")
                    if completion[row, column] < 1.0:
                        flags.append(f"done {completion[row, column]:.0%}")
                    if flags:
                        label = "\n".join((label, *flags))
                text_color = "black"
                if axis is axes[1]:
                    value_range = np.ptp(values)
                    normalized = (
                        0.0
                        if value_range <= np.finfo(float).eps
                        else (values[row, column] - np.min(values)) / value_range
                    )
                    text_color = "white" if normalized > 0.45 else "black"
                axis.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=7.5 if axis is axes[1] else 8,
                )
        fig.colorbar(image, ax=axis, shrink=0.82)
    fig.suptitle("ETA-Priority Harbor Communication Robustness", fontsize=15)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def _draw_paths(axis, result: HarborResult, agents: list[HarborAgent]) -> None:
    for agent in agents:
        positions = result.positions[agent.name]
        color = _COLORS[agent.model.kind]
        linestyle = "--" if agent.model.kind == "rov" else "-"
        axis.plot(
            positions[:, 0],
            positions[:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=2.4,
            label=f"{_LABELS[agent.model.kind]} path",
        )
        axis.scatter(positions[0, 0], positions[0, 1], color=color, marker="o", s=65)
        axis.scatter(agent.goal[0], agent.goal[1], color=color, marker="*", s=170)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper left")


def _draw_harbor_map(axis, config) -> None:
    """Draw the configured water, shoreline/quay, and ground regions."""
    x_min, x_max = config.world_x_bounds
    y_min, y_max = config.world_y_bounds
    axis.axhspan(y_min, config.shoreline_y, color="#dceff7", zorder=-4)
    axis.axhspan(config.shoreline_y, y_max, color="#e5e8df", zorder=-4)
    axis.axhline(config.shoreline_y, color="#667078", linewidth=5, zorder=-2)
    axis.text(
        x_min + 0.25,
        config.shoreline_y + 0.18,
        "quay / ground",
        color="#394047",
        fontsize=9,
    )
    axis.text(x_min + 0.25, y_min + 0.25, "harbor water", color="#32789a", fontsize=9)
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
