"""Static and animated diagnostics for heterogeneous harbor rollouts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.plotting._backend import configure_matplotlib

from .simulation import HarborAgent, HarborResult

configure_matplotlib()

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


_PALETTES = {
    "ugv": ("#248f4b", "#82a832", "#146b3a"),
    "usv": ("#1f77b4", "#17a2b8"),
    "rov": ("#8e5bb7", "#6f42c1"),
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
    fig = plt.figure(figsize=(6.4 * len(results), 8.0))
    grid = fig.add_gridspec(2, len(results), height_ratios=(3.2, 1.0))
    axes = np.asarray([fig.add_subplot(grid[0, index]) for index in range(len(results))])
    diagnostics = grid[1, :].subgridspec(1, 2)
    depth_axis = fig.add_subplot(diagnostics[0, 0])
    attitude_axis = fig.add_subplot(diagnostics[0, 1])
    colors = _agent_colors(agents)
    for axis, (label, result) in zip(axes, results.items(), strict=True):
        _draw_harbor_map(axis, simulation_config)
        _draw_paths(axis, result, agents, colors)
        axis.set_title(
            f"{label}\nviolations={result.pairwise_violation_count}, "
            f"min distance={result.min_pairwise_distance:.3f} m"
        )
        for agent in agents:
            if agent.model.kind != "rov":
                continue
            depth_axis.plot(
                result.positions[agent.name][:, 2],
                linewidth=2.0,
                linestyle="--" if label == "independent" else "-",
                color="#777777" if label == "independent" else colors[agent.name],
                label=f"{label} ROV depth",
            )
    depth_axis.axhline(0.0, color="#4aa3df", linewidth=1.2, label="water surface")
    depth_axis.axhline(
        simulation_config.seabed_z,
        color="#665544",
        linewidth=1.8,
        label="seabed",
    )
    depth_axis.set_title("ROV Dive Profile")
    depth_axis.set_xlabel("simulation step")
    depth_axis.set_ylabel("z [m]")
    depth_axis.set_ylim(simulation_config.seabed_z - 0.2, 0.35)
    depth_axis.grid(True, alpha=0.3)
    depth_axis.legend(loc="lower right", fontsize=8)
    reference_label, reference_result = list(results.items())[-1]
    for agent in agents:
        if agent.model.kind != "rov":
            continue
        attitude = reference_result.states[agent.name][:, 3:6]
        for index, (name, color) in enumerate(
            (("roll", "#d95f02"), ("pitch", "#1b9e77"), ("yaw", "#7570b3"))
        ):
            attitude_axis.plot(attitude[:, index], color=color, label=name)
            attitude_axis.axhline(
                agent.goal[index + 3], color=color, linestyle=":", alpha=0.55
            )
    attitude_axis.set_title(f"ROV Attitude ({reference_label})")
    attitude_axis.set_xlabel("simulation step")
    attitude_axis.set_ylabel("angle [rad]")
    attitude_axis.grid(True, alpha=0.3)
    attitude_axis.legend(loc="upper right", ncol=3, fontsize=8)
    handles = []
    labels = []
    for axis in axes:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels, strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.suptitle("Untethered Heterogeneous Harbor Coordination", fontsize=16)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=4)
    fig.subplots_adjust(top=0.82, bottom=0.08, hspace=0.38, wspace=0.25)
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
    """Save XY-yaw and ROV x-z-pitch views of a coordinated rollout."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, (map_axis, depth_axis) = plt.subplots(
        1, 2, figsize=(11.0, 5.2), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )
    fig.subplots_adjust(bottom=0.19, wspace=0.25)
    models = {agent.name: agent.model for agent in agents}
    by_name = {agent.name: agent for agent in agents}
    colors = _agent_colors(agents)
    radii = {agent.name: agent.radius for agent in agents}
    goals = {agent.name: agent.goal for agent in agents}
    names = [agent.name for agent in agents]
    rov_agent = next(agent for agent in agents if agent.model.kind == "rov")
    frames = max(len(result.positions[name]) for name in names)
    accepted_safe_set = result.all_goals_reached and result.pairwise_violation_count == 0

    def update(frame: int):
        map_axis.clear()
        depth_axis.clear()
        _draw_harbor_map(map_axis, simulation_config)
        for name in names:
            kind = models[name].kind
            color = colors[name]
            positions = result.positions[name]
            index = min(frame, len(positions) - 1)
            shown = positions[: index + 1]
            if accepted_safe_set:
                samples = positions[::10]
                map_axis.scatter(
                    samples[:, 0],
                    samples[:, 1],
                    facecolors="white",
                    edgecolors=color,
                    linewidths=0.7,
                    s=15,
                    alpha=0.45,
                )
            map_axis.plot(shown[:, 0], shown[:, 1], color=color, linewidth=2)
            map_axis.scatter(
                shown[-1, 0],
                shown[-1, 1],
                color=color,
                s=90,
                label=_display_name(by_name[name]),
            )
            map_axis.scatter(
                goals[name][0], goals[name][1], color=color, marker="*", s=150
            )
            footprint = plt.Circle(
                shown[-1, :2], radii[name], facecolor=color, alpha=0.12, edgecolor=color
            )
            map_axis.add_patch(footprint)
            state = result.states[name][index]
            yaw = state[5] if kind == "rov" else state[2]
            map_axis.quiver(
                shown[-1, 0],
                shown[-1, 1],
                0.5 * np.cos(yaw),
                0.5 * np.sin(yaw),
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color=color,
                width=0.008,
                zorder=5,
            )
            for waypoint in by_name[name].route[:-1]:
                map_axis.scatter(
                    waypoint[0],
                    waypoint[1],
                    color=color,
                    marker="D",
                    s=38,
                    alpha=0.8,
                )
        map_axis.set_title(f"ETA-priority coordination, step {frame}")
        map_axis.set_xlabel("x [m]")
        map_axis.set_ylabel("y [m]")
        map_axis.grid(True, alpha=0.3)
        map_axis.set_aspect("equal", adjustable="box")
        map_axis.set_xlim(*simulation_config.world_x_bounds)
        map_axis.set_ylim(*simulation_config.world_y_bounds)
        map_axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=4,
            fontsize=8,
            framealpha=0.95,
        )
        rov_positions = result.positions[rov_agent.name]
        rov_index = min(frame, len(rov_positions) - 1)
        rov_shown = rov_positions[: rov_index + 1]
        rov_color = colors[rov_agent.name]
        depth_axis.plot(
            rov_shown[:, 0], rov_shown[:, 2], color=rov_color, linewidth=2.4
        )
        depth_axis.scatter(
            rov_shown[-1, 0], rov_shown[-1, 2], color=rov_color, s=90
        )
        depth_axis.scatter(
            rov_agent.goal[0],
            rov_agent.goal[2],
            color=rov_color,
            marker="*",
            s=150,
        )
        for waypoint in rov_agent.route[:-1]:
            depth_axis.scatter(
                waypoint[0], waypoint[2], color=rov_color, marker="D", s=42
            )
        pitch = result.states[rov_agent.name][rov_index, 4]
        depth_axis.quiver(
            rov_shown[-1, 0],
            rov_shown[-1, 2],
            0.65 * np.cos(pitch),
            0.65 * np.sin(pitch),
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=rov_color,
            width=0.012,
            zorder=5,
        )
        depth_axis.axhline(0.0, color="#4aa3df", linewidth=1.2, alpha=0.7)
        depth_axis.axhline(
            simulation_config.seabed_z,
            color="#665544",
            linewidth=2.0,
            alpha=0.8,
        )
        depth_axis.set_title(f"ROV Side Elevation, pitch={pitch:+.2f} rad")
        depth_axis.set_xlabel("x [m]")
        depth_axis.set_ylabel("z [m]")
        depth_axis.set_xlim(*simulation_config.world_x_bounds)
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


def _draw_paths(
    axis,
    result: HarborResult,
    agents: list[HarborAgent],
    colors: dict[str, str],
) -> None:
    accepted_safe_set = result.all_goals_reached and result.pairwise_violation_count == 0
    safety_label_pending = True
    safe_set_label_pending = accepted_safe_set
    for agent in agents:
        positions = result.positions[agent.name]
        color = colors[agent.name]
        linestyle = "--" if agent.model.kind == "rov" else "-"
        for point in positions[::10]:
            axis.add_patch(
                plt.Circle(
                    point[:2],
                    agent.radius,
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.035,
                    label=("physical safety envelope" if safety_label_pending else None),
                    zorder=0,
                )
            )
            safety_label_pending = False
        axis.plot(
            positions[:, 0],
            positions[:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=2.4,
            label=f"{_display_name(agent)} path",
        )
        if accepted_safe_set:
            samples = positions[::10]
            axis.scatter(
                samples[:, 0],
                samples[:, 1],
                facecolors="white",
                edgecolors=color,
                linewidths=0.8,
                s=18,
                alpha=0.75,
                label=(
                    "successful rollout samples" if safe_set_label_pending else None
                ),
                zorder=2,
            )
            safe_set_label_pending = False
        axis.scatter(positions[0, 0], positions[0, 1], color=color, marker="o", s=65)
        axis.scatter(agent.goal[0], agent.goal[1], color=color, marker="*", s=170)
        yaw = agent.goal[2] if agent.model.pose_dim == 3 else agent.goal[5]
        axis.quiver(
            agent.goal[0],
            agent.goal[1],
            0.45 * np.cos(yaw),
            0.45 * np.sin(yaw),
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=color,
            width=0.006,
            zorder=4,
        )
        if len(agent.route) > 1:
            waypoints = agent.route[:-1]
            axis.scatter(
                waypoints[:, 0],
                waypoints[:, 1],
                color=color,
                marker="D",
                s=42,
                label=f"{_display_name(agent)} waypoint",
            )
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.3)


def _agent_colors(agents: list[HarborAgent]) -> dict[str, str]:
    """Assign stable distinct colors to agents of the same platform kind."""
    counts: dict[str, int] = {}
    colors = {}
    for agent in agents:
        index = counts.get(agent.model.kind, 0)
        palette = _PALETTES[agent.model.kind]
        colors[agent.name] = palette[index % len(palette)]
        counts[agent.model.kind] = index + 1
    return colors


def _display_name(agent: HarborAgent) -> str:
    suffix = agent.name.rsplit("_", 1)[-1]
    if suffix.isdigit():
        return f"{_LABELS[agent.model.kind]} {suffix}"
    return _LABELS[agent.model.kind]


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
