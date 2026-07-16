"""Static and animated diagnostics for heterogeneous harbor rollouts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.plotting._backend import configure_matplotlib

from .simulation import HarborAgent, HarborDisturbanceConfig, HarborResult

configure_matplotlib()

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D


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
    label: str = "ETA-priority coordination",
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
        map_axis.set_title(f"{label}, step {frame}")
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


def save_harbor_learning_progress(
    iterations,
    agents: list[HarborAgent],
    simulation_config,
    path: str | Path,
) -> Path:
    """Save one compact trajectory, performance, and solver-health dashboard."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = len(iterations)
    fig = plt.figure(figsize=(5.4 * count, 9.0))
    grid = fig.add_gridspec(2, count, height_ratios=(2.5, 1.0))
    colors = _agent_colors(agents)
    display_labels = {
        "guidance_iter_0": "Guidance seed",
        "distributed_mpc": "Distributed MPC",
    }
    for index, iteration in enumerate(iterations):
        axis = fig.add_subplot(grid[0, index])
        _draw_harbor_map(axis, simulation_config)
        if iteration.label.startswith("distributed_lmpc_"):
            prior = next(
                (
                    candidate
                    for candidate in reversed(iterations[:index])
                    if candidate.admitted
                ),
                None,
            )
            if prior is not None:
                for agent_index, agent in enumerate(agents):
                    safe_positions = prior.result.positions[agent.name]
                    axis.plot(
                        safe_positions[:, 0],
                        safe_positions[:, 1],
                        color=colors[agent.name],
                        linewidth=1.4,
                        linestyle=":",
                        alpha=0.55,
                        label=("prior admitted safe trajectory" if agent_index == 0 else None),
                    )
        _draw_paths(axis, iteration.result, agents, colors)
        label = display_labels.get(
            iteration.label,
            iteration.label.replace("distributed_lmpc_", "Distributed LMPC "),
        )
        axis.set_title(
            f"{label}\nsteps={iteration.completion_step_sum}, "
            f"min={iteration.result.min_pairwise_distance:.3f} m, "
            f"{'admitted' if iteration.admitted else 'rejected'}"
        )
        if index == count - 1:
            axis.legend(loc="upper left", fontsize=7, framealpha=0.92)

    labels = [
        display_labels.get(
            item.label,
            item.label.replace("distributed_lmpc_", "LMPC "),
        )
        for item in iterations
    ]
    x = np.arange(count)
    cost_axis = fig.add_subplot(grid[1, : max(1, count // 2)])
    costs = [item.completion_step_sum for item in iterations]
    bars = cost_axis.bar(
        x,
        costs,
        color=["#377eb8" if item.admitted else "#999999" for item in iterations],
        hatch=[None if item.admitted else "//" for item in iterations],
    )
    cost_axis.bar_label(bars, padding=3)
    baseline = costs[0]
    for index, cost in enumerate(costs[1:], start=1):
        if iterations[index].admitted:
            improvement = 100.0 * (baseline - cost) / baseline
            annotation = f"{improvement:.1f}% faster"
        else:
            annotation = "rejected"
        cost_axis.text(
            index,
            cost * 0.55,
            annotation,
            ha="center",
            color="white",
        )
    cost_axis.set_xticks(x, labels=labels, rotation=12, ha="right")
    cost_axis.set_ylabel("sum of first-goal steps")
    cost_axis.set_title("Task Completion Cost (lower is better)")
    cost_axis.grid(axis="y", alpha=0.25)

    health_axis = fig.add_subplot(grid[1, max(1, count // 2) :])
    separations = [item.result.min_pairwise_distance for item in iterations]
    health_bars = health_axis.bar(
        x - 0.18,
        separations,
        width=0.36,
        color=["#4daf4a" if item.admitted else "#999999" for item in iterations],
        hatch=[None if item.admitted else "//" for item in iterations],
    )
    health_axis.bar_label(health_bars, fmt="%.3f", padding=3)
    health_axis.set_ylabel("minimum swept separation [m]")
    health_axis.set_xticks(x, labels=labels, rotation=12, ha="right")
    health_axis.set_title("Safety and Solver Reliability")
    fallback_axis = health_axis.twinx()
    fallbacks = [item.solver_fallbacks for item in iterations]
    fallback_bars = fallback_axis.bar(
        x + 0.18, fallbacks, width=0.36, color="#e41a1c", alpha=0.75
    )
    fallback_axis.bar_label(fallback_bars, padding=3)
    fallback_axis.set_ylabel("solver fallbacks")
    fallback_axis.set_ylim(0.0, max(1.0, max(fallbacks, default=0) + 0.5))
    health_axis.grid(axis="y", alpha=0.25)

    fig.suptitle("Distributed Harbor LMPC Research Progress", fontsize=17)
    fig.subplots_adjust(top=0.91, bottom=0.12, hspace=0.35, wspace=0.28)
    fig.savefig(output, dpi=180, bbox_inches="tight")
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


def save_horizon_efficiency(
    records: list[dict[str, float | int | str | bool]], path: str | Path
) -> Path:
    """Save task-performance and solve-time curves for matched horizons."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    horizons = sorted({int(record["prediction_horizon"]) for record in records})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    styles = {
        "MPC": ("#377eb8", "o"),
        "LMPC": ("#e41a1c", "s"),
    }
    for controller, (color, marker) in styles.items():
        selected = {
            int(record["prediction_horizon"]): record
            for record in records
            if record["controller"] == controller
        }
        costs = [float(selected[horizon]["completion_step_sum"]) for horizon in horizons]
        times = [float(selected[horizon]["mean_solve_time_ms"]) for horizon in horizons]
        axes[0].plot(
            horizons,
            costs,
            color=color,
            marker=marker,
            linewidth=2.2,
            label=controller,
        )
        axes[1].plot(
            horizons,
            times,
            color=color,
            marker=marker,
            linewidth=2.2,
            label=controller,
        )
        for horizon, cost in zip(horizons, costs, strict=True):
            record = selected[horizon]
            suffix = "" if record["complete"] else "\nincomplete"
            axes[0].annotate(
                f"{cost:.0f}{suffix}",
                (horizon, cost),
                xytext=(0, 8 if controller == "MPC" else -16),
                textcoords="offset points",
                ha="center",
                color=color,
                fontsize=8,
            )
    axes[0].set_title("Completion Cost and Liveness")
    axes[0].set_ylabel("sum of first-goal steps")
    axes[1].set_title("Mean Per-Agent NLP Solve Latency")
    axes[1].set_ylabel("milliseconds per solve")
    for axis in axes:
        axis.set_xlabel("prediction horizon N")
        axis.set_xticks(horizons)
        axis.margins(y=0.12)
        axis.grid(True, alpha=0.28)
        axis.legend()
    fig.suptitle("Horizon-Dependent MPC and LMPC Performance", fontsize=15)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def save_model_mismatch_diagnostics(
    trials,
    agents: list[HarborAgent],
    simulation_config,
    disturbance: HarborDisturbanceConfig,
    path: str | Path,
) -> Path:
    """Show robust paths, terminal error, and both local estimate histories."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18.0, 8.2))
    grid = fig.add_gridspec(
        2, 3, width_ratios=(1.12, 1.0, 0.9), hspace=0.34, wspace=0.27
    )
    map_axis = fig.add_subplot(grid[:, 0])
    error_axis = fig.add_subplot(grid[0, 1])
    residual_axis = fig.add_subplot(grid[1, 1])
    effectiveness_axis = fig.add_subplot(grid[:, 2])
    colors = _agent_colors(agents)
    nominal = next(trial for trial in trials if trial.label == "Nominal MPC")
    adaptive = next(trial for trial in trials if trial.label == "Joint-adaptive LMPC")
    _draw_harbor_map(map_axis, simulation_config)
    for agent in agents:
        color = colors[agent.name]
        for trial, style, alpha in ((nominal, "--", 0.5), (adaptive, "-", 1.0)):
            positions = trial.result.positions[agent.name]
            map_axis.plot(
                positions[:, 0],
                positions[:, 1],
                color=color,
                linestyle=style,
                linewidth=1.7 if style == "--" else 2.5,
                alpha=alpha,
                label=(
                    f"{_display_name(agent)}: {trial.label}"
                    if agent.model.kind in {"usv", "rov"}
                    else None
                ),
            )
        map_axis.scatter(agent.goal[0], agent.goal[1], color=color, marker="*", s=120)
    current = np.asarray(disturbance.water_current, dtype=float)
    map_axis.quiver(
        -4.4,
        -4.3,
        6.0 * current[0],
        6.0 * current[1],
        color="#222222",
        width=0.007,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        label="hidden water current",
    )
    map_axis.set_title("Executed Paths Under the Same Hidden Plant Mismatch")
    map_axis.set_xlabel("x [m]")
    map_axis.set_ylabel("y [m]")
    map_axis.set_aspect("equal", adjustable="box")
    map_axis.grid(True, alpha=0.25)
    map_axis.legend(loc="upper left", fontsize=8)

    marine = [agent for agent in agents if agent.model.kind in {"usv", "rov"}]
    labels = [
        trial.label.replace("-adaptive ", " ").replace("Nominal MPC", "Nominal")
        for trial in trials
    ]
    x_values = np.arange(len(labels))
    width = 0.34
    for index, agent in enumerate(marine):
        errors = [trial.result.final_goal_errors[agent.name] for trial in trials]
        bars = error_axis.bar(
            x_values + (index - 0.5) * width,
            errors,
            width,
            color=colors[agent.name],
            label=_display_name(agent),
        )
        for trial, bar in zip(trials, bars, strict=True):
            if not trial.valid:
                bar.set_hatch("//")
                bar.set_edgecolor("#333333")
    error_axis.axhline(
        simulation_config.goal_tolerance,
        color="#555555",
        linestyle=":",
        label="position tolerance",
    )
    error_axis.set_title(
        f"Terminal Error After {disturbance.evaluation_hold_steps}-Step Hold"
    )
    error_axis.set_ylabel("position error [m]")
    error_axis.set_xticks(x_values, labels, rotation=12, ha="right")
    error_axis.grid(True, axis="y", alpha=0.25)
    error_axis.legend(fontsize=8)

    component_styles = ((0, "x", "-"), (1, "y", "--"), (2, "z", ":"))
    for agent in marine:
        history = adaptive.residual_history[agent.name]
        for component, component_name, style in component_styles:
            if agent.model.kind == "usv" and component == 2:
                continue
            residual_axis.plot(
                history[:, component],
                color=colors[agent.name],
                linestyle=style,
                linewidth=1.8,
                label=f"{_display_name(agent)} {component_name}",
            )
            truth = current[component]
            residual_axis.axhline(truth, color="#444444", linestyle=style, alpha=0.3)
    residual_axis.set_title("Locally Estimated Position-Velocity Residual")
    residual_axis.set_xlabel("control update")
    residual_axis.set_ylabel("estimated residual [m/s]")
    residual_axis.grid(True, alpha=0.25)
    residual_axis.legend(ncol=2, fontsize=8)

    true_effectiveness = {
        agent.name: disturbance.effectiveness(agent.model, agent.name)
        for agent in agents
    }
    for agent in agents:
        history = adaptive.effectiveness_history[agent.name]
        effectiveness_axis.plot(
            np.mean(history, axis=1),
            color=colors[agent.name],
            linewidth=2.0,
            label=f"{_display_name(agent)} estimate",
        )
        effectiveness_axis.axhline(
            float(np.mean(true_effectiveness[agent.name])),
            color=colors[agent.name],
            linestyle=":",
            linewidth=1.3,
            alpha=0.7,
        )
    effectiveness_axis.set_title("Locally Estimated Control Effectiveness")
    effectiveness_axis.set_xlabel("control update")
    effectiveness_axis.set_ylabel("applied / commanded control")
    effectiveness_axis.set_ylim(
        min(float(np.min(value)) for value in true_effectiveness.values()) - 0.04,
        1.03,
    )
    effectiveness_axis.grid(True, alpha=0.25)
    effectiveness_axis.legend(fontsize=8)
    effectiveness_axis.text(
        0.03,
        0.03,
        "dotted lines: hidden plant values\nhatched bars: invalid trial",
        transform=effectiveness_axis.transAxes,
        fontsize=8,
        color="#444444",
    )
    fig.suptitle(
        "Distributed Harbor MPC Under Current and Actuator Mismatch", fontsize=15
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def save_actuator_fault_diagnostics(
    trials,
    agents: list[HarborAgent],
    simulation_config,
    disturbance: HarborDisturbanceConfig,
    path: str | Path,
) -> Path:
    """Show asymmetric-fault paths, cost, convergence, and channel estimates."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18.0, 8.4))
    grid = fig.add_gridspec(
        2, 3, width_ratios=(1.05, 0.82, 1.18), hspace=0.34, wspace=0.30
    )
    map_axis = fig.add_subplot(grid[:, 0])
    cost_axis = fig.add_subplot(grid[0, 1])
    rmse_axis = fig.add_subplot(grid[1, 1])
    matrix_axis = fig.add_subplot(grid[:, 2])
    colors = _agent_colors(agents)
    scalar = next(trial for trial in trials if trial.label == "Scalar-adaptive MPC")
    diagonal = next(
        trial for trial in trials if trial.label == "Diagonal-adaptive MPC"
    )
    _draw_harbor_map(map_axis, simulation_config)
    for agent in agents:
        for trial, style, alpha in ((scalar, "--", 0.55), (diagonal, "-", 1.0)):
            positions = trial.result.positions[agent.name]
            map_axis.plot(
                positions[:, 0],
                positions[:, 1],
                color=colors[agent.name],
                linestyle=style,
                linewidth=1.7 if style == "--" else 2.5,
                alpha=alpha,
            )
        map_axis.scatter(
            agent.goal[0], agent.goal[1], color=colors[agent.name], marker="*", s=115
        )
    map_axis.set_title("Executed Paths Under Identical Asymmetric Faults")
    map_axis.set_xlabel("x [m]")
    map_axis.set_ylabel("y [m]")
    map_axis.set_aspect("equal", adjustable="box")
    map_axis.grid(True, alpha=0.25)
    platform_handles = [
        Line2D(
            [0],
            [0],
            color=colors[agent.name],
            linewidth=2.5,
            label=_short_platform_name(agent),
        )
        for agent in agents
    ]
    style_handles = [
        Line2D([0], [0], color="#333333", linestyle="--", label="scalar MPC"),
        Line2D([0], [0], color="#333333", linestyle="-", label="diagonal MPC"),
    ]
    map_axis.legend(
        handles=[*platform_handles, *style_handles],
        loc="upper left",
        fontsize=7,
        ncol=2,
    )

    labels = [trial.label.replace("-adaptive", "") for trial in trials]
    bars = cost_axis.bar(
        np.arange(len(trials)),
        [trial.completion_step_sum for trial in trials],
        color=["#8c8c8c", "#d8a02b", "#2c9c69", "#2878b5"],
    )
    for trial, bar in zip(trials, bars, strict=True):
        cost_axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(trial.completion_step_sum),
            ha="center",
            va="bottom",
            fontsize=8,
        )
        if not trial.valid:
            bar.set_hatch("//")
            bar.set_edgecolor("#333333")
    cost_axis.set_title("Completion Cost")
    cost_axis.set_ylabel("sum of first-goal steps")
    cost_axis.set_xticks(np.arange(len(labels)), labels, rotation=14, ha="right")
    cost_axis.grid(True, axis="y", alpha=0.25)

    adaptive_trials = [trial for trial in trials if "adaptive" in trial.label]
    for trial, color in zip(
        adaptive_trials, ("#d8a02b", "#2c9c69", "#2878b5"), strict=True
    ):
        errors = _effectiveness_rmse_history(trial, agents, disturbance)
        rmse_axis.plot(errors, color=color, linewidth=2.0, label=trial.label)
    rmse_axis.set_title("Local Effectiveness Estimation Error")
    rmse_axis.set_xlabel("control update")
    rmse_axis.set_ylabel("channel RMSE")
    rmse_axis.grid(True, alpha=0.25)
    rmse_axis.legend(fontsize=8)

    row_labels = []
    truth_values = []
    for agent in agents:
        truth = disturbance.effectiveness(agent.model, agent.name)
        for channel, value in zip(
            _control_channel_labels(agent), truth, strict=True
        ):
            row_labels.append(f"{_short_platform_name(agent)} {channel}")
            truth_values.append(value)
    columns = [
        ("hidden", np.asarray(truth_values)),
        (
            "scalar MPC",
            _flatten_effectiveness(scalar.final_effectiveness_estimates, agents),
        ),
        (
            "diagonal MPC",
            _flatten_effectiveness(diagonal.final_effectiveness_estimates, agents),
        ),
        (
            "diagonal LMPC",
            _flatten_effectiveness(
                next(
                    trial
                    for trial in trials
                    if trial.label == "Diagonal-adaptive LMPC"
                ).final_effectiveness_estimates,
                agents,
            ),
        ),
    ]
    matrix = np.column_stack([values for _, values in columns])
    image = matrix_axis.imshow(
        matrix,
        cmap="viridis",
        vmin=min(0.5, float(np.min(matrix))),
        vmax=1.0,
        aspect="auto",
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            matrix_axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value < 0.78 else "#111111",
                fontsize=8,
            )
    matrix_axis.set_title("Hidden and Final Channel Effectiveness")
    matrix_axis.set_xticks(
        np.arange(len(columns)), [label for label, _ in columns], rotation=12
    )
    matrix_axis.set_yticks(np.arange(len(row_labels)), row_labels)
    fig.colorbar(image, ax=matrix_axis, fraction=0.045, pad=0.03)
    fig.suptitle(
        "Local Diagonal Fault Identification for Distributed Harbor MPC",
        fontsize=15,
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def _effectiveness_rmse_history(trial, agents, disturbance) -> np.ndarray:
    lengths = [len(trial.effectiveness_history[agent.name]) for agent in agents]
    count = min(lengths, default=0)
    values = []
    for index in range(count):
        error = np.concatenate(
            [
                trial.effectiveness_history[agent.name][index]
                - disturbance.effectiveness(agent.model, agent.name)
                for agent in agents
            ]
        )
        values.append(float(np.sqrt(np.mean(error * error))))
    return np.asarray(values, dtype=float)


def _flatten_effectiveness(estimates, agents: list[HarborAgent]) -> np.ndarray:
    return np.concatenate(
        [np.asarray(estimates[agent.name], dtype=float) for agent in agents]
    )


def _control_channel_labels(agent: HarborAgent) -> tuple[str, ...]:
    if agent.model.kind == "ugv":
        return ("F", "N")
    if agent.model.kind == "usv":
        return ("T", "N")
    return ("X", "Y", "Z", "K", "M", "N")


def _short_platform_name(agent: HarborAgent) -> str:
    if agent.name == "ground_rover_1":
        return "RobEn"
    if agent.name == "ground_rover_2":
        return "Inspector-Gadget"
    if agent.model.kind == "usv":
        return "Heron"
    return "BlueROV2"


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
    if agent.display_name is not None:
        return agent.display_name
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
