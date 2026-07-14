"""Prepare and save diagnostics for one manta APF/LMPC run."""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.config import ProjectConfig
from scripts.learning import MantaLMPCRunResult, summarize_optimizer_slack
from scripts.metrics import (
    compute_rollout_metrics,
    cost_by_iteration,
    history_to_tensor,
    pairwise_distances,
)
from scripts.plotting import (
    plot_cost_decrease,
    plot_learning_progression,
    plot_pairwise_distances,
    plot_trajectories,
    save_manta_animation,
)


@dataclass(frozen=True)
class MantaRunReport:
    """Prepared numerical data shared by serialization and plotting."""

    summary: dict[str, Any]
    final_states: np.ndarray
    pairwise_distances: np.ndarray
    report_histories: list[dict[int, np.ndarray]]


def prepare_manta_report(
    result: MantaLMPCRunResult,
    project_config: ProjectConfig,
    *,
    config_path: str | Path,
) -> MantaRunReport:
    """Compute the stable summary contract and plot-ready numerical arrays."""
    scenario = project_config.scenario
    lmpc = project_config.lmpc
    _, final_states = history_to_tensor(result.final_history)
    metrics = compute_rollout_metrics(
        states=final_states,
        goals=scenario.goals,
        safety_distance=scenario.safety_distance,
        dt=lmpc.dt,
        controls=result.final_controls,
        goal_tolerance=lmpc.goal_tolerance,
    )
    report_histories = result.report_histories
    costs = cost_by_iteration(
        result.histories,
        scenario.goals,
        goal_tolerance=lmpc.goal_tolerance,
    )
    report_costs = cost_by_iteration(
        report_histories,
        scenario.goals,
        goal_tolerance=project_config.plots.cost_goal_tolerance,
    )

    summary = {
        "scenario": scenario.name,
        "config": str(Path(config_path).resolve()),
        "iterations": lmpc.iterations,
        "dt": lmpc.dt,
        "prediction_horizon": lmpc.prediction_horizon,
        "k_hull": lmpc.k_hull,
        "max_steps": lmpc.max_steps,
        "apf_max_steps": project_config.apf.max_steps,
        "goal_tolerance": lmpc.goal_tolerance,
        "cost_goal_tolerance": project_config.plots.cost_goal_tolerance,
        "cost_by_iteration_goal_tolerance": lmpc.goal_tolerance,
        "plot_cost_goal_tolerance": project_config.plots.cost_goal_tolerance,
        "selected_iteration": result.selected_iteration,
        "success_by_iteration": result.success_by_iteration,
        "goal_reached_by_iteration": result.goal_reached_by_iteration,
        "learned_by_iteration": result.learned_by_iteration,
        "validation_by_iteration": [
            validation.to_dict() for validation in result.validation_by_iteration
        ],
        "status_counts_by_iteration": _status_counts_by_iteration(
            result.statuses_by_iteration
        ),
        "optimizer_slack_by_iteration": [
            summarize_optimizer_slack(np.zeros((0, len(scenario.starts), 2)))
        ]
        + [summarize_optimizer_slack(values) for values in result.slack_by_iteration],
        "cost_by_iteration": {
            str(agent): values for agent, values in costs.items()
        },
        "report_cost_by_iteration": {
            str(agent): values for agent, values in report_costs.items()
        },
        "final_goal_error_by_agent": {
            str(agent): error
            for agent, error in enumerate(
                _final_goal_errors(final_states, scenario.goals)
            )
        },
        "metrics_final_iteration": metrics.to_dict(),
    }
    return MantaRunReport(
        summary=summary,
        final_states=final_states,
        pairwise_distances=pairwise_distances(final_states),
        report_histories=report_histories,
    )


def save_manta_run_report(
    output_dir: str | Path,
    result: MantaLMPCRunResult,
    project_config: ProjectConfig,
    *,
    config_path: str | Path,
) -> MantaRunReport:
    """Save JSON, CSV, plots, and the optional GIF for a completed run."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    report = prepare_manta_report(result, project_config, config_path=config_path)
    scenario = project_config.scenario
    lmpc = project_config.lmpc
    apf = project_config.apf

    with (root / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(report.summary, file, indent=2)
    _save_histories_csv(
        root / "states_by_iteration.csv", result.histories, lmpc.dt
    )
    plot_learning_progression(
        report.report_histories,
        scenario.goals,
        scenario.obstacle,
        str(root / "learning_progression.png"),
        goal_tolerance=lmpc.goal_tolerance,
        obstacle_padding=apf.obstacle_padding,
        safety_distance=scenario.safety_distance,
        statuses_by_iteration=result.report_statuses,
    )
    plot_cost_decrease(
        report.report_histories,
        scenario.goals,
        str(root / "cost_decrease.png"),
        goal_tolerance=project_config.plots.cost_goal_tolerance,
    )
    plot_trajectories(
        states=report.final_states,
        goals=scenario.goals,
        title=f"{scenario.name}: final manta LMPC iteration",
        out_path=str(root / "final_trajectories.png"),
        goal_tolerance=lmpc.goal_tolerance,
        obstacle=scenario.obstacle,
        obstacle_padding=apf.obstacle_padding,
        safety_distance=scenario.safety_distance,
        statuses=result.final_statuses,
    )
    plot_pairwise_distances(
        distances=report.pairwise_distances,
        dt=lmpc.dt,
        safety_distance=scenario.safety_distance,
        title=f"{scenario.name}: pairwise distances",
        out_path=str(root / "pairwise_distances.png"),
    )
    if project_config.make_video:
        save_manta_animation(
            result.final_history,
            scenario.goals,
            scenario.obstacle,
            lmpc.dt,
            str(root / "final_iteration.gif"),
            fps=project_config.plots.animation_fps,
            goal_tolerance=lmpc.goal_tolerance,
            obstacle_padding=apf.obstacle_padding,
            safety_distance=scenario.safety_distance,
            statuses=result.final_statuses,
        )
    return report


def _status_counts_by_iteration(
    statuses_by_iteration: list[list[dict[int, str]]],
) -> list[dict[str, int]]:
    counts_by_iteration: list[dict[str, int]] = [{}]
    for iteration_statuses in statuses_by_iteration:
        counter: Counter[str] = Counter()
        for step_statuses in iteration_statuses:
            counter.update(step_statuses.values())
        counts_by_iteration.append(dict(sorted(counter.items())))
    return counts_by_iteration


def _final_goal_errors(states: np.ndarray, goals: np.ndarray) -> list[float]:
    goal_positions = np.asarray(goals, dtype=float)[:, :2]
    final_positions = np.asarray(states, dtype=float)[-1, :, :2]
    return np.linalg.norm(final_positions - goal_positions, axis=1).tolist()


def _save_histories_csv(
    path: Path,
    histories: list[dict[int, np.ndarray]],
    dt: float,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(
            (
                "iteration",
                "step",
                "time",
                "agent",
                "x",
                "y",
                "theta",
                "p_L",
                "q_L",
                "p_R",
                "q_R",
            )
        )
        for iteration, history in enumerate(histories):
            for agent in sorted(history):
                trajectory = np.asarray(history[agent], dtype=float)
                for step, state in enumerate(trajectory):
                    writer.writerow(
                        (iteration, step, step * dt, agent, *state.tolist())
                    )
