"""CLI entry point for config-driven manta APF/LMPC runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.config import DEFAULT_CONFIG_PATH, ProjectConfig, load_project_config
from scripts.learning import run_manta_lmpc
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


def parse_args() -> argparse.Namespace:
    """Parse CLI overrides for the YAML-backed manta LMPC run."""
    parser = argparse.ArgumentParser(description="Run config-driven manta APF/LMPC.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--apf-max-steps", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--mpc-horizon", type=int, default=None)
    parser.add_argument("--k-hull", type=int, default=None)
    parser.add_argument("--goal-tolerance", type=float, default=None)
    parser.add_argument("--make-video", action="store_true", default=None)
    parser.add_argument("--no-video", action="store_false", dest="make_video")
    parser.add_argument("--quiet", action="store_true", default=None)
    parser.add_argument("--verbose", action="store_false", dest="quiet")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="override output directory for this run",
    )
    parser.add_argument(
        "--stop-file",
        default="STOP_RUN",
        help="file path that stops the run when created",
    )
    return parser.parse_args()


def main() -> None:
    """Run APF/LMPC from YAML config and save all diagnostics."""
    args = parse_args()
    project_config = _load_effective_config(args)
    scenario = project_config.scenario
    lmpc_config = project_config.lmpc
    apf_config = project_config.apf

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = (
        Path(args.output_dir)
        if args.output_dir
        else project_config.output.root_dir
        / f"{project_config.output.run_prefix}_{timestamp}"
    )
    root.mkdir(parents=True, exist_ok=True)
    stop_path = _resolve_stop_file(args.stop_file)
    if stop_path.exists():
        stop_path.unlink()

    try:
        result = run_manta_lmpc(
            scenario,
            config=lmpc_config,
            apf_config=apf_config,
            dynamics_config=project_config.dynamics,
            should_stop=stop_path.exists,
            verbose=not project_config.quiet,
        )
    except KeyboardInterrupt:
        if stop_path.exists():
            stop_path.unlink()
        print("Stopped manta LMPC run.")
        raise SystemExit(130) from None

    _, final_states = history_to_tensor(result.final_history)
    final_controls = result.final_controls
    metrics = compute_rollout_metrics(
        states=final_states,
        goals=scenario.goals,
        safety_distance=scenario.safety_distance,
        dt=lmpc_config.dt,
        controls=final_controls,
        goal_tolerance=lmpc_config.goal_tolerance,
    )
    dists = pairwise_distances(final_states)
    report_histories = result.report_histories
    costs = cost_by_iteration(
        result.histories,
        scenario.goals,
        goal_tolerance=lmpc_config.goal_tolerance,
    )
    status_counts = _status_counts_by_iteration(result.statuses_by_iteration)
    report_costs = cost_by_iteration(
        report_histories,
        scenario.goals,
        goal_tolerance=project_config.plots.cost_goal_tolerance,
    )
    final_goal_errors = _final_goal_errors(final_states, scenario.goals)

    summary = {
        "scenario": scenario.name,
        "config": str(Path(args.config).resolve()),
        "iterations": lmpc_config.iterations,
        "dt": lmpc_config.dt,
        "prediction_horizon": lmpc_config.prediction_horizon,
        "k_hull": lmpc_config.k_hull,
        "max_steps": lmpc_config.max_steps,
        "apf_max_steps": apf_config.max_steps,
        "goal_tolerance": lmpc_config.goal_tolerance,
        "cost_goal_tolerance": project_config.plots.cost_goal_tolerance,
        "cost_by_iteration_goal_tolerance": lmpc_config.goal_tolerance,
        "plot_cost_goal_tolerance": project_config.plots.cost_goal_tolerance,
        "selected_iteration": result.selected_iteration,
        "success_by_iteration": result.success_by_iteration,
        "goal_reached_by_iteration": result.goal_reached_by_iteration,
        "learned_by_iteration": result.learned_by_iteration,
        "validation_by_iteration": [
            validation.to_dict() for validation in result.validation_by_iteration
        ],
        "status_counts_by_iteration": status_counts,
        "cost_by_iteration": {str(agent): values for agent, values in costs.items()},
        "report_cost_by_iteration": {
            str(agent): values for agent, values in report_costs.items()
        },
        "final_goal_error_by_agent": {
            str(agent): error for agent, error in enumerate(final_goal_errors)
        },
        "metrics_final_iteration": metrics.to_dict(),
    }
    with (root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _save_histories_csv(
        root / "states_by_iteration.csv", result.histories, lmpc_config.dt
    )
    plot_learning_progression(
        report_histories,
        scenario.goals,
        scenario.obstacle,
        str(root / "learning_progression.png"),
        goal_tolerance=lmpc_config.goal_tolerance,
        obstacle_padding=apf_config.obstacle_padding,
        safety_distance=scenario.safety_distance,
        statuses_by_iteration=result.report_statuses,
    )
    plot_cost_decrease(
        report_histories,
        scenario.goals,
        str(root / "cost_decrease.png"),
        goal_tolerance=project_config.plots.cost_goal_tolerance,
    )
    plot_trajectories(
        states=final_states,
        goals=scenario.goals,
        title=f"{scenario.name}: final manta LMPC iteration",
        out_path=str(root / "final_trajectories.png"),
        goal_tolerance=lmpc_config.goal_tolerance,
        obstacle=scenario.obstacle,
        obstacle_padding=apf_config.obstacle_padding,
        safety_distance=scenario.safety_distance,
        statuses=result.final_statuses,
    )
    plot_pairwise_distances(
        distances=dists,
        dt=lmpc_config.dt,
        safety_distance=scenario.safety_distance,
        title=f"{scenario.name}: pairwise distances",
        out_path=str(root / "pairwise_distances.png"),
    )
    if project_config.make_video:
        save_manta_animation(
            result.final_history,
            scenario.goals,
            scenario.obstacle,
            lmpc_config.dt,
            str(root / "final_iteration.gif"),
            fps=project_config.plots.animation_fps,
            goal_tolerance=lmpc_config.goal_tolerance,
            obstacle_padding=apf_config.obstacle_padding,
            safety_distance=scenario.safety_distance,
            statuses=result.final_statuses,
        )

    print(f"Saved manta LMPC outputs to: {root}")


def _load_effective_config(args: argparse.Namespace) -> ProjectConfig:
    """Load YAML config, then apply explicit CLI overrides."""
    config = load_project_config(args.config, scenario_name=args.scenario)
    lmpc_updates = {
        "iterations": args.iterations,
        "max_steps": args.max_steps,
        "dt": args.dt,
        "prediction_horizon": args.mpc_horizon,
        "k_hull": args.k_hull,
        "goal_tolerance": args.goal_tolerance,
    }
    lmpc_updates = {
        key: value for key, value in lmpc_updates.items() if value is not None
    }
    apf_updates = {"max_steps": args.apf_max_steps}
    apf_updates = {
        key: value for key, value in apf_updates.items() if value is not None
    }

    make_video = config.make_video if args.make_video is None else args.make_video
    quiet = config.quiet if args.quiet is None else args.quiet

    return replace(
        config,
        lmpc=replace(config.lmpc, **lmpc_updates),
        apf=replace(config.apf, **apf_updates),
        make_video=make_video,
        quiet=quiet,
    )


def _resolve_stop_file(path: str | Path) -> Path:
    stop_path = Path(path)
    if not stop_path.is_absolute():
        stop_path = Path.cwd() / stop_path
    return stop_path


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
    goal_pos = np.asarray(goals, dtype=float)[:, :2]
    final_pos = np.asarray(states, dtype=float)[-1, :, :2]
    return np.linalg.norm(final_pos - goal_pos, axis=1).astype(float).tolist()


def _save_histories_csv(
    path: Path, histories: list[dict[int, np.ndarray]], dt: float
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
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
        for iteration, run in enumerate(histories):
            for agent in sorted(run):
                traj = np.asarray(run[agent], dtype=float)
                for step, state in enumerate(traj):
                    writer.writerow(
                        (iteration, step, step * dt, agent, *state.tolist())
                    )


if __name__ == "__main__":
    main()
