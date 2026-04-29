"""Run closed-loop decentralized baseline MPC for the 3-agent scenarios."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.metrics import compute_rollout_metrics, pairwise_distances
from src.mpc import MPCController
from src.plotting import plot_pairwise_distances, plot_trajectories, save_rollout_animation
from src.simulation import EnvConfig, ThreeAgentSingleIntegratorEnv, get_scenario, list_scenarios


BASELINE_COLLISION_DEFAULTS = {
    "crossing_paths": ("soft_penalty", 10000.0),
}


def parse_args() -> argparse.Namespace:
    """Parse baseline MPC run arguments."""
    parser = argparse.ArgumentParser(description="Run decentralized baseline MPC.")
    parser.add_argument("--scenario", default="all", help="scenario name or 'all'")
    parser.add_argument("--horizon", type=int, default=80, help="closed-loop simulation horizon")
    parser.add_argument("--dt", type=float, default=0.1, help="simulation time step")
    parser.add_argument("--mpc-horizon", type=int, default=20, help="MPC prediction horizon")
    parser.add_argument("--u-max", type=float, default=1.0, help="componentwise input limit")
    parser.add_argument(
        "--collision-mode",
        choices=("hard_linearized", "soft_penalty"),
        default=None,
        help="override the scenario baseline collision mode",
    )
    parser.add_argument("--collision-soft-weight", type=float, default=None, help="override soft collision slack penalty")
    parser.add_argument("--make-video", action="store_true", help="save one trajectory GIF per scenario")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="override output directory (default: results/baseline/baseline_<timestamp>)",
    )
    return parser.parse_args()


def main() -> None:
    """Execute closed-loop MPC runs and save metrics, CSVs, and plots."""
    args = parse_args()
    names = list_scenarios() if args.scenario == "all" else [args.scenario]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.output_dir) if args.output_dir else Path("results") / "baseline" / f"baseline_{timestamp}"
    root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, object]] = {}

    for name in names:
        scenario = get_scenario(name)
        default_collision_mode, default_soft_weight = _baseline_collision_defaults(name)
        collision_mode = args.collision_mode or default_collision_mode
        collision_soft_weight = (
            args.collision_soft_weight if args.collision_soft_weight is not None else default_soft_weight
        )
        run_dir = root / name
        run_dir.mkdir(parents=True, exist_ok=True)

        controller = MPCController(
            dt=args.dt,
            horizon=args.mpc_horizon,
            u_max=args.u_max,
            collision_mode=collision_mode,
            collision_soft_weight=collision_soft_weight,
        )

        env = ThreeAgentSingleIntegratorEnv(
            starts=scenario.starts,
            goals=scenario.goals,
            config=EnvConfig(dt=args.dt, horizon=args.horizon),
        )

        states, controls, solver_statuses = _closed_loop_rollout(
            env=env,
            controller=controller,
            goals=scenario.goals,
            safety_distance=scenario.safety_distance,
            prediction_horizon=args.mpc_horizon,
        )

        dists = pairwise_distances(states)
        metrics = compute_rollout_metrics(
            states=states,
            goals=scenario.goals,
            safety_distance=scenario.safety_distance,
            dt=args.dt,
            controls=controls,
        )
        run_metrics = _run_record(
            metrics=metrics.to_dict(),
            states=states,
            goals=scenario.goals,
            starts=scenario.starts,
            safety_distance=scenario.safety_distance,
            prediction_horizon=args.mpc_horizon,
            solver_statuses=solver_statuses,
            collision_mode=collision_mode,
            collision_soft_weight=collision_soft_weight,
        )
        summary[name] = run_metrics

        _save_states_csv(run_dir / "states.csv", states, args.dt)
        _save_controls_csv(run_dir / "controls.csv", controls, args.dt)
        with (run_dir / "solver_statuses.json").open("w", encoding="utf-8") as f:
            json.dump(_solver_diagnostics(solver_statuses, collision_mode, collision_soft_weight), f, indent=2)
        with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(run_metrics, f, indent=2)

        plot_trajectories(
            states=states,
            goals=scenario.goals,
            title=f"{name}: baseline decentralized MPC",
            out_path=str(run_dir / "trajectories.png"),
        )
        plot_pairwise_distances(
            distances=dists,
            dt=args.dt,
            safety_distance=scenario.safety_distance,
            title=f"{name}: pairwise distances (baseline MPC)",
            out_path=str(run_dir / "pairwise_distances.png"),
        )
        if args.make_video:
            save_rollout_animation(
                states=states,
                goals=scenario.goals,
                safety_distance=scenario.safety_distance,
                dt=args.dt,
                out_path=str(run_dir / "animation.gif"),
            )

    with (root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved baseline MPC outputs to: {root}")


def _closed_loop_rollout(
    env: ThreeAgentSingleIntegratorEnv,
    controller: MPCController,
    goals: np.ndarray,
    safety_distance: float,
    prediction_horizon: int,
) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
    current = env.reset()
    states = np.zeros((env.config.horizon + 1, 3, 2), dtype=float)
    controls = np.zeros((env.config.horizon, 3, 2), dtype=float)
    states[0] = current

    reference = _goal_reference(current, goals, prediction_horizon)
    solver_statuses: list[list[str]] = []
    for k in range(env.config.horizon):
        predictions = np.zeros_like(reference)
        step_statuses: list[str] = []
        for agent_index in range(3):
            controls[k, agent_index] = controller.solve(
                agent_index=agent_index,
                current_states=current,
                goals=goals,
                reference_trajectories=reference,
                safety_distance=safety_distance,
            )
            prediction = controller.last_prediction
            predictions[:, agent_index] = reference[:, agent_index] if prediction is None else prediction
            step_statuses.append(controller.last_status)

        current = env.step(controls[k])
        states[k + 1] = current
        reference = _shift_reference(predictions, current, goals)
        solver_statuses.append(step_statuses)

    return states, controls, solver_statuses


def _baseline_collision_defaults(scenario_name: str) -> tuple[str, float]:
    return BASELINE_COLLISION_DEFAULTS.get(scenario_name, ("hard_linearized", 200.0))


def _goal_reference(current_states: np.ndarray, goals: np.ndarray, horizon: int) -> np.ndarray:
    reference = np.zeros((horizon + 1, 3, 2), dtype=float)
    alpha = np.linspace(0.0, 1.0, horizon + 1)
    for agent_index in range(3):
        reference[:, agent_index] = (1.0 - alpha[:, None]) * current_states[agent_index] + alpha[:, None] * goals[
            agent_index
        ]
    return reference


def _shift_reference(predictions: np.ndarray, current_states: np.ndarray, goals: np.ndarray) -> np.ndarray:
    shifted = np.empty_like(predictions)
    shifted[0] = current_states
    shifted[1:-1] = predictions[2:]
    shifted[-1] = goals
    return shifted


def _run_record(
    metrics: dict[str, float | int | None],
    states: np.ndarray,
    goals: np.ndarray,
    starts: np.ndarray,
    safety_distance: float,
    prediction_horizon: int,
    solver_statuses: list[list[str]],
    collision_mode: str,
    collision_soft_weight: float,
) -> dict[str, object]:
    final_goal_error = np.linalg.norm(states[-1] - goals, axis=1)
    flat_statuses = [status for step in solver_statuses for status in step]
    feasible = [status in ("optimal", "optimal_inaccurate") for status in flat_statuses]
    initial_distances = pairwise_distances(starts[None, :, :])[0]
    reference = _goal_reference(starts, goals, prediction_horizon)
    reference_distances = pairwise_distances(reference)
    record: dict[str, object] = dict(metrics)
    record["collision_mode"] = collision_mode
    record["collision_soft_weight"] = collision_soft_weight if collision_mode == "soft_penalty" else None
    record["initial_min_pairwise_distance"] = float(np.min(initial_distances))
    record["straight_line_reference_min_pairwise_distance"] = float(np.min(reference_distances))
    record["final_goal_error_by_agent"] = final_goal_error.tolist()
    record["success_by_agent"] = (final_goal_error <= 0.1).tolist()
    record["solver_feasibility_rate"] = float(np.mean(feasible)) if feasible else 0.0
    record["solver_status_counts_by_agent"] = _status_counts_by_agent(solver_statuses)
    return record


def _solver_diagnostics(
    solver_statuses: list[list[str]],
    collision_mode: str,
    collision_soft_weight: float,
) -> dict[str, object]:
    return {
        "collision_mode": collision_mode,
        "collision_soft_weight": collision_soft_weight if collision_mode == "soft_penalty" else None,
        "status_counts_by_agent": _status_counts_by_agent(solver_statuses),
        "steps": [
            {"step": step, "agent_statuses": {str(agent): status for agent, status in enumerate(statuses)}}
            for step, statuses in enumerate(solver_statuses)
        ],
    }


def _status_counts_by_agent(solver_statuses: list[list[str]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {str(agent): {} for agent in range(3)}
    for statuses in solver_statuses:
        for agent, status in enumerate(statuses):
            agent_counts = counts[str(agent)]
            agent_counts[status] = agent_counts.get(status, 0) + 1
    return counts


def _save_states_csv(path: Path, states: np.ndarray, dt: float) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("step", "time", "agent", "x", "y"))
        for step in range(states.shape[0]):
            for agent_index in range(3):
                writer.writerow(
                    (step, step * dt, agent_index, states[step, agent_index, 0], states[step, agent_index, 1])
                )


def _save_controls_csv(path: Path, controls: np.ndarray, dt: float) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("step", "time", "agent", "ux", "uy"))
        for step in range(controls.shape[0]):
            for agent_index in range(3):
                writer.writerow(
                    (step, step * dt, agent_index, controls[step, agent_index, 0], controls[step, agent_index, 1])
                )


if __name__ == "__main__":
    main()
