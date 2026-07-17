"""Compare peer-motion prediction models under noisy hidden actuator faults."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.harbor import (
    DEFAULT_HARBOR_CONFIG,
    load_harbor_config,
    load_harbor_fault_config,
    load_harbor_fault_ensemble_config,
    load_harbor_fault_study_config,
    load_harbor_observation_noise_config,
)
from scripts.harbor.experiments import run_obstacle_prediction_generalization
from scripts.harbor.mpc import load_harbor_mpc_config
from scripts.harbor.plotting import save_prediction_ablation_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_HARBOR_CONFIG))
    parser.add_argument("--artifact-dir", default="results/latest/harbor")
    return parser.parse_args()


def summarize_prediction_ablation(records: list[dict]) -> dict:
    """Aggregate matched prediction-mode feasibility and task outcomes."""
    labels = ("Constant-velocity prediction", "Goal-bounded prediction")
    controllers = {}
    for label in labels:
        selected = [record for record in records if record["controller"] == label]
        controllers[label] = {
            "trials": len(selected),
            "total_solver_fallbacks": int(
                sum(record["solver_fallbacks"] for record in selected)
            ),
            "fallback_free_rate": float(
                np.mean([record["solver_fallbacks"] == 0 for record in selected])
            ),
            "completion_rate": float(
                np.mean([record["all_goals_reached"] for record in selected])
            ),
            "safety_rate": float(
                np.mean(
                    [
                        record["pairwise_violation_count"] == 0
                        and record.get("max_collision_slack", 0.0) <= 1e-9
                        for record in selected
                    ]
                )
            ),
            "mean_sustained_completion_cost": float(
                np.mean(
                    [record["sustained_completion_cost"] for record in selected]
                )
            ),
            "mean_effectiveness_rmse": float(
                np.mean([record["effectiveness_rmse"] for record in selected])
            ),
        }
    by_key = {
        (record["seed"], record["controller"]): record for record in records
    }
    seeds = sorted({record["seed"] for record in records})
    legacy = [by_key[(seed, labels[0])] for seed in seeds]
    bounded = [by_key[(seed, labels[1])] for seed in seeds]
    return {
        "controllers": controllers,
        "paired_goal_bounded_vs_constant_velocity": {
            "trials": len(seeds),
            "constant_velocity_fallbacks": int(
                sum(record["solver_fallbacks"] for record in legacy)
            ),
            "goal_bounded_fallbacks": int(
                sum(record["solver_fallbacks"] for record in bounded)
            ),
            "fallback_reduction": int(
                sum(record["solver_fallbacks"] for record in legacy)
                - sum(record["solver_fallbacks"] for record in bounded)
            ),
            "complete_safe_pairs": int(
                sum(
                    left["all_goals_reached"]
                    and right["all_goals_reached"]
                    and left["pairwise_violation_count"] == 0
                    and right["pairwise_violation_count"] == 0
                    and left.get("max_collision_slack", 0.0) <= 1e-9
                    and right.get("max_collision_slack", 0.0) <= 1e-9
                    for left, right in zip(legacy, bounded, strict=True)
                )
            ),
            "mean_completion_cost_delta": float(
                np.mean(
                    [
                        right["sustained_completion_cost"]
                        - left["sustained_completion_cost"]
                        for left, right in zip(legacy, bounded, strict=True)
                    ]
                )
            ),
        },
    }


def main() -> None:
    """Run the matched prediction ablation and replace its two artifacts."""
    args = parse_args()
    agents, simulation, communication = load_harbor_config(args.config)
    ensemble = load_harbor_fault_ensemble_config(args.config)
    cases = run_obstacle_prediction_generalization(
        agents,
        simulation,
        communication,
        load_harbor_mpc_config(args.config),
        load_harbor_fault_config(args.config),
        load_harbor_fault_study_config(args.config),
        ensemble,
        load_harbor_observation_noise_config(args.config),
    )
    records = []
    for case in cases:
        truth = np.concatenate(
            [
                case.disturbance.effectiveness(agent.model, agent.name)
                for agent in agents
            ]
        )
        for trial in case.trials:
            estimate = np.concatenate(
                [trial.final_effectiveness_estimates[agent.name] for agent in agents]
            )
            sustained_cost = sum(
                (
                    trial.result.first_goal_steps[agent.name]
                    if trial.result.first_goal_steps[agent.name] is not None
                    and trial.result.final_goal_errors[agent.name]
                    <= simulation.goal_tolerance
                    and trial.result.final_orientation_errors[agent.name]
                    <= simulation.orientation_tolerance
                    else simulation.horizon + 1
                )
                for agent in agents
            )
            records.append(
                {
                    "seed": case.seed,
                    "observation_seed": case.observation_seed,
                    "controller": trial.label,
                    "all_goals_reached": trial.result.all_goals_reached,
                    "pairwise_violation_count": trial.result.pairwise_violation_count,
                    "min_pairwise_distance": trial.result.min_pairwise_distance,
                    "sustained_completion_cost": sustained_cost,
                    "solver_fallbacks": trial.solver_fallbacks,
                    "solver_fallbacks_by_agent": trial.solver_fallbacks_by_agent,
                    "solver_failure_steps_by_agent": (
                        trial.solver_failure_steps_by_agent
                    ),
                    "solver_failure_status_counts": (
                        trial.solver_failure_status_counts
                    ),
                    "max_collision_slack": trial.max_collision_slack,
                    "effectiveness_rmse": float(
                        np.sqrt(np.mean((estimate - truth) ** 2))
                    ),
                }
            )
    summary = summarize_prediction_ablation(records)
    output = Path(args.artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "prediction_ablation.json"
    figure_path = output / "prediction_ablation.png"
    payload = {
        "ensemble": {
            "seeds": list(ensemble.seeds),
            "comparison": "common safe-memory, hidden fault, and observation seed",
        },
        "summary": summary,
        "trials": records,
    }
    data_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    save_prediction_ablation_plot(records, summary, figure_path)
    print(json.dumps(summary, indent=2))
    print(f"Saved prediction-ablation data: {data_path}")
    print(f"Saved prediction-ablation plot: {figure_path}")


if __name__ == "__main__":
    main()
