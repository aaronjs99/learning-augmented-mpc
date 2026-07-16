"""Benchmark nominal and locally adaptive harbor control under plant mismatch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.harbor import (
    DEFAULT_HARBOR_CONFIG,
    load_harbor_config,
    load_harbor_disturbance_config,
)
from scripts.harbor.experiments import run_model_mismatch_study
from scripts.harbor.mpc import load_harbor_mpc_config
from scripts.harbor.plotting import (
    save_harbor_animation,
    save_model_mismatch_diagnostics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_HARBOR_CONFIG))
    parser.add_argument("--artifact-dir", default="results/latest/harbor")
    parser.add_argument("--no-gif", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the matched robustness study and replace its curated artifacts."""
    args = parse_args()
    agents, simulation, communication = load_harbor_config(args.config)
    disturbance = load_harbor_disturbance_config(args.config)
    trials = run_model_mismatch_study(
        agents,
        simulation,
        communication,
        load_harbor_mpc_config(args.config),
        disturbance,
    )
    records = [
        {
            "controller": trial.label,
            "valid": trial.valid,
            "all_goals_reached": trial.result.all_goals_reached,
            "completion_step_sum": trial.completion_step_sum,
            "first_goal_steps": trial.result.first_goal_steps,
            "final_goal_errors": trial.result.final_goal_errors,
            "final_orientation_errors": trial.result.final_orientation_errors,
            "min_pairwise_distance": trial.result.min_pairwise_distance,
            "pairwise_violation_count": trial.result.pairwise_violation_count,
            "solver_fallbacks": trial.solver_fallbacks,
            "max_collision_slack": trial.max_collision_slack,
            "final_residual_estimates": {
                name: value.tolist()
                for name, value in trial.final_residual_estimates.items()
            },
            "final_effectiveness_estimates": {
                name: value.tolist()
                for name, value in trial.final_effectiveness_estimates.items()
            },
        }
        for trial in trials
    ]
    output = Path(args.artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "model_mismatch_study.json"
    data_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    figure = save_model_mismatch_diagnostics(
        trials,
        agents,
        simulation,
        disturbance,
        output / "model_mismatch_diagnostics.png",
    )
    adaptive = next(trial for trial in trials if trial.label == "Joint-adaptive LMPC")
    if not args.no_gif:
        save_harbor_animation(
            adaptive.result,
            agents,
            simulation,
            output / "robust_harbor_lmpc.gif",
            label="Joint-adaptive distributed LMPC",
        )
    print(json.dumps(records, indent=2))
    print(f"Saved robustness diagnostics: {figure}")


if __name__ == "__main__":
    main()
