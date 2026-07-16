"""Run distributed harbor MPC and safe-set LMPC from a verified guidance seed."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from scripts.harbor import DEFAULT_HARBOR_CONFIG, load_harbor_config
from scripts.harbor.learning import run_distributed_harbor_lmpc
from scripts.harbor.mpc import load_harbor_mpc_config
from scripts.harbor.plotting import (
    save_harbor_animation,
    save_harbor_learning_progress,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_HARBOR_CONFIG))
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--artifact-dir", default="results/latest/harbor")
    parser.add_argument("--no-gif", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the controlled learning experiment and print auditable telemetry."""
    args = parse_args()
    agents, simulation, communication = load_harbor_config(args.config)
    mpc_config = load_harbor_mpc_config(args.config)
    if args.iterations is not None:
        if args.iterations <= 0:
            raise ValueError("--iterations must be positive")
        mpc_config = replace(mpc_config, learning_iterations=args.iterations)
    iterations = run_distributed_harbor_lmpc(
        agents, simulation, communication, mpc_config
    )
    records = [
        {
            "label": item.label,
            "admitted": item.admitted,
            "all_goals_reached": item.result.all_goals_reached,
            "pairwise_violation_count": item.result.pairwise_violation_count,
            "min_pairwise_distance": item.result.min_pairwise_distance,
            "completion_step_sum": item.completion_step_sum,
            "first_goal_steps": item.result.first_goal_steps,
            "solver_calls": item.solver_calls,
            "solver_fallbacks": item.solver_fallbacks,
            "solve_time_seconds": item.solve_time_seconds,
            "max_collision_slack": item.max_collision_slack,
            "max_terminal_slack": item.max_terminal_slack,
            "solve_count_by_agent": item.solve_count_by_agent,
            "fallback_count_by_agent": item.fallback_count_by_agent,
            "failure_steps_by_agent": item.failure_steps_by_agent,
            "failure_status_counts": item.failure_status_counts,
            "final_residual_estimates": item.final_residual_estimates,
            "final_effectiveness_estimates": {
                name: value.tolist()
                for name, value in item.final_effectiveness_estimates.items()
            },
        }
        for item in iterations
    ]
    text = json.dumps(records, indent=2)
    print(text)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "metrics.json").write_text(text + "\n", encoding="utf-8")
    save_harbor_learning_progress(
        iterations,
        agents,
        simulation,
        artifact_dir / "research_progress.png",
    )
    if not args.no_gif:
        best = min(
            (item for item in iterations if item.admitted),
            key=lambda item: item.completion_step_sum,
        )
        animation_label = (
            "Distributed MPC"
            if best.label == "distributed_mpc"
            else best.label.replace("distributed_lmpc_", "Distributed LMPC ")
        )
        save_harbor_animation(
            best.result,
            agents,
            simulation,
            artifact_dir / "harbor_lmpc.gif",
            label=animation_label,
        )


if __name__ == "__main__":
    main()
