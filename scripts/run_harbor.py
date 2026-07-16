"""Run communication ablations for the heterogeneous harbor scenario."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from scripts.harbor import (
    DEFAULT_HARBOR_CONFIG,
    load_harbor_config,
    run_harbor_simulation,
)
from scripts.harbor.plotting import save_harbor_animation, save_harbor_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_HARBOR_CONFIG))
    parser.add_argument(
        "--mode", choices=("compare", "communication", "independent"), default="compare"
    )
    parser.add_argument(
        "--policy",
        choices=("reciprocal", "eta_priority"),
        default=None,
        help="override the communication coordination policy",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="optional JSON path; no artifact is written when omitted",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="optional directory for a comparison PNG and coordinated GIF",
    )
    return parser.parse_args()


def main() -> None:
    """Run one or both communication modes and print compact metrics."""
    args = parse_args()
    agents, simulation, communication = load_harbor_config(args.config)
    if args.policy is not None:
        simulation = replace(simulation, coordination_policy=args.policy)
    modes = (
        ("independent", "communication") if args.mode == "compare" else (args.mode,)
    )
    records = []
    results = {}
    for mode in modes:
        link = replace(communication, enabled=mode == "communication")
        result = run_harbor_simulation(agents, simulation, link)
        label = mode
        if mode == "communication":
            label = simulation.coordination_policy
        results[label] = result
        records.append(
            {
                "mode": mode,
                "coordination_policy": (
                    simulation.coordination_policy if mode == "communication" else None
                ),
                "all_goals_reached": result.all_goals_reached,
                "first_goal_steps": result.first_goal_steps,
                "completion_step_sum": sum(
                    step if step is not None else simulation.horizon + 1
                    for step in result.first_goal_steps.values()
                ),
                "makespan": max(
                    step if step is not None else simulation.horizon + 1
                    for step in result.first_goal_steps.values()
                ),
                "final_goal_errors": result.final_goal_errors,
                "min_pairwise_distance": result.min_pairwise_distance,
                "pairwise_violation_count": result.pairwise_violation_count,
                "messages_sent": result.messages_sent,
                "messages_delivered": result.messages_delivered,
                "messages_dropped": result.messages_dropped,
                "guidance_update_count": result.guidance_update_count,
            }
        )
    text = json.dumps(records, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
        comparison = save_harbor_comparison(
            results, agents, simulation, plot_dir / "harbor_comparison.png"
        )
        coordinated = next(
            (result for name, result in results.items() if name != "independent"),
            None,
        )
        print(f"Saved comparison plot: {comparison}")
        if coordinated is not None:
            animation = save_harbor_animation(
                coordinated,
                agents,
                simulation,
                plot_dir / "harbor_coordination.gif",
            )
            print(f"Saved coordination animation: {animation}")


if __name__ == "__main__":
    main()
