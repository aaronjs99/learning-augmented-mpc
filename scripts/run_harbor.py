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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_HARBOR_CONFIG))
    parser.add_argument(
        "--mode", choices=("compare", "communication", "independent"), default="compare"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="optional JSON path; no artifact is written when omitted",
    )
    return parser.parse_args()


def main() -> None:
    """Run one or both communication modes and print compact metrics."""
    args = parse_args()
    agents, simulation, communication = load_harbor_config(args.config)
    modes = (
        ("independent", "communication") if args.mode == "compare" else (args.mode,)
    )
    records = []
    for mode in modes:
        link = replace(communication, enabled=mode == "communication")
        result = run_harbor_simulation(agents, simulation, link)
        records.append(
            {
                "mode": mode,
                "all_goals_reached": result.all_goals_reached,
                "first_goal_steps": result.first_goal_steps,
                "final_goal_errors": result.final_goal_errors,
                "min_pairwise_distance": result.min_pairwise_distance,
                "pairwise_violation_count": result.pairwise_violation_count,
                "messages_sent": result.messages_sent,
                "messages_delivered": result.messages_delivered,
                "messages_dropped": result.messages_dropped,
            }
        )
    text = json.dumps(records, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
