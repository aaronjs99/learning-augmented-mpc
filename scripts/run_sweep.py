"""Run compact APF/LMPC benchmark sweeps without saving plot artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.config import (
    DEFAULT_CONFIG_PATH,
    list_config_scenarios,
    load_project_config,
    override_project_config,
)
from scripts.learning import run_manta_lmpc, summarize_optimizer_slack
from scripts.metrics import cost_by_iteration


def parse_args() -> argparse.Namespace:
    """Parse sweep options."""
    parser = argparse.ArgumentParser(description="Run manta scenario benchmarks.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="scenario to run; repeat for multiple scenarios",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="LMPC iterations per scenario; default 0 is APF-only",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--apf-max-steps", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="directory for benchmark_summary.csv/json",
    )
    return parser.parse_args()


def main() -> None:
    """Run the selected scenarios and save a compact benchmark table."""
    args = parse_args()
    scenarios = args.scenario or list_config_scenarios(args.config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"results/tmp/sweep_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for scenario_name in scenarios:
        config = load_project_config(args.config, scenario_name=scenario_name)
        config = override_project_config(
            config,
            lmpc={"iterations": args.iterations, "max_steps": args.max_steps},
            apf={"max_steps": args.apf_max_steps},
        )
        lmpc = config.lmpc
        apf = config.apf
        if not args.quiet:
            print(
                f"Running {scenario_name}: "
                f"iterations={lmpc.iterations}, max_steps={lmpc.max_steps}"
            )
        started_at = perf_counter()
        result = run_manta_lmpc(
            config.scenario,
            config=lmpc,
            apf_config=apf,
            dynamics_config=config.dynamics,
            verbose=False,
        )
        elapsed_seconds = perf_counter() - started_at
        records.append(
            _record_for_result(
                config.scenario.goals,
                lmpc,
                result,
                elapsed_seconds=elapsed_seconds,
            )
        )

    _write_csv(output_dir / "benchmark_summary.csv", records)
    with (output_dir / "benchmark_summary.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    _print_table(records)
    print(f"Saved benchmark sweep to: {output_dir}")


def _record_for_result(
    goals,
    lmpc,
    result,
    *,
    elapsed_seconds: float,
) -> dict[str, object]:
    selected = result.selected_iteration
    selected_validation = (
        result.validation_by_iteration[selected] if selected is not None else None
    )
    latest_validation = result.validation_by_iteration[-1]
    costs = cost_by_iteration(
        result.histories,
        goals,
        goal_tolerance=lmpc.goal_tolerance,
    )
    selected_costs = {
        str(agent): values[selected] if selected is not None else values[-1]
        for agent, values in costs.items()
    }
    selected_slack = (
        summarize_optimizer_slack(result.slack_by_iteration[selected - 1])
        if selected is not None and selected > 0
        else summarize_optimizer_slack(np.zeros((0, len(goals), 3)))
    )
    latest_slack = (
        summarize_optimizer_slack(result.slack_by_iteration[-1])
        if result.slack_by_iteration
        else summarize_optimizer_slack(np.zeros((0, len(goals), 3)))
    )

    return {
        "scenario": result.scenario_name,
        "iterations": lmpc.iterations,
        "max_steps": lmpc.max_steps,
        "elapsed_seconds": elapsed_seconds,
        "selected_iteration": selected,
        "selected_valid": selected_validation.valid if selected_validation else False,
        "selected_safe": selected_validation.safe if selected_validation else False,
        "selected_solver_clean": (
            selected_validation.solver_clean if selected_validation else False
        ),
        "selected_min_pairwise": (
            selected_validation.min_pairwise_distance if selected_validation else None
        ),
        "selected_min_obstacle_clearance": (
            selected_validation.min_obstacle_clearance
            if selected_validation
            else None
        ),
        "selected_fallback_count": (
            selected_validation.fallback_count if selected_validation else None
        ),
        "selected_safety_interventions": (
            selected_validation.safety_intervention_count
            if selected_validation
            else None
        ),
        "selected_max_static_slack": selected_slack["max_static_slack"],
        "selected_max_hyperplane_slack": selected_slack["max_hyperplane_slack"],
        "selected_max_terminal_slack": selected_slack["max_terminal_slack"],
        "selected_nonzero_static_slack_steps": selected_slack[
            "nonzero_static_slack_steps"
        ],
        "selected_nonzero_hyperplane_slack_steps": selected_slack[
            "nonzero_hyperplane_slack_steps"
        ],
        "selected_nonzero_terminal_slack_steps": selected_slack[
            "nonzero_terminal_slack_steps"
        ],
        "latest_max_static_slack": latest_slack["max_static_slack"],
        "latest_max_hyperplane_slack": latest_slack["max_hyperplane_slack"],
        "latest_max_terminal_slack": latest_slack["max_terminal_slack"],
        "latest_nonzero_static_slack_steps": latest_slack[
            "nonzero_static_slack_steps"
        ],
        "latest_nonzero_hyperplane_slack_steps": latest_slack[
            "nonzero_hyperplane_slack_steps"
        ],
        "latest_nonzero_terminal_slack_steps": latest_slack[
            "nonzero_terminal_slack_steps"
        ],
        "latest_valid": latest_validation.valid,
        "latest_safe": latest_validation.safe,
        "latest_solver_clean": latest_validation.solver_clean,
        "latest_fallback_count": latest_validation.fallback_count,
        "latest_safety_interventions": latest_validation.safety_intervention_count,
        "cost_by_iteration": {str(agent): values for agent, values in costs.items()},
        "selected_costs": selected_costs,
    }


def _write_csv(path: Path, records: list[dict[str, object]]) -> None:
    fields = [
        "scenario",
        "iterations",
        "max_steps",
        "elapsed_seconds",
        "selected_iteration",
        "selected_valid",
        "selected_safe",
        "selected_solver_clean",
        "selected_min_pairwise",
        "selected_min_obstacle_clearance",
        "selected_fallback_count",
        "selected_safety_interventions",
        "selected_max_static_slack",
        "selected_max_hyperplane_slack",
        "selected_max_terminal_slack",
        "selected_nonzero_static_slack_steps",
        "selected_nonzero_hyperplane_slack_steps",
        "selected_nonzero_terminal_slack_steps",
        "latest_max_static_slack",
        "latest_max_hyperplane_slack",
        "latest_max_terminal_slack",
        "latest_nonzero_static_slack_steps",
        "latest_nonzero_hyperplane_slack_steps",
        "latest_nonzero_terminal_slack_steps",
        "latest_valid",
        "latest_safe",
        "latest_solver_clean",
        "latest_fallback_count",
        "latest_safety_interventions",
        "selected_costs",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            row = {field: record[field] for field in fields}
            row["selected_costs"] = json.dumps(row["selected_costs"])
            writer.writerow(row)


def _print_table(records: list[dict[str, object]]) -> None:
    print(
        "scenario, seconds, selected, valid, safe, selected_clean, latest_clean, "
        "safety_interventions, "
        "selected_static_slack, latest_static_slack, selected_terminal_slack, "
        "latest_terminal_slack, min_pairwise, "
        "min_obs_clearance, selected_costs"
    )
    for record in records:
        print(
            f"{record['scenario']}, {_fmt(record['elapsed_seconds'])}, "
            f"{record['selected_iteration']}, "
            f"{record['selected_valid']}, {record['selected_safe']}, "
            f"{record['selected_solver_clean']}, "
            f"{record['latest_solver_clean']}, "
            f"{record['selected_safety_interventions']}, "
            f"{_fmt(record['selected_max_static_slack'])}, "
            f"{_fmt(record['latest_max_static_slack'])}, "
            f"{_fmt(record['selected_max_terminal_slack'])}, "
            f"{_fmt(record['latest_max_terminal_slack'])}, "
            f"{_fmt(record['selected_min_pairwise'])}, "
            f"{_fmt(record['selected_min_obstacle_clearance'])}, "
            f"{record['selected_costs']}"
        )


def _fmt(value: object) -> str:
    return "None" if value is None else f"{float(value):.3f}"


if __name__ == "__main__":
    main()
