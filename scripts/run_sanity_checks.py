"""Run open-loop sanity checks for all 3-agent scenarios.

This script intentionally uses zero control to validate simulation, metrics,
and plotting layers before MPC is implemented.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.metrics import compute_rollout_metrics, pairwise_distances
from src.plotting import plot_pairwise_distances, plot_trajectories
from src.simulation import EnvConfig, ThreeAgentSingleIntegratorEnv, get_scenario, list_scenarios, rollout


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for sanity-check runs."""
    parser = argparse.ArgumentParser(description="Run simulation/metrics sanity checks.")
    parser.add_argument("--scenario", default="all", help="scenario name or 'all'")
    parser.add_argument("--horizon", type=int, default=80, help="simulation horizon")
    parser.add_argument("--dt", type=float, default=0.1, help="simulation time step")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="override output directory (default: results/sanity_<timestamp>)",
    )
    return parser.parse_args()


def main() -> None:
    """Execute zero-control rollouts and save metrics/plots."""
    args = parse_args()
    names = list_scenarios() if args.scenario == "all" else [args.scenario]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.output_dir) if args.output_dir else Path("results") / f"sanity_{timestamp}"
    root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, float | int | None]] = {}

    for name in names:
        scenario = get_scenario(name)
        run_dir = root / name
        run_dir.mkdir(parents=True, exist_ok=True)

        env = ThreeAgentSingleIntegratorEnv(
            starts=scenario.starts,
            goals=scenario.goals,
            config=EnvConfig(dt=args.dt, horizon=args.horizon),
        )
        states, controls = rollout(env, controls=None)

        dists = pairwise_distances(states)
        metrics = compute_rollout_metrics(
            states=states,
            goals=scenario.goals,
            safety_distance=scenario.safety_distance,
            dt=args.dt,
            controls=controls,
        )
        summary[name] = metrics.to_dict()

        plot_trajectories(
            states=states,
            goals=scenario.goals,
            title=f"{name}: trajectories (zero control)",
            out_path=str(run_dir / "trajectories.png"),
        )
        plot_pairwise_distances(
            distances=dists,
            dt=args.dt,
            safety_distance=scenario.safety_distance,
            title=f"{name}: pairwise distances (zero control)",
            out_path=str(run_dir / "pairwise_distances.png"),
        )

        with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2)

    with (root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved sanity-check outputs to: {root}")


if __name__ == "__main__":
    main()
