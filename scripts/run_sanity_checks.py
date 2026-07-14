"""CLI entry point for zero-control manta sanity checks."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.config import (
    DEFAULT_CONFIG_PATH,
    list_config_scenarios,
    load_project_config,
)
from scripts.metrics import compute_rollout_metrics, pairwise_distances
from scripts.plotting import plot_pairwise_distances, plot_trajectories
from scripts.simulation import (
    MantaEnvConfig,
    MultiMantaRayEnv,
    manta_rollout,
)


def parse_args() -> argparse.Namespace:
    """Parse YAML config path and lightweight sanity-check overrides."""
    parser = argparse.ArgumentParser(description="Run manta simulation sanity checks.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--scenario", default="all", help="scenario name or 'all'")
    parser.add_argument("--horizon", type=int, default=None, help="simulation horizon")
    parser.add_argument("--dt", type=float, default=None, help="simulation time step")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="override output directory (default: results/sanity_<timestamp>)",
    )
    return parser.parse_args()


def main() -> None:
    """Execute zero-control manta rollouts and save metrics/plots."""
    args = parse_args()
    names = (
        list_config_scenarios(args.config)
        if args.scenario == "all"
        else [args.scenario]
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = (
        Path(args.output_dir)
        if args.output_dir
        else Path("results") / f"sanity_{timestamp}"
    )
    root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, float | int | None]] = {}

    for name in names:
        project_config = load_project_config(args.config, scenario_name=name)
        scenario = project_config.scenario
        dt = project_config.lmpc.dt if args.dt is None else args.dt
        horizon = (
            project_config.lmpc.max_steps if args.horizon is None else args.horizon
        )
        run_dir = root / name
        run_dir.mkdir(parents=True, exist_ok=True)

        env = MultiMantaRayEnv(
            starts=scenario.starts,
            goals=scenario.goals,
            config=MantaEnvConfig(
                dt=dt,
                horizon=horizon,
                dynamics=project_config.dynamics,
            ),
        )
        states, controls = manta_rollout(env, controls=None)

        dists = pairwise_distances(states)
        metrics = compute_rollout_metrics(
            states=states,
            goals=scenario.goals,
            safety_distance=scenario.safety_distance,
            dt=dt,
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
            dt=dt,
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
