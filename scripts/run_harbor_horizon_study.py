"""Benchmark distributed MPC and LMPC across matched prediction horizons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.harbor import DEFAULT_HARBOR_CONFIG, load_harbor_config
from scripts.harbor.experiments import sweep_prediction_horizons
from scripts.harbor.mpc import load_harbor_mpc_config
from scripts.harbor.plotting import save_horizon_efficiency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_HARBOR_CONFIG))
    parser.add_argument("--horizons", default="8,12,15")
    parser.add_argument("--artifact-dir", default="results/latest/harbor")
    return parser.parse_args()


def main() -> None:
    """Run the controlled horizon study and overwrite its two artifacts."""
    args = parse_args()
    horizons = [
        int(value.strip()) for value in args.horizons.split(",") if value.strip()
    ]
    agents, simulation, communication = load_harbor_config(args.config)
    records = sweep_prediction_horizons(
        agents,
        simulation,
        communication,
        load_harbor_mpc_config(args.config),
        horizons=horizons,
    )
    output = Path(args.artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "horizon_study.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    figure = save_horizon_efficiency(records, output / "horizon_efficiency.png")
    print(json.dumps(records, indent=2))
    print(f"Saved horizon-efficiency plot: {figure}")


if __name__ == "__main__":
    main()
