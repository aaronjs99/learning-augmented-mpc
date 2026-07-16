"""Sweep harbor communication delay and dropout with seeded trials."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.harbor import DEFAULT_HARBOR_CONFIG, load_harbor_config
from scripts.harbor.experiments import sweep_network_robustness
from scripts.harbor.plotting import save_network_robustness_heatmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_HARBOR_CONFIG))
    parser.add_argument("--delays", default="0,1,2,4,6,8")
    parser.add_argument("--dropouts", default="0,0.1,0.2,0.3,0.5")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--output-dir", default="results/tmp/harbor_network_sweep")
    return parser.parse_args()


def main() -> None:
    """Run the sweep and save compact data plus a visual robustness envelope."""
    args = parse_args()
    agents, simulation, communication = load_harbor_config(args.config)
    records = sweep_network_robustness(
        agents,
        simulation,
        communication,
        delays=_parse_values(args.delays, int),
        dropout_probabilities=_parse_values(args.dropouts, float),
        seeds=_parse_values(args.seeds, int),
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "network_sweep.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "network_sweep.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    heatmap = save_network_robustness_heatmap(
        records, output / "network_robustness.png"
    )
    print(json.dumps(records, indent=2))
    print(f"Saved network robustness heatmap: {heatmap}")


def _parse_values(text: str, value_type):
    values = [value_type(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("sweep value lists must not be empty")
    return values


if __name__ == "__main__":
    main()
