"""Compare dead reckoning, known anchors, and range-aided harbor SLAM."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.harbor import (
    DEFAULT_HARBOR_CONFIG,
    HarborObservationNoiseConfig,
    HarborRangeLocalization,
    load_harbor_config,
    load_harbor_disturbance_config,
    load_range_aided_slam_config,
    run_harbor_simulation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_HARBOR_CONFIG))
    parser.add_argument("--artifact-dir", default="results/latest/harbor")
    return parser.parse_args()


def _position_estimates(result, agents) -> dict[str, np.ndarray]:
    return {
        agent.name: np.asarray(
            [agent.model.position(state) for state in result.observed_states[agent.name]]
        )
        for agent in agents
    }


def main() -> None:
    """Run matched localization modes and save one compact diagnostic artifact."""
    args = parse_args()
    agents, simulation, communication = load_harbor_config(args.config)
    base = load_range_aided_slam_config(args.config)
    definitions = (
        ("Dead reckoning", replace(base, enabled=False, mode="known_anchor_ekf")),
        ("Known-anchor ranges", replace(base, enabled=True, mode="known_anchor_ekf")),
        ("Joint landmark SLAM", replace(base, enabled=True, mode="joint_landmark_ekf")),
        (
            "SLAM with 60% range loss",
            replace(
                base,
                enabled=True,
                mode="joint_landmark_ekf",
                dropout_probability=0.60,
            ),
        ),
    )
    records = []
    histories = {}
    for label, localization_config in definitions:
        localization = HarborRangeLocalization(localization_config)
        result = run_harbor_simulation(
            agents,
            simulation,
            communication,
            disturbance=load_harbor_disturbance_config(args.config),
            observation_noise=HarborObservationNoiseConfig(enabled=True, seed=1901),
            localization_provider=localization,
        )
        estimates = _position_estimates(result, agents)
        error_histories = {
            agent.name: np.linalg.norm(
                estimates[agent.name] - result.positions[agent.name], axis=1
            )
            for agent in agents
        }
        histories[label] = error_histories
        records.append(
            {
                "mode": label,
                "all_goals_reached": result.all_goals_reached,
                "pairwise_violation_count": result.pairwise_violation_count,
                "position_rmse": {
                    agent.name: float(
                        np.sqrt(np.mean(error_histories[agent.name] ** 2))
                    )
                    for agent in agents
                },
                "final_observability": {
                    name: {
                        "rank": reports[-1].rank,
                        "state_dimension": reports[-1].state_dimension,
                        "smallest_singular_value": reports[-1].smallest_singular_value,
                        "condition_number": reports[-1].condition_number,
                        "observable": reports[-1].observable,
                    }
                    for name, reports in localization.reports.items()
                },
                "observable_fraction": {
                    name: float(np.mean([report.observable for report in reports]))
                    for name, reports in localization.reports.items()
                },
            }
        )
        print(
            f"{label}: complete={result.all_goals_reached}, "
            f"ROV RMSE={records[-1]['position_rmse']['underwater_rov']:.3f} m",
            flush=True,
        )

    output = Path(args.artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "range_aided_slam_development.json"
    plot_path = output / "range_aided_slam_development.png"
    data_path.write_text(json.dumps({"trials": records}, indent=2) + "\n", encoding="utf-8")

    names = [agent.name for agent in agents]
    short_names = ["UGV 1", "UGV 2", "Heron", "BlueROV2"]
    x = np.arange(len(names))
    width = 0.19
    colors = ("#667085", "#1976b9", "#2a9d8f", "#c8553d")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    for index, (record, color) in enumerate(zip(records, colors)):
        offset = (index - 1.5) * width
        axes[0, 0].bar(
            x + offset,
            [record["position_rmse"][name] for name in names],
            width,
            color=color,
            label=record["mode"],
        )
        axes[1, 0].bar(
            x + offset,
            [record["observable_fraction"][name] for name in names],
            width,
            color=color,
        )
    for label, color in zip(histories, colors):
        axes[0, 1].plot(
            histories[label]["underwater_rov"], color=color, label=label, linewidth=1.8
        )
        axes[1, 1].plot(
            histories[label]["surface_vessel"], color=color, label=label, linewidth=1.8
        )
    axes[0, 0].set_title("Position Estimate RMSE")
    axes[0, 0].set_ylabel("RMSE [m]")
    axes[1, 0].set_title("Full Observability Fraction")
    axes[1, 0].set_ylabel("fraction of simulation")
    axes[1, 0].set_ylim(0.0, 1.05)
    for axis in (axes[0, 0], axes[1, 0]):
        axis.set_xticks(x, short_names)
    axes[0, 1].set_title("BlueROV2 Position Error")
    axes[1, 1].set_title("Heron Position Error")
    for axis in (axes[0, 1], axes[1, 1]):
        axis.set_xlabel("simulation step")
        axis.set_ylabel("error [m]")
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle(
        "Range-Aided Harbor Localization and SLAM Development\n"
        "matched dynamics, odometry drift, range noise, and dropout",
        fontsize=15,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved range-SLAM data: {data_path}")
    print(f"Saved range-SLAM plot: {plot_path}")


if __name__ == "__main__":
    main()
