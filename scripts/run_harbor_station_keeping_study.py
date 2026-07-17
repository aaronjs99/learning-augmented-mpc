"""Evaluate actuator-independent marine-current observation for station keeping."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from scripts.harbor import (
    DEFAULT_HARBOR_CONFIG,
    HarborStationKeepingCriteriaConfig,
    load_harbor_config,
    load_harbor_fault_study_config,
    load_harbor_observation_noise_config,
    load_harbor_station_keeping_criteria_config,
    load_harbor_temporary_fault_ensemble_config,
    load_harbor_time_varying_fault_config,
)
from scripts.harbor.experiments import run_temporary_fault_generalization
from scripts.harbor.mpc import load_harbor_mpc_config
from scripts.harbor.plotting import save_station_keeping_plot
from scripts.run_harbor_time_varying_fault_study import (
    build_time_varying_fault_records,
    compact_time_varying_fault_records,
)


BASELINE = "Hard-envelope transient-offset RLS"
KINEMATIC_EWMA = "Kinematic-current EWMA transient-offset RLS"
KINEMATIC_RLS = "Kinematic-current RLS transient-offset RLS"
USV_NAME = "surface_vessel"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_HARBOR_CONFIG))
    parser.add_argument("--artifact-dir", default="results/latest/harbor")
    parser.add_argument("--confirmation", action="store_true")
    return parser.parse_args()


def summarize_station_keeping(records: list[dict]) -> dict:
    """Aggregate paired station-keeping and observer metrics."""
    labels = tuple(dict.fromkeys(record["controller"] for record in records))
    seeds = sorted({record["seed"] for record in records})
    by_key = {(record["seed"], record["controller"]): record for record in records}
    controllers = {}
    for label in labels:
        selected = [by_key[(seed, label)] for seed in seeds]
        controllers[label] = {
            "completion_rate": float(
                np.mean([record["all_goals_reached"] for record in selected])
            ),
            "safety_rate": float(
                np.mean(
                    [
                        record["pairwise_violation_count"] == 0
                        and record["max_collision_slack"] <= 1.0e-9
                        for record in selected
                    ]
                )
            ),
            "fallback_free_rate": float(
                np.mean([record["solver_fallbacks"] == 0 for record in selected])
            ),
            "mean_usv_yaw_error": float(
                np.mean(
                    [record["final_orientation_errors"][USV_NAME] for record in selected]
                )
            ),
            "mean_usv_position_error": float(
                np.mean([record["final_goal_errors"][USV_NAME] for record in selected])
            ),
            "mean_usv_current_rmse": float(
                np.mean(
                    [record["platform_current_rmse"][USV_NAME] for record in selected]
                )
            ),
            "mean_usv_control_total_variation": float(
                np.mean(
                    [record["control_total_variation"][USV_NAME] for record in selected]
                )
            ),
        }
    baseline = [by_key[(seed, BASELINE)] for seed in seeds]
    candidate = [by_key[(seed, KINEMATIC_RLS)] for seed in seeds]

    def paired_delta(field: str, nested: str | None = None) -> np.ndarray:
        def value(record):
            item = record[field]
            return item[nested] if nested is not None else item

        return np.asarray([value(c) - value(b) for b, c in zip(baseline, candidate)])

    yaw_delta = paired_delta("final_orientation_errors", USV_NAME)
    return {
        "trials": len(seeds),
        "controllers": controllers,
        "usv_yaw_wins": int(np.count_nonzero(yaw_delta < 0.0)),
        "usv_yaw_win_rate": float(np.mean(yaw_delta < 0.0)),
        "mean_usv_yaw_error_delta": float(np.mean(yaw_delta)),
        "mean_usv_position_error_delta": float(
            np.mean(paired_delta("final_goal_errors", USV_NAME))
        ),
        "mean_usv_current_rmse_delta": float(
            np.mean(paired_delta("platform_current_rmse", USV_NAME))
        ),
        "mean_recovery_rmse_delta": float(
            np.mean(paired_delta("recovery_rmse"))
        ),
        "mean_completion_cost_delta": float(
            np.mean(paired_delta("sustained_completion_cost"))
        ),
        "mean_usv_control_total_variation_delta": float(
            np.mean(paired_delta("control_total_variation", USV_NAME))
        ),
    }


def evaluate_station_keeping(
    summary: dict, criteria: HarborStationKeepingCriteriaConfig
) -> dict:
    """Apply predeclared station-keeping confirmation gates."""
    candidate = summary["controllers"][criteria.controller_labels[1]]
    checks = {
        "candidate_completion_rate": candidate["completion_rate"]
        >= criteria.minimum_candidate_completion_rate,
        "usv_yaw_win_rate": summary["usv_yaw_win_rate"]
        >= criteria.minimum_usv_yaw_win_rate,
        "safety_rate": candidate["safety_rate"] >= criteria.minimum_safety_rate,
        "fallback_free_rate": candidate["fallback_free_rate"]
        >= criteria.minimum_fallback_free_rate,
        "usv_yaw_error_delta": summary["mean_usv_yaw_error_delta"]
        <= criteria.maximum_mean_usv_yaw_error_delta,
        "usv_position_error_delta": summary["mean_usv_position_error_delta"]
        <= criteria.maximum_mean_usv_position_error_delta,
        "usv_current_rmse_delta": summary["mean_usv_current_rmse_delta"]
        <= criteria.maximum_mean_usv_current_rmse_delta,
        "recovery_rmse_delta": summary["mean_recovery_rmse_delta"]
        <= criteria.maximum_mean_recovery_rmse_delta,
        "completion_cost_delta": summary["mean_completion_cost_delta"]
        <= criteria.maximum_mean_completion_cost_delta,
    }
    return {"passed": all(checks.values()), "checks": checks}


def main() -> None:
    """Run development or untouched confirmation cases and save evidence."""
    args = parse_args()
    agents, simulation, communication = load_harbor_config(args.config)
    disturbance, experiment = load_harbor_time_varying_fault_config(args.config)
    section = (
        "station_keeping_confirmation"
        if args.confirmation
        else "station_keeping_development"
    )
    ensemble = load_harbor_temporary_fault_ensemble_config(args.config, section)
    criteria = (
        load_harbor_station_keeping_criteria_config(args.config)
        if args.confirmation
        else None
    )
    labels = (
        criteria.controller_labels
        if criteria is not None
        else (BASELINE, KINEMATIC_EWMA, KINEMATIC_RLS)
    )
    mpc_config = load_harbor_mpc_config(args.config)
    cases = run_temporary_fault_generalization(
        agents,
        simulation,
        communication,
        mpc_config,
        disturbance,
        load_harbor_fault_study_config(args.config),
        experiment,
        ensemble,
        load_harbor_observation_noise_config(args.config),
        controller_labels=labels,
        residual_adaptation=True,
        residual_adaptation_kinds=("usv", "rov"),
    )
    records = build_time_varying_fault_records(
        cases, agents, simulation, experiment.event_detection_window_steps
    )
    summary = summarize_station_keeping(records)
    confirmation = (
        evaluate_station_keeping(summary, criteria) if criteria is not None else None
    )
    payload = {
        "ensemble": asdict(ensemble),
        "experiment": {
            "evaluation_role": section,
            "candidate_status": (
                "confirmed in simulation"
                if confirmation is not None and confirmation["passed"]
                else "confirmation criteria not met"
                if confirmation is not None
                else "development"
            ),
            "actuator_independent_current_measurement": True,
            "plant_rate_observer_updates": True,
        },
        "summary": summary,
        "trials": compact_time_varying_fault_records(records),
    }
    if criteria is not None:
        payload["confirmation_criteria"] = asdict(criteria)
        payload["confirmation_result"] = confirmation
    output = Path(args.artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / f"{section}.json"
    plot_path = output / f"{section}.png"
    data_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    save_station_keeping_plot(records, summary, plot_path, confirmation)
    print(json.dumps(summary, indent=2))
    if confirmation is not None:
        print(json.dumps(confirmation, indent=2))
    print(f"Saved station-keeping data: {data_path}")
    print(f"Saved station-keeping plot: {plot_path}")


if __name__ == "__main__":
    main()
