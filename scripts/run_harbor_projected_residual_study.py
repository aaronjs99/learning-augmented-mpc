"""Test controllability-projected current injection for underactuated USVs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from scripts.harbor import (
    DEFAULT_HARBOR_CONFIG,
    HarborProjectedResidualCriteriaConfig,
    load_harbor_config,
    load_harbor_fault_study_config,
    load_harbor_observation_noise_config,
    load_harbor_projected_residual_criteria_config,
    load_harbor_temporary_fault_ensemble_config,
    load_harbor_time_varying_fault_config,
)
from scripts.harbor.experiments import run_temporary_fault_generalization
from scripts.harbor.mpc import load_harbor_mpc_config
from scripts.harbor.plotting import save_projected_residual_plot
from scripts.run_harbor_time_varying_fault_study import (
    build_time_varying_fault_records,
    compact_time_varying_fault_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_HARBOR_CONFIG))
    parser.add_argument("--artifact-dir", default="results/latest/harbor")
    parser.add_argument(
        "--confirmation",
        action="store_true",
        help="use the untouched projected-residual confirmation ensemble",
    )
    return parser.parse_args()


def summarize_projected_residual(records: list[dict]) -> dict:
    """Build paired task, solver, and estimator projection metrics."""
    baseline_label = "Transient-offset threshold RLS"
    candidate_label = "Projected transient-offset RLS"
    seeds = sorted({record["seed"] for record in records})
    by_key = {(record["seed"], record["controller"]): record for record in records}
    baseline = [by_key[(seed, baseline_label)] for seed in seeds]
    candidate = [by_key[(seed, candidate_label)] for seed in seeds]
    baseline_complete = np.asarray(
        [record["all_goals_reached"] for record in baseline], dtype=bool
    )
    candidate_complete = np.asarray(
        [record["all_goals_reached"] for record in candidate], dtype=bool
    )

    def mean_delta(metric: str) -> float:
        return float(
            np.mean(
                [
                    candidate_record[metric] - baseline_record[metric]
                    for baseline_record, candidate_record in zip(
                        baseline, candidate, strict=True
                    )
                ]
            )
        )

    return {
        "trials": len(seeds),
        "completion_rescues": int(
            np.count_nonzero(candidate_complete & ~baseline_complete)
        ),
        "completion_regressions": int(
            np.count_nonzero(~candidate_complete & baseline_complete)
        ),
        "baseline_completion_rate": float(np.mean(baseline_complete)),
        "candidate_completion_rate": float(np.mean(candidate_complete)),
        "candidate_safety_rate": float(
            np.mean(
                [
                    record["pairwise_violation_count"] == 0
                    and record["max_collision_slack"] <= 1.0e-9
                    for record in candidate
                ]
            )
        ),
        "candidate_fallback_free_rate": float(
            np.mean([record["solver_fallbacks"] == 0 for record in candidate])
        ),
        "mean_completion_cost_delta": mean_delta("sustained_completion_cost"),
        "mean_recovery_rmse_delta": mean_delta("recovery_rmse"),
        "mean_current_rmse_delta": mean_delta("current_rmse"),
        "mean_control_current_rmse_delta": mean_delta("control_current_rmse"),
        "mean_final_control_current_rmse_delta": mean_delta(
            "final_control_current_rmse"
        ),
        "mean_usv_final_yaw_error_delta": float(
            np.mean(
                [
                    candidate_record["final_orientation_errors"]["surface_vessel"]
                    - baseline_record["final_orientation_errors"]["surface_vessel"]
                    for baseline_record, candidate_record in zip(
                        baseline, candidate, strict=True
                    )
                ]
            )
        ),
    }


def evaluate_projected_residual(
    summary: dict, criteria: HarborProjectedResidualCriteriaConfig
) -> dict:
    """Apply frozen closed-loop and estimator non-inferiority gates."""
    trials = summary["trials"]
    rescue_rate = summary["completion_rescues"] / trials
    regression_rate = summary["completion_regressions"] / trials
    checks = {
        "candidate_completion_rate": (
            summary["candidate_completion_rate"]
            >= criteria.minimum_candidate_completion_rate
        ),
        "completion_rescue_rate": (
            rescue_rate >= criteria.minimum_completion_rescue_rate
        ),
        "completion_regression_rate": (
            regression_rate <= criteria.maximum_completion_regression_rate
        ),
        "safety_rate": (
            summary["candidate_safety_rate"] >= criteria.minimum_safety_rate
        ),
        "fallback_free_rate": (
            summary["candidate_fallback_free_rate"]
            >= criteria.minimum_fallback_free_rate
        ),
        "completion_cost_delta": (
            summary["mean_completion_cost_delta"]
            <= criteria.maximum_mean_completion_cost_delta
        ),
        "recovery_rmse_delta": (
            summary["mean_recovery_rmse_delta"]
            <= criteria.maximum_mean_recovery_rmse_delta
        ),
        "current_rmse_delta": (
            summary["mean_current_rmse_delta"]
            <= criteria.maximum_mean_current_rmse_delta
        ),
        "control_current_rmse_delta": (
            summary["mean_control_current_rmse_delta"]
            <= criteria.maximum_mean_control_current_rmse_delta
        ),
        "final_control_current_rmse_delta": (
            summary["mean_final_control_current_rmse_delta"]
            <= criteria.maximum_mean_final_control_current_rmse_delta
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "completion_rescue_rate": rescue_rate,
            "completion_regression_rate": regression_rate,
            **{
                key: value
                for key, value in summary.items()
                if key != "trials"
            },
        },
    }


def main() -> None:
    """Run matched full-versus-projected residual injection."""
    args = parse_args()
    agents, simulation, communication = load_harbor_config(args.config)
    base_disturbance, experiment = load_harbor_time_varying_fault_config(
        args.config
    )
    section = (
        "projected_residual_confirmation"
        if args.confirmation
        else "projected_residual_development"
    )
    ensemble = load_harbor_temporary_fault_ensemble_config(
        args.config, section=section
    )
    criteria = (
        load_harbor_projected_residual_criteria_config(args.config)
        if args.confirmation
        else None
    )
    labels = (
        criteria.controller_labels
        if criteria is not None
        else (
            "Transient-offset threshold RLS",
            "Projected transient-offset RLS",
        )
    )
    mpc_config = load_harbor_mpc_config(args.config)
    cases = run_temporary_fault_generalization(
        agents,
        simulation,
        communication,
        mpc_config,
        base_disturbance,
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
    summary = summarize_projected_residual(records)
    confirmation = (
        evaluate_projected_residual(summary, criteria)
        if criteria is not None
        else None
    )
    payload = {
        "ensemble": asdict(ensemble),
        "experiment": {
            "evaluation_role": section,
            "residual_adaptation_kinds": ["usv", "rov"],
            "baseline_residual_projection": "full",
            "candidate_residual_projection": "actuation_subspace",
            "controller_blind_to_hidden_fault_and_current": True,
            "candidate_status": (
                "projected residual confirmed in simulation"
                if confirmation is not None and confirmation["passed"]
                else (
                    "projected residual confirmation criteria not met"
                    if confirmation is not None
                    else "projected residual development"
                )
            ),
        },
        "projected_residual_summary": summary,
        "trials": compact_time_varying_fault_records(records),
    }
    if criteria is not None:
        payload["projected_residual_confirmation_criteria"] = asdict(criteria)
        payload["projected_residual_confirmation_result"] = confirmation
    output = Path(args.artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / f"{section}.json"
    figure_path = output / f"{section}.png"
    data_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    save_projected_residual_plot(
        records,
        summary,
        figure_path,
        study_title=(
            "Independent Actuation-Subspace Confirmation"
            if args.confirmation
            else "Actuation-Subspace Residual Development"
        ),
        confirmation_passed=(
            confirmation["passed"] if confirmation is not None else None
        ),
    )
    print(json.dumps(summary, indent=2))
    if confirmation is not None:
        print(json.dumps(confirmation, indent=2))
    print(f"Saved projected-residual data: {data_path}")
    print(f"Saved projected-residual plot: {figure_path}")


if __name__ == "__main__":
    main()
