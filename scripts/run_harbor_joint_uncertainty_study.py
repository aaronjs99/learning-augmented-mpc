"""Evaluate joint current and temporary-actuator uncertainty in harbor MPC."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from scripts.harbor import (
    DEFAULT_HARBOR_CONFIG,
    load_harbor_config,
    load_harbor_fault_study_config,
    load_harbor_observation_noise_config,
    HarborJointUncertaintyCriteriaConfig,
    load_harbor_joint_uncertainty_criteria_config,
    load_harbor_temporary_fault_ensemble_config,
    load_harbor_time_varying_fault_config,
)
from scripts.harbor.experiments import run_temporary_fault_generalization
from scripts.harbor.mpc import load_harbor_mpc_config
from scripts.harbor.plotting import save_joint_uncertainty_plot
from scripts.run_harbor_temporary_fault_generalization import (
    summarize_recovery_prior,
)
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
        help="use the untouched joint-uncertainty confirmation ensemble",
    )
    return parser.parse_args()


def summarize_joint_uncertainty(records: list[dict], recovery: dict) -> dict:
    """Summarize paired closed-loop resilience under both hidden effects."""
    baseline_label = "Innovation-threshold RLS"
    candidate_label = "Transient-offset threshold RLS"
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
    return {
        **recovery,
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
    }


def evaluate_joint_uncertainty(
    summary: dict, criteria: HarborJointUncertaintyCriteriaConfig
) -> dict:
    """Apply frozen task and estimator non-inferiority gates."""
    trials = summary["trials"]
    rescue_rate = summary["completion_rescues"] / trials
    regression_rate = summary["completion_regressions"] / trials
    recovery_win_rate = summary["recovery_wins"] / trials
    final_effectiveness_delta = -summary["mean_final_rmse_reduction"]
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
        "recovery_win_rate": (
            recovery_win_rate >= criteria.minimum_recovery_win_rate
        ),
        "completion_cost_delta": (
            summary["mean_completion_cost_delta"]
            <= criteria.maximum_mean_completion_cost_delta
        ),
        "final_effectiveness_rmse_delta": (
            final_effectiveness_delta
            <= criteria.maximum_mean_final_effectiveness_rmse_delta
        ),
        "current_rmse_delta": (
            summary["mean_current_rmse_delta"]
            <= criteria.maximum_mean_current_rmse_delta
        ),
        "final_current_rmse_delta": (
            summary["mean_final_current_rmse_delta"]
            <= criteria.maximum_mean_final_current_rmse_delta
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "completion_rescue_rate": rescue_rate,
            "completion_regression_rate": regression_rate,
            "recovery_win_rate": recovery_win_rate,
            "final_effectiveness_rmse_delta": final_effectiveness_delta,
            **{
                key: summary[key]
                for key in (
                    "baseline_completion_rate",
                    "candidate_completion_rate",
                    "candidate_safety_rate",
                    "candidate_fallback_free_rate",
                    "mean_completion_cost_delta",
                    "mean_current_rmse_delta",
                    "mean_final_current_rmse_delta",
                )
            },
        },
    }


def main() -> None:
    """Run matched joint-adaptive policies and retain auditable evidence."""
    args = parse_args()
    agents, simulation, communication = load_harbor_config(args.config)
    base_disturbance, experiment = load_harbor_time_varying_fault_config(
        args.config
    )
    section = (
        "joint_uncertainty_confirmation"
        if args.confirmation
        else "joint_uncertainty_development"
    )
    artifact_stem = section
    ensemble = load_harbor_temporary_fault_ensemble_config(
        args.config, section=section
    )
    criteria = (
        load_harbor_joint_uncertainty_criteria_config(args.config)
        if args.confirmation
        else None
    )
    controller_labels = (
        criteria.controller_labels
        if criteria is not None
        else (
            "Innovation-threshold RLS",
            "Transient-offset threshold RLS",
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
        controller_labels=controller_labels,
        residual_adaptation=True,
        residual_adaptation_kinds=("usv", "rov"),
    )
    records = build_time_varying_fault_records(
        cases, agents, simulation, experiment.event_detection_window_steps
    )
    recovery_comparison = summarize_recovery_prior(
        records,
        ensemble.bootstrap_samples,
        candidate_label="Transient-offset threshold RLS",
    )
    summary = summarize_joint_uncertainty(records, recovery_comparison)
    confirmation = (
        evaluate_joint_uncertainty(summary, criteria)
        if criteria is not None
        else None
    )
    status = (
        "joint uncertainty confirmed in simulation"
        if confirmation is not None and confirmation["passed"]
        else (
            "joint uncertainty confirmation criteria not met"
            if confirmation is not None
            else "joint uncertainty development"
        )
    )
    payload = {
        "ensemble": asdict(ensemble),
        "experiment": {
            "evaluation_role": section,
            "residual_adaptation": True,
            "residual_adaptation_kinds": ["usv", "rov"],
            "effectiveness_adaptation": True,
            "controller_blind_to_hidden_fault_and_current": True,
            "change_threshold": experiment.change_threshold,
            "covariance_inflation": experiment.covariance_inflation,
            "recovery_prior_gain": experiment.recovery_prior_gain,
            "recovery_offset_decay": (
                mpc_config.effectiveness_recovery_offset_decay
            ),
            "recovery_minimum_dwell_steps": (
                mpc_config.effectiveness_recovery_minimum_dwell_steps
            ),
            "candidate_status": status,
        },
        "joint_uncertainty_summary": summary,
        "trials": compact_time_varying_fault_records(records),
    }
    if criteria is not None:
        payload["joint_uncertainty_confirmation_criteria"] = asdict(criteria)
        payload["joint_uncertainty_confirmation_result"] = confirmation

    output = Path(args.artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / f"{artifact_stem}.json"
    figure_path = output / f"{artifact_stem}.png"
    data_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    save_joint_uncertainty_plot(
        records,
        summary,
        figure_path,
        study_title=(
            "Independent Joint-Uncertainty Confirmation"
            if args.confirmation
            else "Joint Current and Temporary-Fault Development"
        ),
        confirmation_passed=(
            confirmation["passed"] if confirmation is not None else None
        ),
    )
    print(json.dumps(summary, indent=2))
    if confirmation is not None:
        print(json.dumps(confirmation, indent=2))
    print(f"Saved joint-uncertainty data: {data_path}")
    print(f"Saved joint-uncertainty plot: {figure_path}")


if __name__ == "__main__":
    main()
