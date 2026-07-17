"""Evaluate adaptive distributed MPC over hidden temporary actuator faults."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from scripts.harbor import (
    DEFAULT_HARBOR_CONFIG,
    HarborConfirmationCriteriaConfig,
    HarborRecoveryConfirmationCriteriaConfig,
    load_harbor_config,
    load_harbor_confirmation_criteria_config,
    load_harbor_recovery_confirmation_criteria_config,
    load_harbor_fault_study_config,
    load_harbor_observation_noise_config,
    load_harbor_temporary_fault_ensemble_config,
    load_harbor_time_varying_fault_config,
)
from scripts.harbor.experiments import run_temporary_fault_generalization
from scripts.harbor.mpc import load_harbor_mpc_config
from scripts.harbor.plotting import save_temporary_fault_generalization_plot
from scripts.run_harbor_time_varying_fault_study import (
    build_time_varying_fault_records,
    summarize_time_varying_faults,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_HARBOR_CONFIG))
    parser.add_argument("--artifact-dir", default="results/latest/harbor")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--holdout",
        action="store_true",
        help="use the separately seeded holdout ensemble and artifact names",
    )
    mode.add_argument(
        "--confirmation",
        action="store_true",
        help="run the frozen threshold-RLS confirmation ensemble",
    )
    mode.add_argument(
        "--recovery-development",
        action="store_true",
        help="compare the channel-selective recovery prior on new development cases",
    )
    mode.add_argument(
        "--recovery-confirmation",
        action="store_true",
        help="run the frozen recovery-prior confirmation ensemble",
    )
    mode.add_argument(
        "--transient-recovery-development",
        action="store_true",
        help="compare a decaying controller offset with ordinary threshold RLS",
    )
    mode.add_argument(
        "--transient-recovery-confirmation",
        action="store_true",
        help="run the frozen rank-gated transient recovery confirmation",
    )
    return parser.parse_args()


def evaluate_confirmation(
    summary: dict, criteria: HarborConfirmationCriteriaConfig
) -> dict:
    """Apply predeclared closed-loop acceptance gates to threshold RLS."""
    label = "Innovation-threshold RLS"
    controller = summary["controllers"][label]
    comparison = summary["paired_vs_fixed"][label]
    lower_bound = comparison["paired_mean_rmse_reduction_bootstrap_95_ci"][0]
    win_rate = comparison["adaptive_wins"] / comparison["trials"]
    checks = {
        "adaptive_win_rate": win_rate >= criteria.minimum_adaptive_win_rate,
        "positive_bootstrap_lower_bound": (
            lower_bound > 0.0
            if criteria.require_positive_bootstrap_lower_bound
            else True
        ),
        "completion_rate": (
            controller["completion_rate"] >= criteria.minimum_completion_rate
        ),
        "safety_rate": controller["safety_rate"] >= criteria.minimum_safety_rate,
        "fallback_free_rate": (
            controller["fallback_free_rate"]
            >= criteria.minimum_fallback_free_rate
        ),
        "completion_cost_delta": (
            comparison["mean_completion_cost_delta"]
            <= criteria.maximum_mean_completion_cost_delta
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "adaptive_win_rate": win_rate,
            "bootstrap_lower_bound": lower_bound,
            "completion_rate": controller["completion_rate"],
            "safety_rate": controller["safety_rate"],
            "fallback_free_rate": controller["fallback_free_rate"],
            "mean_completion_cost_delta": comparison[
                "mean_completion_cost_delta"
            ],
        },
    }


def summarize_recovery_prior(
    records: list[dict],
    bootstrap_samples: int,
    candidate_label: str = "Recovery-prior threshold RLS",
) -> dict:
    """Compare one recovery-aware threshold RLS with ordinary threshold RLS."""
    baseline_label = "Innovation-threshold RLS"
    seeds = sorted({record["seed"] for record in records})
    by_key = {(record["seed"], record["controller"]): record for record in records}

    def values(label: str, metric: str) -> np.ndarray:
        return np.asarray([by_key[(seed, label)][metric] for seed in seeds], dtype=float)

    baseline_recovery = values(baseline_label, "recovery_rmse")
    candidate_recovery = values(candidate_label, "recovery_rmse")
    recovery_reduction = baseline_recovery - candidate_recovery
    baseline_fault = values(baseline_label, "fault_interval_rmse")
    candidate_fault = values(candidate_label, "fault_interval_rmse")
    baseline_final = values(baseline_label, "final_effectiveness_rmse")
    candidate_final = values(candidate_label, "final_effectiveness_rmse")
    cost_delta = values(candidate_label, "sustained_completion_cost") - values(
        baseline_label, "sustained_completion_cost"
    )
    rng = np.random.default_rng(20260717)
    bootstrap = np.mean(
        rng.choice(
            recovery_reduction,
            size=(bootstrap_samples, len(recovery_reduction)),
            replace=True,
        ),
        axis=1,
    )
    candidate = [by_key[(seed, candidate_label)] for seed in seeds]
    return {
        "trials": len(seeds),
        "recovery_wins": int(np.count_nonzero(recovery_reduction > 0.0)),
        "mean_recovery_rmse_reduction": float(np.mean(recovery_reduction)),
        "mean_relative_recovery_rmse_reduction": float(
            np.mean(recovery_reduction / np.maximum(baseline_recovery, 1.0e-12))
        ),
        "paired_mean_recovery_reduction_bootstrap_95_ci": [
            float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
        ],
        "mean_fault_interval_rmse_delta": float(
            np.mean(candidate_fault - baseline_fault)
        ),
        "mean_final_rmse_reduction": float(
            np.mean(baseline_final - candidate_final)
        ),
        "mean_completion_cost_delta": float(np.mean(cost_delta)),
        "completion_rate": float(
            np.mean([record["all_goals_reached"] for record in candidate])
        ),
        "safety_rate": float(
            np.mean(
                [
                    record["pairwise_violation_count"] == 0
                    and record["max_collision_slack"] <= 1.0e-9
                    for record in candidate
                ]
            )
        ),
        "fallback_free_rate": float(
            np.mean([record["solver_fallbacks"] == 0 for record in candidate])
        ),
    }


def evaluate_recovery_confirmation(
    comparison: dict, criteria: HarborRecoveryConfirmationCriteriaConfig
) -> dict:
    """Apply predeclared closed-loop gates to the recovery-prior comparator."""
    lower_bound = comparison[
        "paired_mean_recovery_reduction_bootstrap_95_ci"
    ][0]
    win_rate = comparison["recovery_wins"] / comparison["trials"]
    final_rmse_delta = -comparison["mean_final_rmse_reduction"]
    checks = {
        "recovery_win_rate": win_rate >= criteria.minimum_recovery_win_rate,
        "positive_bootstrap_lower_bound": (
            lower_bound > 0.0
            if criteria.require_positive_bootstrap_lower_bound
            else True
        ),
        "completion_rate": (
            comparison["completion_rate"] >= criteria.minimum_completion_rate
        ),
        "safety_rate": comparison["safety_rate"] >= criteria.minimum_safety_rate,
        "fallback_free_rate": (
            comparison["fallback_free_rate"]
            >= criteria.minimum_fallback_free_rate
        ),
        "fault_interval_rmse_delta": (
            comparison["mean_fault_interval_rmse_delta"]
            <= criteria.maximum_mean_fault_interval_rmse_delta
        ),
        "final_rmse_delta": (
            final_rmse_delta <= criteria.maximum_mean_final_rmse_delta
        ),
        "completion_cost_delta": (
            comparison["mean_completion_cost_delta"]
            <= criteria.maximum_mean_completion_cost_delta
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "recovery_win_rate": win_rate,
            "bootstrap_lower_bound": lower_bound,
            "completion_rate": comparison["completion_rate"],
            "safety_rate": comparison["safety_rate"],
            "fallback_free_rate": comparison["fallback_free_rate"],
            "mean_fault_interval_rmse_delta": comparison[
                "mean_fault_interval_rmse_delta"
            ],
            "mean_final_rmse_delta": final_rmse_delta,
            "mean_completion_cost_delta": comparison[
                "mean_completion_cost_delta"
            ],
        },
    }


def main() -> None:
    """Run the stratified ensemble and replace its compact artifacts."""
    args = parse_args()
    transient_mode = (
        args.transient_recovery_development
        or args.transient_recovery_confirmation
    )
    recovery_confirmation_mode = (
        args.recovery_confirmation or args.transient_recovery_confirmation
    )
    agents, simulation, communication = load_harbor_config(args.config)
    base_disturbance, experiment = load_harbor_time_varying_fault_config(args.config)
    if transient_mode:
        if args.transient_recovery_confirmation:
            ensemble_section = "temporary_fault_transient_recovery_confirmation"
            evaluation_role = "transient_recovery_confirmation"
            artifact_stem = "temporary_fault_transient_recovery_confirmation"
            recovery_criteria = load_harbor_recovery_confirmation_criteria_config(
                args.config,
                section=(
                    "temporary_fault_transient_recovery_confirmation_criteria"
                ),
            )
            controller_labels = recovery_criteria.controller_labels
        else:
            ensemble_section = "temporary_fault_transient_recovery_development"
            evaluation_role = "transient_recovery_development"
            artifact_stem = "temporary_fault_transient_recovery_development"
            controller_labels = (
                "Fixed-covariance RLS",
                "Innovation-threshold RLS",
                "Transient-offset threshold RLS",
            )
            recovery_criteria = None
        criteria = None
    elif args.recovery_development or args.recovery_confirmation:
        if args.recovery_confirmation:
            ensemble_section = "temporary_fault_recovery_confirmation"
            evaluation_role = "recovery_confirmation"
            artifact_stem = "temporary_fault_recovery_confirmation"
            recovery_criteria = (
                load_harbor_recovery_confirmation_criteria_config(args.config)
            )
        else:
            ensemble_section = "temporary_fault_recovery_development"
            evaluation_role = "recovery_development"
            artifact_stem = "temporary_fault_recovery_development"
            recovery_criteria = None
        controller_labels = (
            "Fixed-covariance RLS",
            "Innovation-threshold RLS",
            "Recovery-prior threshold RLS",
        )
        criteria = None
    elif args.confirmation:
        ensemble_section = "temporary_fault_confirmation"
        evaluation_role = "confirmation"
        artifact_stem = "temporary_fault_confirmation"
        criteria = load_harbor_confirmation_criteria_config(args.config)
        controller_labels = criteria.controller_labels
        recovery_criteria = None
    elif args.holdout:
        ensemble_section = "temporary_fault_holdout"
        evaluation_role = "holdout"
        artifact_stem = "temporary_fault_holdout"
        controller_labels = None
        criteria = None
        recovery_criteria = None
    else:
        ensemble_section = "temporary_fault_ensemble"
        evaluation_role = "development"
        artifact_stem = "temporary_fault_generalization"
        controller_labels = None
        criteria = None
        recovery_criteria = None
    ensemble = load_harbor_temporary_fault_ensemble_config(
        args.config, section=ensemble_section
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
    )
    records = build_time_varying_fault_records(
        cases, agents, simulation, experiment.event_detection_window_steps
    )
    summary = summarize_time_varying_faults(records, ensemble.bootstrap_samples)
    if transient_mode:
        recovery_comparison = summarize_recovery_prior(
            records,
            ensemble.bootstrap_samples,
            candidate_label="Transient-offset threshold RLS",
        )
    elif args.recovery_development or args.recovery_confirmation:
        recovery_comparison = summarize_recovery_prior(
            records, ensemble.bootstrap_samples
        )
    else:
        recovery_comparison = None
    output = Path(args.artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / f"{artifact_stem}.json"
    figure_path = output / f"{artifact_stem}.png"
    payload = {
        "ensemble": asdict(ensemble),
        "experiment": {
            "evaluation_role": evaluation_role,
            "change_threshold": experiment.change_threshold,
            "covariance_inflation": experiment.covariance_inflation,
            "cusum_drift": mpc_config.effectiveness_rls_cusum_drift,
            "cusum_threshold": mpc_config.effectiveness_rls_cusum_threshold,
            "event_detection_window_steps": experiment.event_detection_window_steps,
            "recovery_prior_gain": experiment.recovery_prior_gain,
            "recovery_offset_decay": mpc_config.effectiveness_recovery_offset_decay,
            "recovery_minimum_dwell_steps": (
                mpc_config.effectiveness_recovery_minimum_dwell_steps
            ),
            "recovery_require_full_rank": (
                mpc_config.effectiveness_recovery_require_full_rank
            ),
            "recovery_rank_tolerance": (
                mpc_config.effectiveness_recovery_rank_tolerance
            ),
            "recovery_episode_hysteresis": True,
            "recovery_max_episodes_per_agent": (
                mpc_config.effectiveness_recovery_max_episodes_per_agent
            ),
            "controller_blind_to_hidden_schedule": True,
            "deployment_candidate": (
                "Transient-offset threshold RLS"
                if transient_mode
                else (
                    "Recovery-prior threshold RLS"
                    if args.recovery_development or args.recovery_confirmation
                    else "Innovation-threshold RLS"
                )
            ),
            "candidate_status": (
                "confirmation under evaluation"
                if args.confirmation
                else "provisional pending a fresh confirmation ensemble"
            ),
            "triggered_probing_role": "ablation",
        },
        "summary": summary,
        "trials": records,
    }
    if recovery_comparison is not None:
        comparison_key = (
            "transient_recovery_vs_threshold"
            if transient_mode
            else "recovery_prior_vs_threshold"
        )
        payload[comparison_key] = recovery_comparison
        payload["experiment"]["candidate_status"] = (
            "transient recovery offset under development"
            if transient_mode
            else "recovery prior under development"
        )
    if recovery_confirmation_mode:
        assert recovery_comparison is not None and recovery_criteria is not None
        recovery_confirmation = evaluate_recovery_confirmation(
            recovery_comparison, recovery_criteria
        )
        result_prefix = (
            "transient_recovery_confirmation"
            if args.transient_recovery_confirmation
            else "recovery_confirmation"
        )
        payload[f"{result_prefix}_criteria"] = asdict(recovery_criteria)
        payload[f"{result_prefix}_result"] = recovery_confirmation
        method_name = (
            "transient recovery offset"
            if args.transient_recovery_confirmation
            else "recovery prior"
        )
        payload["experiment"]["candidate_status"] = (
            f"{method_name} confirmed in simulation"
            if recovery_confirmation["passed"]
            else f"{method_name} confirmation criteria not met"
        )
        payload["experiment"]["deployment_candidate"] = (
            (
                "Transient-offset threshold RLS"
                if args.transient_recovery_confirmation
                else "Recovery-prior threshold RLS"
            )
            if recovery_confirmation["passed"]
            else "Innovation-threshold RLS"
        )
    if args.confirmation:
        assert criteria is not None
        confirmation = evaluate_confirmation(summary, criteria)
        payload["confirmation_criteria"] = asdict(criteria)
        payload["confirmation_result"] = confirmation
        payload["experiment"]["candidate_status"] = (
            "confirmed in simulation"
            if confirmation["passed"]
            else "confirmation criteria not met"
        )
    data_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    save_temporary_fault_generalization_plot(
        records,
        summary,
        figure_path,
        study_title=(
            (
                "Independent Transient Recovery-Offset Confirmation"
                if args.transient_recovery_confirmation
                else "Transient Recovery-Offset Development"
            )
            if transient_mode
            else (
                "Channel-Selective Recovery-Prior Development"
                if args.recovery_development
                else (
                    "Independent Recovery-Prior Confirmation"
                    if recovery_confirmation_mode
                    else (
                        "Independent Threshold-RLS Confirmation"
                        if args.confirmation
                        else "Generalization Across Hidden Temporary Actuator Faults"
                    )
                )
            )
        ),
        headline=(
            (
                f"{'transient-offset' if transient_mode else 'recovery-prior'} "
                f"wins {recovery_comparison['recovery_wins']}/"
                f"{recovery_comparison['trials']}; mean recovery RMSE reduction "
                f"{100.0 * recovery_comparison['mean_relative_recovery_rmse_reduction']:.1f}%; "
                f"fallback-free {100.0 * recovery_comparison['fallback_free_rate']:.0f}%"
                + (
                    f"; overall gate "
                    f"{'PASSED' if recovery_confirmation['passed'] else 'FAILED'}"
                    if recovery_confirmation_mode
                    else ""
                )
            )
            if recovery_comparison is not None
            else None
        ),
    )
    print(json.dumps(summary, indent=2))
    if args.confirmation:
        print(json.dumps(payload["confirmation_result"], indent=2))
    if recovery_confirmation_mode:
        result_key = (
            "transient_recovery_confirmation_result"
            if args.transient_recovery_confirmation
            else "recovery_confirmation_result"
        )
        print(json.dumps(payload[result_key], indent=2))
    print(f"Saved temporary-fault generalization data: {data_path}")
    print(f"Saved temporary-fault generalization plot: {figure_path}")


if __name__ == "__main__":
    main()
