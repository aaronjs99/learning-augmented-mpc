"""Evaluate adaptive distributed MPC over hidden temporary actuator faults."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from scripts.harbor import (
    DEFAULT_HARBOR_CONFIG,
    HarborConfirmationCriteriaConfig,
    load_harbor_config,
    load_harbor_confirmation_criteria_config,
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


def main() -> None:
    """Run the stratified ensemble and replace its compact artifacts."""
    args = parse_args()
    agents, simulation, communication = load_harbor_config(args.config)
    base_disturbance, experiment = load_harbor_time_varying_fault_config(args.config)
    if args.confirmation:
        ensemble_section = "temporary_fault_confirmation"
        evaluation_role = "confirmation"
        artifact_stem = "temporary_fault_confirmation"
        criteria = load_harbor_confirmation_criteria_config(args.config)
        controller_labels = criteria.controller_labels
    elif args.holdout:
        ensemble_section = "temporary_fault_holdout"
        evaluation_role = "holdout"
        artifact_stem = "temporary_fault_holdout"
        controller_labels = None
        criteria = None
    else:
        ensemble_section = "temporary_fault_ensemble"
        evaluation_role = "development"
        artifact_stem = "temporary_fault_generalization"
        controller_labels = None
        criteria = None
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
            "controller_blind_to_hidden_schedule": True,
            "deployment_candidate": "Innovation-threshold RLS",
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
            "Independent Threshold-RLS Confirmation"
            if args.confirmation
            else "Generalization Across Hidden Temporary Actuator Faults"
        ),
    )
    print(json.dumps(summary, indent=2))
    if args.confirmation:
        print(json.dumps(payload["confirmation_result"], indent=2))
    print(f"Saved temporary-fault generalization data: {data_path}")
    print(f"Saved temporary-fault generalization plot: {figure_path}")


if __name__ == "__main__":
    main()
