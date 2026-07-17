"""Test elastic dynamic-state envelopes under hidden current and actuator loss."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from scripts.harbor import (
    DEFAULT_HARBOR_CONFIG,
    HarborDynamicEnvelopeCriteriaConfig,
    load_harbor_config,
    load_harbor_dynamic_envelope_criteria_config,
    load_harbor_fault_study_config,
    load_harbor_observation_noise_config,
    load_harbor_temporary_fault_ensemble_config,
    load_harbor_time_varying_fault_config,
)
from scripts.harbor.experiments import run_temporary_fault_generalization
from scripts.harbor.mpc import load_harbor_mpc_config
from scripts.harbor.plotting import save_dynamic_envelope_plot
from scripts.run_harbor_time_varying_fault_study import (
    build_time_varying_fault_records,
    compact_time_varying_fault_records,
)


BASELINE = "Hard-envelope transient-offset RLS"
CANDIDATE = "Retry-elastic transient-offset RLS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_HARBOR_CONFIG))
    parser.add_argument("--artifact-dir", default="results/latest/harbor")
    parser.add_argument(
        "--confirmation",
        action="store_true",
        help="use the untouched dynamic-envelope confirmation ensemble",
    )
    return parser.parse_args()


def summarize_dynamic_envelope(records: list[dict]) -> dict:
    """Build paired task, solver, estimator, and relaxation metrics."""
    seeds = sorted({record["seed"] for record in records})
    by_key = {(record["seed"], record["controller"]): record for record in records}
    baseline = [by_key[(seed, BASELINE)] for seed in seeds]
    candidate = [by_key[(seed, CANDIDATE)] for seed in seeds]
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
        "maximum_dynamic_state_slack": float(
            max(record["max_dynamic_state_slack"] for record in candidate)
        ),
        "mean_dynamic_state_slack": float(
            np.mean([record["max_dynamic_state_slack"] for record in candidate])
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


def evaluate_dynamic_envelope(
    summary: dict, criteria: HarborDynamicEnvelopeCriteriaConfig
) -> dict:
    """Apply frozen completion, safety, and bounded-relaxation gates."""
    trials = summary["trials"]
    checks = {
        "candidate_completion_rate": summary["candidate_completion_rate"]
        >= criteria.minimum_candidate_completion_rate,
        "completion_rescue_rate": summary["completion_rescues"] / trials
        >= criteria.minimum_completion_rescue_rate,
        "completion_regression_rate": summary["completion_regressions"] / trials
        <= criteria.maximum_completion_regression_rate,
        "safety_rate": summary["candidate_safety_rate"]
        >= criteria.minimum_safety_rate,
        "fallback_free_rate": summary["candidate_fallback_free_rate"]
        >= criteria.minimum_fallback_free_rate,
        "completion_cost_delta": summary["mean_completion_cost_delta"]
        <= criteria.maximum_mean_completion_cost_delta,
        "recovery_rmse_delta": summary["mean_recovery_rmse_delta"]
        <= criteria.maximum_mean_recovery_rmse_delta,
        "current_rmse_delta": summary["mean_current_rmse_delta"]
        <= criteria.maximum_mean_current_rmse_delta,
        "dynamic_state_slack": summary["maximum_dynamic_state_slack"]
        <= criteria.maximum_dynamic_state_slack,
    }
    return {"passed": all(checks.values()), "checks": checks, "observed": summary}


def main() -> None:
    """Run matched hard-versus-elastic envelope trials and save compact evidence."""
    args = parse_args()
    agents, simulation, communication = load_harbor_config(args.config)
    base_disturbance, experiment = load_harbor_time_varying_fault_config(
        args.config
    )
    section = (
        "dynamic_envelope_retry_confirmation"
        if args.confirmation
        else "dynamic_envelope_retry_development"
    )
    ensemble = load_harbor_temporary_fault_ensemble_config(
        args.config, section=section
    )
    criteria = load_harbor_dynamic_envelope_criteria_config(args.config)
    labels = criteria.controller_labels
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
    summary = summarize_dynamic_envelope(records)
    gate_result = evaluate_dynamic_envelope(summary, criteria)
    payload = {
        "ensemble": asdict(ensemble),
        "experiment": {
            "evaluation_role": section,
            "baseline_dynamic_state_slack_bound": 0.0,
            "candidate_primary_dynamic_state_slack_bound": 0.0,
            "candidate_retry_dynamic_state_slack_bound": mpc_config.dynamic_state_slack_bound,
            "dynamic_state_slack_weight": mpc_config.dynamic_state_slack_weight,
            "hard_domain_and_collision_constraints_retained": True,
            "controller_blind_to_hidden_fault_and_current": True,
        },
        "dynamic_envelope_summary": summary,
        "dynamic_envelope_criteria": asdict(criteria),
        "trials": compact_time_varying_fault_records(records),
    }
    payload[
        "dynamic_envelope_confirmation_result"
        if args.confirmation
        else "dynamic_envelope_development_screen"
    ] = gate_result
    output = Path(args.artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / f"{section}.json"
    figure_path = output / f"{section}.png"
    data_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    save_dynamic_envelope_plot(
        records,
        summary,
        figure_path,
        study_title=(
            "Independent Elastic Dynamic-Envelope Confirmation"
            if args.confirmation
            else "Elastic Dynamic-Envelope Development"
        ),
        confirmation_passed=(
            gate_result["passed"] if args.confirmation else None
        ),
    )
    print(json.dumps(summary, indent=2))
    print(json.dumps(gate_result, indent=2))
    print(f"Saved dynamic-envelope data: {data_path}")
    print(f"Saved dynamic-envelope plot: {figure_path}")


if __name__ == "__main__":
    main()
