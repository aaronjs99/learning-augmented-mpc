"""Evaluate adaptive distributed MPC over hidden temporary actuator faults."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from scripts.harbor import (
    DEFAULT_HARBOR_CONFIG,
    load_harbor_config,
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
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="use the separately seeded holdout ensemble and artifact names",
    )
    return parser.parse_args()


def main() -> None:
    """Run the stratified ensemble and replace its compact artifacts."""
    args = parse_args()
    agents, simulation, communication = load_harbor_config(args.config)
    base_disturbance, experiment = load_harbor_time_varying_fault_config(args.config)
    ensemble_section = (
        "temporary_fault_holdout" if args.holdout else "temporary_fault_ensemble"
    )
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
    )
    records = build_time_varying_fault_records(
        cases, agents, simulation, experiment.event_detection_window_steps
    )
    summary = summarize_time_varying_faults(records, ensemble.bootstrap_samples)
    output = Path(args.artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifact_stem = (
        "temporary_fault_holdout"
        if args.holdout
        else "temporary_fault_generalization"
    )
    data_path = output / f"{artifact_stem}.json"
    figure_path = output / f"{artifact_stem}.png"
    payload = {
        "ensemble": asdict(ensemble),
        "experiment": {
            "evaluation_role": "holdout" if args.holdout else "development",
            "change_threshold": experiment.change_threshold,
            "covariance_inflation": experiment.covariance_inflation,
            "cusum_drift": mpc_config.effectiveness_rls_cusum_drift,
            "cusum_threshold": mpc_config.effectiveness_rls_cusum_threshold,
            "event_detection_window_steps": experiment.event_detection_window_steps,
            "controller_blind_to_hidden_schedule": True,
            "deployment_candidate": "Innovation-threshold RLS",
            "candidate_status": "provisional pending a fresh confirmation ensemble",
            "triggered_probing_role": "ablation",
        },
        "summary": summary,
        "trials": records,
    }
    data_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    save_temporary_fault_generalization_plot(records, summary, figure_path)
    print(json.dumps(summary, indent=2))
    print(f"Saved temporary-fault generalization data: {data_path}")
    print(f"Saved temporary-fault generalization plot: {figure_path}")


if __name__ == "__main__":
    main()
