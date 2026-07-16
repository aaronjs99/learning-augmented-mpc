"""Compare scalar and channel-wise adaptation under asymmetric actuator faults."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.harbor import (
    DEFAULT_HARBOR_CONFIG,
    load_harbor_config,
    load_harbor_fault_config,
    load_harbor_fault_study_config,
)
from scripts.harbor.experiments import run_actuator_fault_study
from scripts.harbor.mpc import load_harbor_mpc_config
from scripts.harbor.plotting import (
    save_actuator_fault_diagnostics,
    save_harbor_animation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_HARBOR_CONFIG))
    parser.add_argument("--artifact-dir", default="results/latest/harbor")
    parser.add_argument("--no-gif", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the asymmetric fault study and replace its curated artifacts."""
    args = parse_args()
    agents, simulation, communication = load_harbor_config(args.config)
    disturbance = load_harbor_fault_config(args.config)
    mpc_config = load_harbor_mpc_config(args.config)
    trials = run_actuator_fault_study(
        agents,
        simulation,
        communication,
        mpc_config,
        disturbance,
        load_harbor_fault_study_config(args.config),
    )
    records = []
    for trial in trials:
        errors = np.concatenate(
            [
                trial.final_effectiveness_estimates[agent.name]
                - disturbance.effectiveness(agent.model, agent.name)
                for agent in agents
            ]
        )
        records.append(
            {
                "controller": trial.label,
                "identification_strategy": trial.identification_strategy,
                "learning_source": trial.source_controller,
                "valid": trial.valid,
                "all_goals_reached": trial.result.all_goals_reached,
                "completion_step_sum": trial.completion_step_sum,
                "first_goal_steps": trial.result.first_goal_steps,
                "final_goal_errors": trial.result.final_goal_errors,
                "final_orientation_errors": trial.result.final_orientation_errors,
                "min_pairwise_distance": trial.result.min_pairwise_distance,
                "pairwise_violation_count": trial.result.pairwise_violation_count,
                "solver_fallbacks": trial.solver_fallbacks,
                "max_collision_slack": trial.max_collision_slack,
                "effectiveness_rmse": float(np.sqrt(np.mean(errors * errors))),
                "hidden_effectiveness": {
                    agent.name: disturbance.effectiveness(
                        agent.model, agent.name
                    ).tolist()
                    for agent in agents
                },
                "final_effectiveness_estimates": {
                    name: value.tolist()
                    for name, value in trial.final_effectiveness_estimates.items()
                },
                "probe_count_by_agent": trial.probe_count_by_agent,
                "probe_channel_counts": {
                    name: value.tolist()
                    for name, value in trial.probe_channel_counts.items()
                },
                "probe_sequence_by_agent": trial.probe_sequence_by_agent,
                "probe_rejection_counts": {
                    name: value.tolist()
                    for name, value in trial.probe_rejection_counts.items()
                },
                "final_excitation_energy": {
                    name: (
                        values[-1].tolist()
                        if len(values)
                        else np.zeros(
                            next(
                                agent.model.control_dim
                                for agent in agents
                                if agent.name == name
                            )
                        ).tolist()
                    )
                    for name, values in trial.excitation_history.items()
                },
                "final_information_std": {
                    name: (
                        values[-1].tolist()
                        if len(values)
                        else np.full(
                            next(
                                agent.model.control_dim
                                for agent in agents
                                if agent.name == name
                            ),
                            mpc_config.identification_prior_std,
                        ).tolist()
                    )
                    for name, values in trial.information_std_history.items()
                },
            }
        )
    output = Path(args.artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "actuator_fault_study.json"
    data_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    figure = save_actuator_fault_diagnostics(
        trials,
        agents,
        simulation,
        disturbance,
        output / "actuator_fault_diagnostics.png",
    )
    adaptive = next(
        trial
        for trial in trials
        if trial.label == "Retained information-ID LMPC"
    )
    if not args.no_gif:
        save_harbor_animation(
            adaptive.result,
            agents,
            simulation,
            output / "fault_aware_harbor_lmpc.gif",
            label="Retained information-ID distributed LMPC",
        )
    print(json.dumps(records, indent=2))
    print(f"Saved actuator-fault diagnostics: {figure}")


if __name__ == "__main__":
    main()
