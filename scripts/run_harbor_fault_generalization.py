"""Evaluate actuator-identification policies over stratified hidden faults."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from scripts.harbor import (
    DEFAULT_HARBOR_CONFIG,
    load_harbor_config,
    load_harbor_fault_config,
    load_harbor_fault_ensemble_config,
    load_harbor_fault_study_config,
    load_harbor_observation_noise_config,
)
from scripts.harbor.experiments import run_actuator_fault_generalization
from scripts.harbor.mpc import load_harbor_mpc_config
from scripts.harbor.plotting import save_fault_generalization_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_HARBOR_CONFIG))
    parser.add_argument("--artifact-dir", default="results/latest/harbor")
    parser.add_argument("--with-observation-noise", action="store_true")
    return parser.parse_args()


def summarize_fault_generalization(records, bootstrap_samples: int) -> dict:
    """Return controller aggregates and a paired equal-budget comparison."""
    labels = tuple(dict.fromkeys(record["controller"] for record in records))
    controllers = {}
    for label in labels:
        selected = [record for record in records if record["controller"] == label]
        controllers[label] = {
            "trials": len(selected),
            "mean_effectiveness_rmse": float(
                np.mean([record["effectiveness_rmse"] for record in selected])
            ),
            "std_effectiveness_rmse": float(
                np.std(
                    [record["effectiveness_rmse"] for record in selected], ddof=1
                )
                if len(selected) > 1
                else 0.0
            ),
            "mean_first_hit_step_sum": float(
                np.mean([record["first_hit_step_sum"] for record in selected])
            ),
            "mean_sustained_completion_cost": float(
                np.mean(
                    [record["sustained_completion_cost"] for record in selected]
                )
            ),
            "valid_rate": float(np.mean([record["valid"] for record in selected])),
            "completion_rate": float(
                np.mean([record["all_goals_reached"] for record in selected])
            ),
            "safety_rate": float(
                np.mean(
                    [record["pairwise_violation_count"] == 0 for record in selected]
                )
            ),
        }

    by_key = {
        (record["seed"], record["controller"]): record for record in records
    }
    seeds = sorted({record["seed"] for record in records})
    labels = {record["controller"] for record in records}
    one_pass_label = (
        "Recursive one-pass MPC"
        if "Recursive one-pass MPC" in labels
        else "One-pass active MPC"
    )
    information_label = (
        "Recursive information-aware MPC"
        if "Recursive information-aware MPC" in labels
        else "Information-aware MPC"
    )
    one_pass = np.asarray(
        [by_key[(seed, one_pass_label)]["effectiveness_rmse"] for seed in seeds]
    )
    information = np.asarray(
        [by_key[(seed, information_label)]["effectiveness_rmse"] for seed in seeds]
    )
    differences = one_pass - information
    rng = np.random.default_rng(20260716)
    bootstrap = np.mean(
        rng.choice(differences, size=(bootstrap_samples, len(differences)), replace=True),
        axis=1,
    )
    cost_differences = np.asarray(
        [
            by_key[(seed, information_label)]["sustained_completion_cost"]
            - by_key[(seed, one_pass_label)]["sustained_completion_cost"]
            for seed in seeds
        ],
        dtype=float,
    )
    recursive_comparison = None
    if "Recursive diagonal MPC" in labels and "Instantaneous diagonal MPC" in labels:
        instantaneous = np.asarray(
            [
                by_key[(seed, "Instantaneous diagonal MPC")]["effectiveness_rmse"]
                for seed in seeds
            ]
        )
        recursive = np.asarray(
            [
                by_key[(seed, "Recursive diagonal MPC")]["effectiveness_rmse"]
                for seed in seeds
            ]
        )
        recursive_differences = instantaneous - recursive
        recursive_bootstrap = np.mean(
            rng.choice(
                recursive_differences,
                size=(bootstrap_samples, len(recursive_differences)),
                replace=True,
            ),
            axis=1,
        )
        recursive_comparison = {
            "trials": len(seeds),
            "recursive_wins": int(np.count_nonzero(recursive_differences > 0.0)),
            "mean_rmse_reduction": float(np.mean(recursive_differences)),
            "mean_relative_rmse_reduction": float(
                np.mean(recursive_differences / np.maximum(instantaneous, 1e-12))
            ),
            "paired_mean_reduction_bootstrap_95_ci": [
                float(value)
                for value in np.quantile(recursive_bootstrap, [0.025, 0.975])
            ],
        }
    return {
        "controllers": controllers,
        "recursive_vs_instantaneous": recursive_comparison,
        "equal_budget_information_vs_one_pass": {
            "trials": len(seeds),
            "information_wins": int(np.count_nonzero(differences > 0.0)),
            "ties": int(np.count_nonzero(np.isclose(differences, 0.0))),
            "mean_rmse_reduction": float(np.mean(differences)),
            "mean_relative_rmse_reduction": float(
                np.mean(differences / np.maximum(one_pass, 1e-12))
            ),
            "paired_mean_reduction_bootstrap_95_ci": [
                float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
            ],
            "mean_completion_cost_delta": float(np.mean(cost_differences)),
            "max_absolute_completion_cost_delta": float(
                np.max(np.abs(cost_differences))
            ),
        },
    }


def main() -> None:
    """Run the ensemble and replace its JSON and plot artifacts."""
    args = parse_args()
    agents, simulation, communication = load_harbor_config(args.config)
    ensemble_config = load_harbor_fault_ensemble_config(args.config)
    observation_noise = load_harbor_observation_noise_config(args.config)
    if args.with_observation_noise:
        observation_noise = replace(observation_noise, enabled=True)
    else:
        observation_noise = None
    cases = run_actuator_fault_generalization(
        agents,
        simulation,
        communication,
        load_harbor_mpc_config(args.config),
        load_harbor_fault_config(args.config),
        load_harbor_fault_study_config(args.config),
        ensemble_config,
        observation_noise=observation_noise,
    )
    records = []
    for case in cases:
        truth = np.concatenate(
            [
                case.disturbance.effectiveness(agent.model, agent.name)
                for agent in agents
            ]
        )
        hidden = {
            agent.name: case.disturbance.effectiveness(
                agent.model, agent.name
            ).tolist()
            for agent in agents
        }
        for trial in case.trials:
            estimate = np.concatenate(
                [trial.final_effectiveness_estimates[agent.name] for agent in agents]
            )
            sustained_cost = sum(
                (
                    trial.result.first_goal_steps[agent.name]
                    if trial.result.first_goal_steps[agent.name] is not None
                    and trial.result.final_goal_errors[agent.name]
                    <= simulation.goal_tolerance
                    and trial.result.final_orientation_errors[agent.name]
                    <= simulation.orientation_tolerance
                    else simulation.horizon + 1
                )
                for agent in agents
            )
            records.append(
                {
                    "seed": case.seed,
                    "observation_seed": case.observation_seed,
                    "controller": trial.label,
                    "valid": trial.valid,
                    "all_goals_reached": trial.result.all_goals_reached,
                    "first_hit_step_sum": trial.completion_step_sum,
                    "sustained_completion_cost": sustained_cost,
                    "first_goal_steps": trial.result.first_goal_steps,
                    "final_goal_errors": trial.result.final_goal_errors,
                    "final_orientation_errors": trial.result.final_orientation_errors,
                    "min_pairwise_distance": trial.result.min_pairwise_distance,
                    "pairwise_violation_count": trial.result.pairwise_violation_count,
                    "solver_fallbacks": trial.solver_fallbacks,
                    "solver_fallbacks_by_agent": trial.solver_fallbacks_by_agent,
                    "solver_failure_status_counts": trial.solver_failure_status_counts,
                    "effectiveness_rmse": float(
                        np.sqrt(np.mean((estimate - truth) ** 2))
                    ),
                    "probe_count": int(sum(trial.probe_count_by_agent.values())),
                    "hidden_effectiveness": hidden,
                    "final_effectiveness_estimates": {
                        name: values.tolist()
                        for name, values in trial.final_effectiveness_estimates.items()
                    },
                    "probe_sequence_by_agent": trial.probe_sequence_by_agent,
                }
            )
    summary = summarize_fault_generalization(
        records, ensemble_config.bootstrap_samples
    )
    payload = {
        "ensemble": {
            "seeds": list(ensemble_config.seeds),
            "effectiveness_min": ensemble_config.effectiveness_min,
            "effectiveness_max": ensemble_config.effectiveness_max,
            "sampling": "per-channel Latin hypercube",
        },
        "observation_noise": (
            {
                "enabled": True,
                "base_seed": observation_noise.seed,
                "kind_state_std": observation_noise.kind_state_std,
                "agent_state_std": observation_noise.agent_state_std,
            }
            if observation_noise is not None
            else {"enabled": False}
        ),
        "summary": summary,
        "trials": records,
    }
    output = Path(args.artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = "fault_noise_generalization" if observation_noise is not None else "fault_generalization"
    data_path = output / f"{stem}.json"
    data_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    figure = save_fault_generalization_plot(
        records,
        summary,
        output / f"{stem}.png",
        noisy_observations=observation_noise is not None,
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved fault-generalization data: {data_path}")
    print(f"Saved fault-generalization plot: {figure}")


if __name__ == "__main__":
    main()
