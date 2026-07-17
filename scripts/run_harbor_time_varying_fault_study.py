"""Evaluate online identification of scheduled heterogeneous actuator faults."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.harbor import (
    DEFAULT_HARBOR_CONFIG,
    load_harbor_config,
    load_harbor_fault_study_config,
    load_harbor_observation_noise_config,
    load_harbor_time_varying_fault_config,
)
from scripts.harbor.experiments import run_time_varying_fault_study
from scripts.harbor.mpc import load_harbor_mpc_config
from scripts.harbor.plotting import save_time_varying_fault_plot


CONTROLLER_LABELS = (
    "Fixed-covariance RLS",
    "Innovation-threshold RLS",
    "Chi-square CUSUM RLS",
    "CUSUM-triggered probing RLS",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_HARBOR_CONFIG))
    parser.add_argument("--artifact-dir", default="results/latest/harbor")
    return parser.parse_args()


def summarize_time_varying_faults(records: list[dict]) -> dict:
    """Aggregate tracking and task metrics for matched observation seeds."""
    controllers = {}
    for label in CONTROLLER_LABELS:
        selected = [record for record in records if record["controller"] == label]
        delays = [
            record["mean_detection_delay"]
            for record in selected
            if record["mean_detection_delay"] is not None
        ]
        controllers[label] = {
            "trials": len(selected),
            "mean_fault_interval_rmse": float(
                np.mean([record["fault_interval_rmse"] for record in selected])
            ),
            "mean_recovery_rmse": float(
                np.mean([record["recovery_rmse"] for record in selected])
            ),
            "mean_final_rmse": float(
                np.mean([record["final_effectiveness_rmse"] for record in selected])
            ),
            "mean_sustained_completion_cost": float(
                np.mean(
                    [record["sustained_completion_cost"] for record in selected]
                )
            ),
            "fallback_free_rate": float(
                np.mean([record["solver_fallbacks"] == 0 for record in selected])
            ),
            "completion_rate": float(
                np.mean([record["all_goals_reached"] for record in selected])
            ),
            "safety_rate": float(
                np.mean(
                    [
                        record["pairwise_violation_count"] == 0
                        and record["max_collision_slack"] <= 1e-9
                        for record in selected
                    ]
                )
            ),
            "mean_change_detections": float(
                np.mean(
                    [
                        sum(len(steps) for steps in record["change_steps_by_agent"].values())
                        for record in selected
                    ]
                )
            ),
            "mean_event_recall": float(
                np.mean([record["event_recall"] for record in selected])
            ),
            "mean_false_inflations": float(
                np.mean([record["false_inflations"] for record in selected])
            ),
            "mean_detection_delay": float(np.mean(delays)) if delays else None,
            "mean_probe_count": float(
                np.mean([record["probe_count"] for record in selected])
            ),
        }
    by_key = {
        (record["seed"], record["controller"]): record for record in records
    }
    seeds = sorted({record["seed"] for record in records})
    fixed = np.asarray(
        [by_key[(seed, CONTROLLER_LABELS[0])]["fault_interval_rmse"] for seed in seeds]
    )
    fixed_final = np.asarray(
        [
            by_key[(seed, CONTROLLER_LABELS[0])]["final_effectiveness_rmse"]
            for seed in seeds
        ]
    )
    fixed_recovery = np.asarray(
        [by_key[(seed, CONTROLLER_LABELS[0])]["recovery_rmse"] for seed in seeds]
    )
    paired = {}
    for label in CONTROLLER_LABELS[1:]:
        adaptive = np.asarray(
            [by_key[(seed, label)]["fault_interval_rmse"] for seed in seeds]
        )
        adaptive_final = np.asarray(
            [by_key[(seed, label)]["final_effectiveness_rmse"] for seed in seeds]
        )
        adaptive_recovery = np.asarray(
            [by_key[(seed, label)]["recovery_rmse"] for seed in seeds]
        )
        reduction = fixed - adaptive
        final_reduction = fixed_final - adaptive_final
        recovery_reduction = fixed_recovery - adaptive_recovery
        paired[label] = {
            "trials": len(seeds),
            "adaptive_wins": int(np.count_nonzero(reduction > 0.0)),
            "mean_rmse_reduction": float(np.mean(reduction)),
            "mean_relative_rmse_reduction": float(
                np.mean(reduction / np.maximum(fixed, 1e-12))
            ),
            "mean_recovery_rmse_reduction": float(np.mean(recovery_reduction)),
            "mean_relative_recovery_rmse_reduction": float(
                np.mean(recovery_reduction / np.maximum(fixed_recovery, 1e-12))
            ),
            "mean_final_rmse_reduction": float(np.mean(final_reduction)),
            "mean_relative_final_rmse_reduction": float(
                np.mean(final_reduction / np.maximum(fixed_final, 1e-12))
            ),
            "mean_completion_cost_delta": float(
                np.mean(
                    [
                        by_key[(seed, label)]["sustained_completion_cost"]
                        - by_key[(seed, CONTROLLER_LABELS[0])][
                            "sustained_completion_cost"
                        ]
                        for seed in seeds
                    ]
                )
            ),
        }
    return {
        "controllers": controllers,
        "paired_vs_fixed": paired,
    }


def _score_inflation_events(
    scheduled_steps: dict[str, list[int]],
    triggered_steps: dict[str, list[int]],
    window_steps: int,
) -> tuple[float, int, float | None]:
    """Score causal event recall, unmatched inflations, and detection delay."""
    event_count = 0
    matched_count = 0
    false_count = 0
    delays = []
    for name, events in scheduled_steps.items():
        triggers = list(triggered_steps[name])
        used = set()
        event_count += len(events)
        for event in events:
            match = next(
                (
                    (index, trigger)
                    for index, trigger in enumerate(triggers)
                    if index not in used and event <= trigger <= event + window_steps
                ),
                None,
            )
            if match is not None:
                index, trigger = match
                used.add(index)
                matched_count += 1
                delays.append(trigger - event)
        false_count += len(triggers) - len(used)
    recall = matched_count / event_count if event_count else 1.0
    return recall, false_count, float(np.mean(delays)) if delays else None


def main() -> None:
    """Run the matched scheduled-fault study and replace its artifacts."""
    args = parse_args()
    agents, simulation, communication = load_harbor_config(args.config)
    disturbance, experiment = load_harbor_time_varying_fault_config(args.config)
    mpc_config = load_harbor_mpc_config(args.config)
    cases = run_time_varying_fault_study(
        agents,
        simulation,
        communication,
        mpc_config,
        disturbance,
        load_harbor_fault_study_config(args.config),
        experiment,
        load_harbor_observation_noise_config(args.config),
    )
    fault_event_steps = {
        name: [event[0] for event in events]
        for name, events in disturbance.agent_control_effectiveness_schedule.items()
        if events
    }
    if any(len(events) < 2 for events in fault_event_steps.values()):
        raise ValueError(
            "time-varying fault study requires an onset and recovery per agent"
        )
    records = []
    for case in cases:
        for trial in case.trials:
            count = min(
                len(trial.effectiveness_history[agent.name]) for agent in agents
            )
            global_history = []
            platform_histories = {agent.name: [] for agent in agents}
            for step in range(count):
                errors = []
                for agent in agents:
                    estimate = trial.effectiveness_history[agent.name][step]
                    truth = trial.result.applied_effectiveness[agent.name][step]
                    platform_error = float(
                        np.sqrt(np.mean((estimate - truth) ** 2))
                    )
                    platform_histories[agent.name].append(platform_error)
                    errors.extend((estimate - truth).tolist())
                global_history.append(float(np.sqrt(np.mean(np.square(errors)))))
            platform_fault_rmse = {
                agent.name: float(
                    np.mean(
                        platform_histories[agent.name][
                            fault_event_steps[agent.name][0] : fault_event_steps[
                                agent.name
                            ][1]
                        ]
                    )
                )
                for agent in agents
            }
            platform_recovery_rmse = {
                agent.name: float(
                    np.mean(
                        platform_histories[agent.name][
                            fault_event_steps[agent.name][1] :
                        ]
                    )
                )
                for agent in agents
            }
            fault_interval_rmse = float(np.mean(list(platform_fault_rmse.values())))
            recovery_rmse = float(np.mean(list(platform_recovery_rmse.values())))
            event_recall, false_inflations, mean_detection_delay = (
                _score_inflation_events(
                    fault_event_steps,
                    trial.effectiveness_change_steps_by_agent,
                    experiment.event_detection_window_steps,
                )
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
                    "controller": trial.label,
                    "fault_event_steps": fault_event_steps,
                    "tracking_rmse_history": global_history,
                    "platform_fault_rmse": platform_fault_rmse,
                    "platform_recovery_rmse": platform_recovery_rmse,
                    "fault_interval_rmse": fault_interval_rmse,
                    "recovery_rmse": recovery_rmse,
                    "final_effectiveness_rmse": global_history[-1],
                    "change_steps_by_agent": (
                        trial.effectiveness_change_steps_by_agent
                    ),
                    "all_goals_reached": trial.result.all_goals_reached,
                    "pairwise_violation_count": trial.result.pairwise_violation_count,
                    "max_collision_slack": trial.max_collision_slack,
                    "solver_fallbacks": trial.solver_fallbacks,
                    "sustained_completion_cost": sustained_cost,
                    "event_recall": event_recall,
                    "false_inflations": false_inflations,
                    "mean_detection_delay": mean_detection_delay,
                    "probe_count": sum(trial.probe_count_by_agent.values()),
                    "probe_count_by_agent": trial.probe_count_by_agent,
                }
            )
    summary = summarize_time_varying_faults(records)
    output = Path(args.artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "time_varying_fault_study.json"
    figure_path = output / "time_varying_fault_study.png"
    payload = {
        "experiment": {
            "observation_seeds": list(experiment.observation_seeds),
            "change_threshold": experiment.change_threshold,
            "covariance_inflation": experiment.covariance_inflation,
            "change_persistence": mpc_config.effectiveness_rls_change_persistence,
            "change_cooldown_steps": (
                mpc_config.effectiveness_rls_change_cooldown_steps
            ),
            "cusum_drift": mpc_config.effectiveness_rls_cusum_drift,
            "cusum_threshold": mpc_config.effectiveness_rls_cusum_threshold,
            "event_detection_window_steps": experiment.event_detection_window_steps,
            "fault_event_steps": fault_event_steps,
        },
        "summary": summary,
        "trials": records,
    }
    data_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    save_time_varying_fault_plot(records, summary, figure_path)
    print(json.dumps(summary, indent=2))
    print(f"Saved time-varying-fault data: {data_path}")
    print(f"Saved time-varying-fault plot: {figure_path}")


if __name__ == "__main__":
    main()
