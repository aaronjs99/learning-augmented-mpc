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


def summarize_time_varying_faults(
    records: list[dict], bootstrap_samples: int = 5000
) -> dict:
    """Aggregate tracking and task metrics for matched observation seeds."""
    present_labels = {record["controller"] for record in records}
    labels = tuple(label for label in CONTROLLER_LABELS if label in present_labels)
    if not labels or labels[0] != CONTROLLER_LABELS[0]:
        raise ValueError(
            "temporary-fault records require the fixed-covariance comparator"
        )
    controllers = {}
    for label in labels:
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
                        sum(
                            len(steps)
                            for steps in record["change_steps_by_agent"].values()
                        )
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
        [by_key[(seed, labels[0])]["fault_interval_rmse"] for seed in seeds]
    )
    fixed_final = np.asarray(
        [
            by_key[(seed, labels[0])]["final_effectiveness_rmse"]
            for seed in seeds
        ]
    )
    fixed_recovery = np.asarray(
        [by_key[(seed, labels[0])]["recovery_rmse"] for seed in seeds]
    )
    paired = {}
    rng = np.random.default_rng(20260716)
    for label in labels[1:]:
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
        rmse_bootstrap = np.mean(
            rng.choice(
                reduction,
                size=(bootstrap_samples, len(reduction)),
                replace=True,
            ),
            axis=1,
        )
        recovery_bootstrap = np.mean(
            rng.choice(
                recovery_reduction,
                size=(bootstrap_samples, len(recovery_reduction)),
                replace=True,
            ),
            axis=1,
        )
        paired[label] = {
            "trials": len(seeds),
            "adaptive_wins": int(np.count_nonzero(reduction > 0.0)),
            "recovery_wins": int(np.count_nonzero(recovery_reduction > 0.0)),
            "final_wins": int(np.count_nonzero(final_reduction > 0.0)),
            "mean_rmse_reduction": float(np.mean(reduction)),
            "mean_relative_rmse_reduction": float(
                np.mean(reduction / np.maximum(fixed, 1e-12))
            ),
            "paired_mean_rmse_reduction_bootstrap_95_ci": [
                float(value) for value in np.quantile(rmse_bootstrap, [0.025, 0.975])
            ],
            "mean_recovery_rmse_reduction": float(np.mean(recovery_reduction)),
            "mean_relative_recovery_rmse_reduction": float(
                np.mean(recovery_reduction / np.maximum(fixed_recovery, 1e-12))
            ),
            "paired_mean_recovery_reduction_bootstrap_95_ci": [
                float(value)
                for value in np.quantile(recovery_bootstrap, [0.025, 0.975])
            ],
            "mean_final_rmse_reduction": float(np.mean(final_reduction)),
            "mean_relative_final_rmse_reduction": float(
                np.mean(final_reduction / np.maximum(fixed_final, 1e-12))
            ),
            "mean_completion_cost_delta": float(
                np.mean(
                    [
                        by_key[(seed, label)]["sustained_completion_cost"]
                        - by_key[(seed, labels[0])][
                            "sustained_completion_cost"
                        ]
                        for seed in seeds
                    ]
                )
            ),
        }
    summary = {
        "controllers": controllers,
        "paired_vs_fixed": paired,
    }
    passive_label = "Chi-square CUSUM RLS"
    active_label = "CUSUM-triggered probing RLS"
    if passive_label in labels and active_label in labels:
        passive_fault = np.asarray(
            [by_key[(seed, passive_label)]["fault_interval_rmse"] for seed in seeds]
        )
        active_fault = np.asarray(
            [by_key[(seed, active_label)]["fault_interval_rmse"] for seed in seeds]
        )
        passive_recovery = np.asarray(
            [by_key[(seed, passive_label)]["recovery_rmse"] for seed in seeds]
        )
        active_recovery = np.asarray(
            [by_key[(seed, active_label)]["recovery_rmse"] for seed in seeds]
        )
        summary["triggered_probing_vs_passive_cusum"] = {
            "trials": len(seeds),
            "degraded_interval_wins": int(
                np.count_nonzero(active_fault < passive_fault)
            ),
            "mean_fault_rmse_reduction": float(
                np.mean(passive_fault - active_fault)
            ),
            "recovery_wins": int(
                np.count_nonzero(active_recovery < passive_recovery)
            ),
            "mean_recovery_rmse_reduction": float(
                np.mean(passive_recovery - active_recovery)
            ),
            "mean_completion_cost_delta": float(
                np.mean(
                    [
                        by_key[(seed, active_label)]["sustained_completion_cost"]
                        - by_key[(seed, passive_label)]["sustained_completion_cost"]
                        for seed in seeds
                    ]
                )
            ),
            "additional_solver_fallbacks": int(
                sum(by_key[(seed, active_label)]["solver_fallbacks"] for seed in seeds)
                - sum(
                    by_key[(seed, passive_label)]["solver_fallbacks"] for seed in seeds
                )
            ),
        }
    return summary


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


def build_time_varying_fault_records(
    cases, agents, simulation, event_detection_window_steps: int
) -> list[dict]:
    """Convert matched temporary-fault trials into schedule-aware metrics."""
    records = []
    for case in cases:
        fault_event_steps = {
            name: [event[0] for event in events]
            for name, events in case.disturbance.agent_control_effectiveness_schedule.items()
            if events
        }
        if set(fault_event_steps) != {agent.name for agent in agents} or any(
            len(events) != 2 for events in fault_event_steps.values()
        ):
            raise ValueError(
                "temporary fault evaluation requires one onset and recovery per agent"
            )
        hidden_schedule = {
            name: [
                {"step": int(step), "effectiveness": list(value)}
                for step, value in events
            ]
            for name, events in (
                case.disturbance.agent_control_effectiveness_schedule.items()
            )
        }
        for trial in case.trials:
            count = min(len(trial.effectiveness_history[agent.name]) for agent in agents)
            global_history = []
            platform_histories = {agent.name: [] for agent in agents}
            for step in range(count):
                errors = []
                for agent in agents:
                    estimate = trial.effectiveness_history[agent.name][step]
                    truth = trial.result.applied_effectiveness[agent.name][step]
                    platform_error = float(np.sqrt(np.mean((estimate - truth) ** 2)))
                    platform_histories[agent.name].append(platform_error)
                    errors.extend((estimate - truth).tolist())
                global_history.append(float(np.sqrt(np.mean(np.square(errors)))))
            platform_fault_rmse = {
                agent.name: float(
                    np.mean(
                        platform_histories[agent.name][
                            fault_event_steps[agent.name][0] : fault_event_steps[agent.name][1]
                        ]
                    )
                )
                for agent in agents
            }
            platform_recovery_rmse = {
                agent.name: float(
                    np.mean(
                        platform_histories[agent.name][fault_event_steps[agent.name][1] :]
                    )
                )
                for agent in agents
            }
            event_recall, false_inflations, mean_detection_delay = _score_inflation_events(
                fault_event_steps,
                trial.effectiveness_change_steps_by_agent,
                event_detection_window_steps,
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
                    "fault_event_steps": fault_event_steps,
                    "hidden_fault_schedule": hidden_schedule,
                    "tracking_rmse_history": global_history,
                    "platform_fault_rmse": platform_fault_rmse,
                    "platform_recovery_rmse": platform_recovery_rmse,
                    "fault_interval_rmse": float(
                        np.mean(list(platform_fault_rmse.values()))
                    ),
                    "recovery_rmse": float(
                        np.mean(list(platform_recovery_rmse.values()))
                    ),
                    "final_effectiveness_rmse": global_history[-1],
                    "change_steps_by_agent": trial.effectiveness_change_steps_by_agent,
                    "valid": trial.valid,
                    "all_goals_reached": trial.result.all_goals_reached,
                    "first_goal_steps": trial.result.first_goal_steps,
                    "final_goal_errors": trial.result.final_goal_errors,
                    "final_orientation_errors": (
                        trial.result.final_orientation_errors
                    ),
                    "min_pairwise_distance": trial.result.min_pairwise_distance,
                    "pairwise_violation_count": trial.result.pairwise_violation_count,
                    "max_collision_slack": trial.max_collision_slack,
                    "solver_fallbacks": trial.solver_fallbacks,
                    "solver_fallbacks_by_agent": trial.solver_fallbacks_by_agent,
                    "solver_failure_steps_by_agent": (
                        trial.solver_failure_steps_by_agent
                    ),
                    "solver_failure_status_counts": (
                        trial.solver_failure_status_counts
                    ),
                    "sustained_completion_cost": sustained_cost,
                    "event_recall": event_recall,
                    "false_inflations": false_inflations,
                    "mean_detection_delay": mean_detection_delay,
                    "probe_count": sum(trial.probe_count_by_agent.values()),
                    "probe_count_by_agent": trial.probe_count_by_agent,
                }
            )
    return records


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
    records = build_time_varying_fault_records(
        cases, agents, simulation, experiment.event_detection_window_steps
    )
    fault_event_steps = records[0]["fault_event_steps"]
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
