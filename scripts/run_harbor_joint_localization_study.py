"""Evaluate estimated-state distributed MPC under current and actuator faults."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.harbor import (
    DEFAULT_HARBOR_CONFIG,
    HarborRangeLocalization,
    load_harbor_config,
    load_harbor_fault_study_config,
    load_harbor_observation_noise_config,
    load_harbor_temporary_fault_ensemble_config,
    load_harbor_time_varying_fault_config,
    load_range_aided_slam_config,
    run_harbor_simulation,
)
from scripts.harbor.experiments import generate_temporary_fault_ensemble
from scripts.harbor.learning import run_distributed_harbor_lmpc
from scripts.harbor.mpc import DistributedHarborMPC, load_harbor_mpc_config
from scripts.harbor.plotting import save_harbor_animation


POLICIES = (
    "Direct position sensor",
    "Dead reckoning",
    "Known harbor map",
    "Joint landmark SLAM",
    "Robust fixed-lag SLAM",
    "Joint SLAM + belief retry",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_HARBOR_CONFIG))
    parser.add_argument("--artifact-dir", default="results/latest/harbor")
    parser.add_argument(
        "--case-limit", type=int, default=None, help="run only the first N YAML cases"
    )
    parser.add_argument("--case-seed", type=int, default=None)
    parser.add_argument("--policy", action="append", choices=POLICIES)
    parser.add_argument("--make-gif", action="store_true")
    parser.add_argument("--gif-only", action="store_true")
    return parser.parse_args()


def _adaptive_config(path: str):
    base = load_harbor_mpc_config(path)
    fault = load_harbor_fault_study_config(path)
    _, experiment = load_harbor_time_varying_fault_config(path)
    return replace(
        base,
        prediction_horizon=fault.prediction_horizon,
        terminal_goal_weight=fault.terminal_goal_weight,
        terminal_slack_bound=fault.terminal_slack_bound,
        terminal_slack_weight=fault.terminal_slack_weight,
        residual_adaptation=True,
        residual_adaptation_kinds=("usv", "rov"),
        residual_measurement_source="kinematic_velocity",
        residual_estimator_mode="constant_bias_rls",
        control_effectiveness_adaptation=True,
        effectiveness_estimator_mode="recursive_diagonal",
        effectiveness_rls_adaptive_covariance=True,
        effectiveness_rls_change_detector="threshold",
        effectiveness_rls_change_threshold=experiment.change_threshold,
        effectiveness_rls_covariance_inflation=experiment.covariance_inflation,
        effectiveness_rls_change_warmup_steps=experiment.change_warmup_steps,
        effectiveness_rls_change_cooldown_steps=experiment.change_cooldown_steps,
        effectiveness_recovery_prior_gain=experiment.recovery_prior_gain,
        effectiveness_recovery_prior_mode="transient",
        dynamic_state_slack_bound=0.0,
        dynamic_state_slack_retry_bound=0.0,
    )


def _localization(policy: str, config, seed: int):
    configured = replace(config, seed=seed)
    if policy == "Direct position sensor":
        return None
    if policy == "Dead reckoning":
        return HarborRangeLocalization(
            replace(configured, enabled=False, mode="known_anchor_ekf")
        )
    if policy == "Known harbor map":
        return HarborRangeLocalization(
            replace(configured, enabled=True, mode="known_anchor_ekf")
        )
    if policy == "Robust fixed-lag SLAM":
        return HarborRangeLocalization(
            replace(configured, enabled=True, mode="fixed_lag_slam")
        )
    return HarborRangeLocalization(
        replace(configured, enabled=True, mode="joint_landmark_ekf")
    )


def _policy_controller_config(config, policy: str):
    if policy != "Joint SLAM + belief retry":
        return config
    return replace(config, dynamic_state_slack_retry_bound=0.02)


def _position_rmse(result, agents) -> dict[str, float]:
    values = {}
    for agent in agents:
        estimates = np.asarray(
            [agent.model.position(state) for state in result.observed_states[agent.name]]
        )
        error = estimates - result.positions[agent.name]
        values[agent.name] = float(np.sqrt(np.mean(np.sum(error * error, axis=1))))
    return values


def _current_rmse(controller, agents, disturbance) -> dict[str, float]:
    values = {}
    for agent in agents:
        if agent.model.kind == "ugv":
            continue
        history = np.asarray(controller.residual_history[agent.name], dtype=float)
        truth = disturbance.current(agent.model)
        values[agent.name] = float(
            np.sqrt(np.mean(np.sum((history - truth) ** 2, axis=1)))
        )
    return values


def _observability(localization, agents) -> dict[str, dict] | None:
    if localization is None:
        return None
    return {
        agent.name: {
            "observable_fraction": float(
                np.mean([report.observable for report in localization.reports[agent.name]])
            ),
            "final_rank": localization.reports[agent.name][-1].rank,
            "final_state_dimension": localization.reports[agent.name][
                -1
            ].state_dimension,
            "pose_observable": localization.estimators[agent.name].pose_observable(),
            "deferred_landmark_updates": localization.estimators[
                agent.name
            ].deferred_landmark_updates,
            "landmark_only_updates": localization.estimators[
                agent.name
            ].landmark_only_updates,
        }
        for agent in agents
    }


def _save_plot(records: list[dict], path: Path) -> None:
    seeds = sorted({record["seed"] for record in records})
    by_key = {(record["seed"], record["policy"]): record for record in records}
    colors = ("#667085", "#c8553d", "#1976b9", "#2a9d8f", "#8e5ea2")
    x = np.arange(len(seeds))
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), sharex=True)
    present = tuple(policy for policy in POLICIES if any(r["policy"] == policy for r in records))
    for policy, color in zip(present, colors):
        selected = [by_key[(seed, policy)] for seed in seeds]
        axes[0, 0].plot(
            x,
            [record["all_goals_reached"] for record in selected],
            "o-",
            color=color,
            label=policy,
        )
        axes[0, 1].plot(
            x,
            [record["solver_fallbacks"] for record in selected],
            "o-",
            color=color,
            label=policy,
        )
        axes[1, 0].plot(
            x,
            [record["position_rmse"]["underwater_rov"] for record in selected],
            "o-",
            color=color,
            label=policy,
        )
        axes[1, 1].plot(
            x,
            [record["current_rmse"]["surface_vessel"] for record in selected],
            "o-",
            color=color,
            label=policy,
        )
    axes[0, 0].set_title("True Sustained Task Completion")
    axes[0, 0].set_yticks((0, 1), ("no", "yes"))
    axes[0, 1].set_title("Distributed MPC Fallbacks")
    axes[1, 0].set_title("BlueROV2 Position-Estimate RMSE")
    axes[1, 0].set_ylabel("RMSE [m]")
    axes[1, 1].set_title("Heron Current-Estimate RMSE")
    axes[1, 1].set_ylabel("RMSE [m/s]")
    for axis in axes.flat:
        axis.set_xticks(x, [str(seed) for seed in seeds])
        axis.set_xlabel("matched joint-uncertainty case")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle(
        "Estimated-State Distributed MPC Under Localization, Current, and Actuator Faults",
        fontsize=15,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarize(records: list[dict]) -> dict:
    """Aggregate task, safety, solver, and estimation evidence by policy."""
    summary = {}
    for policy in POLICIES:
        selected = [record for record in records if record["policy"] == policy]
        if not selected:
            continue
        summary[policy] = {
            "trials": len(selected),
            "completion_rate": float(
                np.mean([record["all_goals_reached"] for record in selected])
            ),
            "safety_rate": float(
                np.mean([record["pairwise_violation_count"] == 0 for record in selected])
            ),
            "fallback_free_rate": float(
                np.mean([record["solver_fallbacks"] == 0 for record in selected])
            ),
            "total_solver_fallbacks": int(
                sum(record["solver_fallbacks"] for record in selected)
            ),
            "mean_completion_step_sum": float(
                np.mean([record["completion_step_sum"] for record in selected])
            ),
            "mean_bluerov2_position_rmse": float(
                np.mean(
                    [record["position_rmse"]["underwater_rov"] for record in selected]
                )
            ),
            "mean_heron_current_rmse": float(
                np.mean(
                    [record["current_rmse"]["surface_vessel"] for record in selected]
                )
            ),
            "total_dynamic_state_retries": int(
                sum(record["dynamic_state_retries"] for record in selected)
            ),
            "maximum_dynamic_state_slack": float(
                max(record["max_dynamic_state_slack"] for record in selected)
            ),
        }
    return summary


def main() -> None:
    """Run matched joint uncertainty cases without exposing plant truth."""
    args = parse_args()
    agents, simulation, communication = load_harbor_config(args.config)
    controller_config = _adaptive_config(args.config)
    base_disturbance, _ = load_harbor_time_varying_fault_config(args.config)
    ensemble = load_harbor_temporary_fault_ensemble_config(
        args.config, "joint_localization_development"
    )
    generated = generate_temporary_fault_ensemble(agents, base_disturbance, ensemble)
    if args.case_seed is not None:
        generated = [case for case in generated if case[0] == args.case_seed]
        if not generated:
            raise ValueError("case-seed is not present in the configured ensemble")
    if args.case_limit is not None:
        if args.case_limit <= 0:
            raise ValueError("case-limit must be positive")
        generated = generated[: args.case_limit]
    seed_iterations = run_distributed_harbor_lmpc(
        agents,
        simulation,
        communication,
        replace(
            controller_config,
            learning_iterations=1,
            residual_adaptation=False,
            control_effectiveness_adaptation=False,
        ),
    )
    seed = min(
        (record for record in seed_iterations if record.admitted),
        key=lambda record: record.completion_step_sum,
    )
    observation_base = load_harbor_observation_noise_config(args.config)
    localization_base = load_range_aided_slam_config(args.config)
    records = []
    selected_policies = tuple(args.policy or POLICIES)
    gif_result = None
    for case_seed, observation_seed, disturbance in generated:
        evaluation = replace(
            simulation,
            guidance_update_interval_steps=controller_config.replan_interval_steps,
            goal_hold_steps=disturbance.evaluation_hold_steps,
        )
        for policy in selected_policies:
            localization = _localization(
                policy, localization_base, observation_seed
            )
            controller = DistributedHarborMPC(
                agents=agents,
                config=_policy_controller_config(controller_config, policy),
                dt=simulation.dt,
                safe_states=seed.result.states,
                safe_controls=seed.result.controls,
                learning=False,
            )
            result = run_harbor_simulation(
                agents,
                evaluation,
                communication,
                control_provider=controller,
                disturbance=disturbance,
                observation_noise=replace(
                    observation_base, enabled=True, seed=observation_seed
                ),
                localization_provider=localization,
            )
            record = {
                "seed": case_seed,
                "observation_seed": observation_seed,
                "policy": policy,
                "all_goals_reached": result.all_goals_reached,
                "pairwise_violation_count": result.pairwise_violation_count,
                "minimum_pairwise_distance": result.min_pairwise_distance,
                "completion_step_sum": sum(
                    step if step is not None else simulation.horizon + 1
                    for step in result.first_goal_steps.values()
                ),
                "final_goal_errors": result.final_goal_errors,
                "final_orientation_errors": result.final_orientation_errors,
                "solver_fallbacks": controller.fallback_count,
                "max_collision_slack": controller.max_collision_slack,
                "max_dynamic_state_slack": controller.max_dynamic_state_slack,
                "dynamic_state_retries": sum(
                    controller.dynamic_state_retry_count_by_agent.values()
                ),
                "position_rmse": _position_rmse(result, agents),
                "current_rmse": _current_rmse(controller, agents, disturbance),
                "observability": _observability(localization, agents),
            }
            records.append(record)
            if policy == "Joint SLAM + belief retry" and gif_result is None:
                gif_result = result
            print(
                f"case {case_seed} {policy}: complete={result.all_goals_reached}, "
                f"safe={result.pairwise_violation_count == 0}, "
                f"fallbacks={controller.fallback_count}",
                flush=True,
            )
    output = Path(args.artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.make_gif or args.gif_only:
        if gif_result is None:
            raise ValueError("GIF generation requires the belief-retry policy")
        gif_path = output / "joint_localization_candidate.gif"
        save_harbor_animation(
            gif_result,
            agents,
            simulation,
            gif_path,
            label="Joint SLAM + belief-feasibility retry",
        )
        print(f"Saved joint localization animation: {gif_path}")
    if args.gif_only:
        return
    data_path = output / "joint_localization_development.json"
    plot_path = output / "joint_localization_development.png"
    payload = {
        "ensemble": asdict(ensemble),
        "case_limit": args.case_limit,
        "controller_uses_plant_truth": False,
        "summary": summarize(records),
        "records": records,
    }
    data_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _save_plot(records, plot_path)
    print(f"Saved joint localization data: {data_path}")
    print(f"Saved joint localization plot: {plot_path}")


if __name__ == "__main__":
    main()
