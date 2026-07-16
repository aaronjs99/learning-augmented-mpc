"""Controlled communication ablations for heterogeneous harbor coordination."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .communication import LinkConfig
from .learning import run_distributed_harbor_lmpc
from .mpc import DistributedHarborMPC, HarborMPCConfig
from .simulation import (
    HarborAgent,
    HarborDisturbanceConfig,
    HarborResult,
    HarborSimulationConfig,
    run_harbor_simulation,
)


@dataclass(frozen=True)
class HarborRobustnessTrial:
    """One matched plant-mismatch trial and its controller telemetry."""

    label: str
    result: HarborResult
    valid: bool
    completion_step_sum: int
    solver_fallbacks: int
    max_collision_slack: float
    residual_history: dict[str, np.ndarray]
    final_residual_estimates: dict[str, np.ndarray]


def sweep_network_robustness(
    agents: list[HarborAgent],
    simulation: HarborSimulationConfig,
    communication: LinkConfig,
    *,
    delays: list[int],
    dropout_probabilities: list[float],
    seeds: list[int],
) -> list[dict[str, float | int]]:
    """Measure safety and completion across delay/dropout network conditions."""
    if not delays or not dropout_probabilities or not seeds:
        raise ValueError("network sweep axes and seeds must not be empty")
    records = []
    for delay in delays:
        for dropout in dropout_probabilities:
            results = [
                run_harbor_simulation(
                    agents,
                    simulation,
                    replace(
                        communication,
                        enabled=True,
                        delay_steps=delay,
                        dropout_probability=dropout,
                        seed=seed,
                    ),
                )
                for seed in seeds
            ]
            completion_costs = [
                sum(
                    (
                        result.first_goal_steps[name]
                        if result.first_goal_steps[name] is not None
                        and result.final_goal_errors[name]
                        <= simulation.goal_tolerance
                        else simulation.horizon + 1
                    )
                    for name in result.first_goal_steps
                )
                for result in results
            ]
            delivery_ratios = [
                result.messages_delivered / max(result.messages_sent, 1)
                for result in results
            ]
            records.append(
                {
                    "delay_steps": delay,
                    "dropout_probability": dropout,
                    "trials": len(results),
                    "safe_rate": float(
                        np.mean(
                            [result.pairwise_violation_count == 0 for result in results]
                        )
                    ),
                    "completion_rate": float(
                        np.mean([result.all_goals_reached for result in results])
                    ),
                    "mean_completion_step_sum": float(np.mean(completion_costs)),
                    "worst_min_pairwise_distance": float(
                        min(result.min_pairwise_distance for result in results)
                    ),
                    "mean_delivery_ratio": float(np.mean(delivery_ratios)),
                }
            )
    return records


def sweep_prediction_horizons(
    agents: list[HarborAgent],
    simulation: HarborSimulationConfig,
    communication: LinkConfig,
    mpc_config: HarborMPCConfig,
    *,
    horizons: list[int],
) -> list[dict[str, float | int | str | bool]]:
    """Compare matched distributed MPC and LMPC across horizon lengths."""
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("prediction horizons must be positive")
    records = []
    for horizon in horizons:
        iterations = run_distributed_harbor_lmpc(
            agents,
            simulation,
            communication,
            replace(
                mpc_config,
                prediction_horizon=horizon,
                learning_iterations=1,
            ),
        )
        for iteration in iterations[1:]:
            records.append(
                {
                    "prediction_horizon": horizon,
                    "controller": (
                        "MPC"
                        if iteration.label == "distributed_mpc"
                        else "LMPC"
                    ),
                    "complete": iteration.result.all_goals_reached,
                    "admitted": iteration.admitted,
                    "completion_step_sum": iteration.completion_step_sum,
                    "solve_time_seconds": iteration.solve_time_seconds,
                    "solver_calls": iteration.solver_calls,
                    "mean_solve_time_ms": (
                        1000.0
                        * iteration.solve_time_seconds
                        / max(iteration.solver_calls, 1)
                    ),
                    "solver_fallbacks": iteration.solver_fallbacks,
                    "pairwise_violation_count": (
                        iteration.result.pairwise_violation_count
                    ),
                    "min_pairwise_distance": (
                        iteration.result.min_pairwise_distance
                    ),
                    "max_terminal_slack": iteration.max_terminal_slack,
                }
            )
    return records


def run_model_mismatch_study(
    agents: list[HarborAgent],
    simulation: HarborSimulationConfig,
    communication: LinkConfig,
    mpc_config: HarborMPCConfig,
    disturbance: HarborDisturbanceConfig,
) -> list[HarborRobustnessTrial]:
    """Compare nominal and residual-adaptive controllers on one hidden plant."""
    seed_iterations = run_distributed_harbor_lmpc(
        agents,
        simulation,
        communication,
        replace(mpc_config, learning_iterations=1, residual_adaptation=False),
    )
    seed = min(
        (record for record in seed_iterations if record.admitted),
        key=lambda record: record.completion_step_sum,
    )
    evaluation = replace(
        simulation,
        guidance_update_interval_steps=mpc_config.replan_interval_steps,
        goal_hold_steps=disturbance.evaluation_hold_steps,
    )
    definitions = (
        ("Nominal MPC", False, False),
        ("Residual-adaptive MPC", True, False),
        ("Residual-adaptive LMPC", True, True),
    )
    trials = []
    for label, adaptive, learning in definitions:
        controller = DistributedHarborMPC(
            agents=agents,
            config=replace(mpc_config, residual_adaptation=adaptive),
            dt=simulation.dt,
            safe_states=seed.result.states,
            safe_controls=seed.result.controls,
            learning=learning,
        )
        result = run_harbor_simulation(
            agents,
            evaluation,
            communication,
            control_provider=controller,
            disturbance=disturbance,
        )
        completion_cost = sum(
            step if step is not None else simulation.horizon + 1
            for step in result.first_goal_steps.values()
        )
        valid = (
            result.all_goals_reached
            and result.pairwise_violation_count == 0
            and controller.fallback_count == 0
            and controller.max_collision_slack <= 1e-9
        )
        trials.append(
            HarborRobustnessTrial(
                label=label,
                result=result,
                valid=valid,
                completion_step_sum=completion_cost,
                solver_fallbacks=controller.fallback_count,
                max_collision_slack=controller.max_collision_slack,
                residual_history={
                    name: np.asarray(values, dtype=float)
                    for name, values in controller.residual_history.items()
                },
                final_residual_estimates={
                    name: value.copy()
                    for name, value in controller.position_drift_estimates.items()
                },
            )
        )
        print(
            f"{label}: complete={result.all_goals_reached}, "
            f"safe={result.pairwise_violation_count == 0}, "
            f"cost={completion_cost}, fallbacks={controller.fallback_count}",
            flush=True,
        )
    return trials
