"""Controlled communication ablations for heterogeneous harbor coordination."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .communication import LinkConfig
from .simulation import HarborAgent, HarborSimulationConfig, run_harbor_simulation


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
