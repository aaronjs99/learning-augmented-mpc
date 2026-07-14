"""Reusable priority and warm-start policies for decentralized LMPC."""

from __future__ import annotations

import numpy as np

from scripts.mpc.manta_lmpc import MantaLMPCConfig


SafeSets = dict[int, list[np.ndarray] | np.ndarray]


def priority_margins(
    agent_i: int,
    agent_j: int,
    safe_sets: SafeSets,
    step: int,
    current_states: dict[int, np.ndarray],
    goals: np.ndarray,
    config: MantaLMPCConfig,
) -> tuple[float, float]:
    """Allocate a fixed pairwise margin budget according to right-of-way."""
    base_margin = config.hyperplane_safety_margin
    if not config.priority_hyperplanes:
        return base_margin, base_margin

    scale = config.priority_margin_scale
    score_i = priority_score(agent_i, safe_sets, step, current_states, goals, config)
    score_j = priority_score(agent_j, safe_sets, step, current_states, goals, config)
    priority_delta = (score_i - score_j) / (abs(score_i) + abs(score_j) + 1e-9)
    return (
        base_margin * (1.0 - scale * priority_delta),
        base_margin * (1.0 + scale * priority_delta),
    )


def priority_score(
    agent: int,
    safe_sets: SafeSets,
    step: int,
    current_states: dict[int, np.ndarray],
    goals: np.ndarray,
    config: MantaLMPCConfig,
) -> float:
    """Return a larger score for agents that should receive right-of-way."""
    current_distance = float(
        np.linalg.norm(current_states[agent][:2] - goals[agent, :2])
    )
    if current_distance <= config.goal_tolerance:
        return 1e6
    if config.priority_metric == "goal_distance":
        return 1.0 / (current_distance + 1e-3)
    if config.priority_metric == "remaining_safe_time":
        states = np.asarray(safe_sets[agent], dtype=float)
        distances = np.linalg.norm(states[:, :2] - goals[agent, :2], axis=1)
        reached = np.flatnonzero(distances <= config.goal_tolerance)
        first_hit = int(reached[0]) if len(reached) else len(states) - 1
        return float(max(first_hit - step, 0))
    raise AssertionError(f"unvalidated priority metric: {config.priority_metric}")


def warm_start_from_safe_set(
    safe_set: list[np.ndarray] | np.ndarray,
    safe_controls: list[np.ndarray] | np.ndarray,
    step: int,
    config: MantaLMPCConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build horizon-sized state and control guesses from learned memory."""
    states = np.asarray(safe_set, dtype=float)
    controls = np.asarray(safe_controls, dtype=float)
    blend = config.warm_start_control_blend
    warm_states = np.empty((states.shape[1], config.prediction_horizon + 1))
    warm_controls = np.full(
        (2, config.prediction_horizon), config.warm_start_control, dtype=float
    )
    for horizon_step in range(config.prediction_horizon + 1):
        index = min(step + horizon_step, len(states) - 1)
        warm_states[:, horizon_step] = states[index]
    for horizon_step in range(config.prediction_horizon):
        if len(controls) == 0:
            break
        index = min(step + horizon_step, len(controls) - 1)
        warm_controls[:, horizon_step] = (
            (1.0 - blend) * warm_controls[:, horizon_step]
            + blend * controls[index]
        )
    return warm_states, warm_controls
