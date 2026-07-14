"""Fallback control and optional terminal recovery for manta LMPC rollouts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from scripts.dynamics import MantaDynamicsConfig, rk4_step_np
from scripts.metrics import segment_point_distances
from scripts.mpc import MantaLMPCConfig
from scripts.simulation import Scenario, StaticObstacle

from .apf import APFConfig, compute_apf_control


@dataclass(frozen=True)
class RecoveryResult:
    """Controls and per-agent statuses appended during terminal recovery."""

    controls: np.ndarray
    statuses: list[dict[int, str]]


def repair_incomplete_with_apf(
    *,
    history: dict[int, list[np.ndarray]],
    current_states: dict[int, np.ndarray],
    safe_sets: dict[int, list[np.ndarray] | np.ndarray],
    reached_agents: set[int],
    goals: np.ndarray,
    scenario: Scenario,
    config: MantaLMPCConfig,
    apf_config: APFConfig,
    dynamics_config: MantaDynamicsConfig,
    should_stop: Callable[[], bool] | None,
    verbose: bool,
) -> RecoveryResult:
    """Append staged APF motion to a safe but incomplete rollout.

    ``history``, ``current_states``, and ``reached_agents`` are updated in place
    so the caller can validate and optionally learn from the repaired rollout.
    """
    num_agents = len(current_states)
    repair_controls: list[np.ndarray] = []
    repair_statuses: list[dict[int, str]] = []
    repair_steps = 0
    zero_control = np.zeros(2, dtype=float)
    unfinished = _repair_order(current_states, goals, reached_agents)

    if verbose and unfinished:
        print(f"  repair: staging APF finish for agents {unfinished}.")

    for active_agent in unfinished:
        while active_agent not in reached_agents and repair_steps < config.repair_max_steps:
            _raise_if_stop_requested(should_stop)
            step_controls = np.zeros((num_agents, 2), dtype=float)
            next_states: dict[int, np.ndarray] = {}
            step_statuses: dict[int, str] = {}

            for agent in range(num_agents):
                if agent == active_agent:
                    extra_obstacles = _agent_static_obstacles(
                        current_states, agent, scenario, config
                    )
                    repair_target = _repair_waypoint_target(
                        current_states[agent], safe_sets[agent], goals[agent], config
                    )
                    control, next_state = safe_fallback_apf_step(
                        current_state=current_states[agent],
                        goal_state=repair_target,
                        obstacle=scenario.obstacle,
                        extra_obstacles=extra_obstacles,
                        apf_config=apf_config,
                        dt=config.dt,
                        config=config,
                        dynamics_config=dynamics_config,
                    )
                    step_statuses[agent] = "repair_apf"
                else:
                    control = zero_control.copy()
                    next_state = rk4_step_np(
                        current_states[agent], control, config.dt, dynamics_config
                    )
                    step_statuses[agent] = "repair_hold"

                step_controls[agent] = control
                next_states[agent] = next_state

            for agent in range(num_agents):
                current_states[agent] = next_states[agent]
                history[agent].append(current_states[agent].copy())
                if (
                    np.linalg.norm(current_states[agent][:2] - goals[agent, :2])
                    <= config.goal_tolerance
                ):
                    reached_agents.add(agent)

            repair_controls.append(step_controls)
            repair_statuses.append(step_statuses)
            repair_steps += 1

        if repair_steps >= config.repair_max_steps:
            break

    if verbose and len(reached_agents) == num_agents:
        print(f"  repair: all agents reached goals after {repair_steps} steps.")
    elif verbose and repair_steps > 0:
        print(
            "  repair: stopped before all agents reached "
            f"({repair_steps}/{config.repair_max_steps} steps)."
        )

    controls = (
        np.asarray(repair_controls, dtype=float)
        if repair_controls
        else np.zeros((0, num_agents, 2), dtype=float)
    )
    return RecoveryResult(controls=controls, statuses=repair_statuses)


def safe_fallback_apf_step(
    *,
    current_state: np.ndarray,
    goal_state: np.ndarray,
    obstacle: StaticObstacle,
    extra_obstacles: list[StaticObstacle] | None = None,
    apf_config: APFConfig,
    dt: float,
    config: MantaLMPCConfig,
    dynamics_config: MantaDynamicsConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose a bounded, one-step APF fallback with maximum feasible clearance."""
    apf_control = compute_apf_control(
        current_state=current_state,
        goal_state=goal_state,
        obstacle=obstacle,
        extra_obstacles=extra_obstacles,
        apf_config=apf_config,
        dynamics_config=dynamics_config,
    )
    candidates = _fallback_control_candidates(apf_control, dynamics_config, config)
    scored_feasible: list[tuple[float, np.ndarray, np.ndarray]] = []
    best_any: tuple[float, np.ndarray, np.ndarray] | None = None

    for control in candidates:
        next_state = rk4_step_np(current_state, control, dt, dynamics_config)
        transition = np.vstack((current_state, next_state))
        clearance = _obstacle_clearance(transition, obstacle)
        extra_clearance = _minimum_extra_clearance(transition, extra_obstacles)
        extra_bonus = extra_clearance if np.isfinite(extra_clearance) else 0.0
        goal_distance = float(np.linalg.norm(next_state[:2] - goal_state[:2]))
        score = (
            goal_distance
            - config.fallback_clearance_weight * clearance
            - config.fallback_extra_clearance_weight * extra_bonus
            + config.fallback_control_weight * float(np.linalg.norm(control))
        )
        if clearance >= 0.0 and extra_clearance >= 0.0:
            scored_feasible.append((score, control, next_state))
        worst_clearance = (
            min(clearance, extra_clearance)
            if np.isfinite(extra_clearance)
            else clearance
        )
        if best_any is None or worst_clearance > best_any[0]:
            best_any = (worst_clearance, control, next_state)

    if scored_feasible:
        _, control, next_state = min(scored_feasible, key=lambda item: item[0])
        return control, next_state
    if best_any is None:
        raise RuntimeError("fallback APF candidate set was empty")
    _, control, next_state = best_any
    return control, next_state


def _repair_order(
    current_states: dict[int, np.ndarray],
    goals: np.ndarray,
    reached_agents: set[int],
) -> list[int]:
    """Return unfinished agents in longest-remaining-distance order."""
    unfinished = [agent for agent in sorted(current_states) if agent not in reached_agents]
    return sorted(
        unfinished,
        key=lambda agent: np.linalg.norm(current_states[agent][:2] - goals[agent, :2]),
        reverse=True,
    )


def _agent_static_obstacles(
    current_states: dict[int, np.ndarray],
    active_agent: int,
    scenario: Scenario,
    config: MantaLMPCConfig,
) -> list[StaticObstacle]:
    """Represent non-active agents as static recovery obstacles."""
    radius = scenario.safety_distance * config.repair_static_agent_scale
    return [
        StaticObstacle(center=tuple(state[:2]), radius=radius)
        for agent, state in current_states.items()
        if agent != active_agent
    ]


def _repair_waypoint_target(
    current_state: np.ndarray,
    safe_set: list[np.ndarray] | np.ndarray,
    goal: np.ndarray,
    config: MantaLMPCConfig,
) -> np.ndarray:
    """Choose a forward recovery waypoint from the stored safe-set route."""
    if np.linalg.norm(current_state[:2] - goal[:2]) <= config.goal_tolerance:
        return np.asarray(goal, dtype=float)

    states = np.asarray(safe_set, dtype=float)
    nearest_idx = int(
        np.argmin(np.linalg.norm(states[:, :2] - current_state[:2], axis=1))
    )
    target_idx = min(nearest_idx + config.repair_waypoint_lookahead, len(states) - 1)
    return states[target_idx]


def _fallback_control_candidates(
    apf_control: np.ndarray,
    dynamics_config: MantaDynamicsConfig,
    config: MantaLMPCConfig,
) -> list[np.ndarray]:
    """Return unique bounded controls considered by the fallback selector."""
    mu_min = dynamics_config.mu_min
    mu_max = dynamics_config.mu_max
    levels = [
        min(mu_max, max(mu_min, level)) for level in config.fallback_control_levels
    ]
    diagonal_levels = [
        min(mu_max, max(mu_min, level))
        for level in config.fallback_diagonal_levels
    ]
    raw = [apf_control, np.zeros(2, dtype=float)]
    for level in levels:
        raw.extend(
            (
                np.array([level, 0.0]),
                np.array([0.0, level]),
            )
        )
    raw.extend(np.array([level, level]) for level in diagonal_levels)

    candidates: list[np.ndarray] = []
    seen: set[tuple[float, float]] = set()
    for control in raw:
        clipped = np.clip(np.asarray(control, dtype=float), mu_min, mu_max)
        key = (round(float(clipped[0]), 6), round(float(clipped[1]), 6))
        if key not in seen:
            seen.add(key)
            candidates.append(clipped)
    return candidates


def _obstacle_clearance(
    trajectory: np.ndarray, obstacle: StaticObstacle
) -> float:
    return float(
        np.min(segment_point_distances(trajectory, obstacle.center))
        - obstacle.radius
    )


def _minimum_extra_clearance(
    trajectory: np.ndarray,
    extra_obstacles: list[StaticObstacle] | None,
) -> float:
    if not extra_obstacles:
        return float("inf")
    return min(
        _obstacle_clearance(trajectory, obstacle)
        for obstacle in extra_obstacles
    )


def _raise_if_stop_requested(should_stop: Callable[[], bool] | None) -> None:
    if should_stop is not None and should_stop():
        raise KeyboardInterrupt
