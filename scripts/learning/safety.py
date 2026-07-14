"""Execution-time swept safety filter for proposed multi-agent transitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.dynamics import MantaDynamicsConfig, rk4_step_np
from scripts.metrics import segment_pairwise_distances, segment_point_distances
from scripts.mpc import MantaLMPCConfig
from scripts.simulation import Scenario, StaticObstacle

from .apf import APFConfig
from .recovery import safe_fallback_apf_step


@dataclass(frozen=True)
class SafetyFilterResult:
    """Controls, states, and statuses replaced by the safety filter."""

    controls: dict[int, np.ndarray]
    next_states: dict[int, np.ndarray]
    statuses: dict[int, str]


def filter_unsafe_transitions(
    *,
    current_states: dict[int, np.ndarray],
    proposed_controls: dict[int, np.ndarray],
    proposed_next_states: dict[int, np.ndarray],
    goals: np.ndarray,
    scenario: Scenario,
    config: MantaLMPCConfig,
    apf_config: APFConfig,
    dynamics_config: MantaDynamicsConfig,
    protected_agents: set[int] | None = None,
) -> SafetyFilterResult:
    """Replace unsafe transitions with swept-safe APF or hold actions."""
    controls = _copy_by_agent(proposed_controls)
    next_states = _copy_by_agent(proposed_next_states)
    statuses: dict[int, str] = {}
    protected = protected_agents or set()

    unsafe = unsafe_transition_agents(
        current_states,
        next_states,
        scenario=scenario,
        clearance_buffer=config.safety_filter_buffer,
    )
    filtered_obstacle = StaticObstacle(
        center=scenario.obstacle.center,
        radius=scenario.obstacle.radius + config.safety_filter_buffer,
        physical_radius=scenario.obstacle.physical_radius,
    )
    for agent in sorted(unsafe):
        if agent in protected:
            control = np.zeros(2, dtype=float)
            controls[agent] = control
            next_states[agent] = rk4_step_np(
                current_states[agent], control, config.dt, dynamics_config
            )
            statuses[agent] = "safety_filter_hold"
            continue
        extra_obstacles = [
            StaticObstacle(
                center=tuple(np.asarray(state, dtype=float)[:2]),
                radius=scenario.safety_distance + config.safety_filter_buffer,
            )
            for other, state in current_states.items()
            if other != agent
        ]
        control, next_state = safe_fallback_apf_step(
            current_state=current_states[agent],
            goal_state=goals[agent],
            obstacle=filtered_obstacle,
            extra_obstacles=extra_obstacles,
            apf_config=apf_config,
            dt=config.dt,
            config=config,
            dynamics_config=dynamics_config,
        )
        controls[agent] = control
        next_states[agent] = next_state
        statuses[agent] = "safety_filter_apf"

    remaining = unsafe_transition_agents(
        current_states,
        next_states,
        scenario=scenario,
        clearance_buffer=config.safety_filter_buffer,
    )
    for agent in sorted(remaining):
        control = np.zeros(2, dtype=float)
        controls[agent] = control
        next_states[agent] = rk4_step_np(
            current_states[agent], control, config.dt, dynamics_config
        )
        statuses[agent] = "safety_filter_hold"

    unresolved = unsafe_transition_agents(
        current_states,
        next_states,
        scenario=scenario,
        clearance_buffer=config.safety_filter_buffer,
    )
    if unresolved:
        raise RuntimeError(
            f"safety filter could not produce safe transitions for agents {sorted(unresolved)}"
        )
    return SafetyFilterResult(controls, next_states, statuses)


def unsafe_transition_agents(
    current_states: dict[int, np.ndarray],
    next_states: dict[int, np.ndarray],
    *,
    scenario: Scenario,
    clearance_buffer: float = 0.0,
) -> set[int]:
    """Return agents involved in a swept obstacle or pairwise violation."""
    agents = sorted(current_states)
    if set(agents) != set(next_states):
        raise ValueError("current and next states must contain the same agents")
    if clearance_buffer < 0.0:
        raise ValueError("clearance_buffer must be nonnegative")

    unsafe: set[int] = set()
    obstacle = scenario.obstacle
    for agent in agents:
        transition = np.vstack((current_states[agent], next_states[agent]))
        clearance = segment_point_distances(transition, obstacle.center)[0]
        if clearance < obstacle.radius + clearance_buffer:
            unsafe.add(agent)

    for index, agent_i in enumerate(agents):
        transition_i = np.vstack(
            (current_states[agent_i], next_states[agent_i])
        )
        for agent_j in agents[index + 1 :]:
            transition_j = np.vstack(
                (current_states[agent_j], next_states[agent_j])
            )
            separation = segment_pairwise_distances(
                transition_i, transition_j
            )[0]
            if separation < scenario.safety_distance + clearance_buffer:
                unsafe.update((agent_i, agent_j))
    return unsafe


def _copy_by_agent(values: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    return {
        agent: np.asarray(value, dtype=float).copy()
        for agent, value in values.items()
    }
