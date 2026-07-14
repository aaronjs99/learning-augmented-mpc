"""Safe-set initialization and terminal sampling utilities for manta LMPC."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np

from .apf import APFConfig, simulate_manta_autopilot_with_controls
from scripts.dynamics import MantaDynamicsConfig, rk4_step_np
from scripts.simulation import Scenario, StaticObstacle


_STATIC_AGENT_RADIUS_SCALES = (1.5, 1.2, 1.0, 0.85, 0.7)


@dataclass(frozen=True)
class _SafeSetScore:
    """Validation metrics used to choose an APF staging order."""

    all_goals_reached: bool
    usable_for_learning: bool
    pairwise_violation_count: int
    obstacle_violation_count: int
    total_first_hit_steps: int
    final_goal_error: float
    min_pairwise_distance: float
    min_obstacle_clearance: float


def hold_trajectory(
    state: np.ndarray,
    steps: int,
    *,
    dt: float,
    dynamics_config: MantaDynamicsConfig,
) -> np.ndarray:
    """Return a dynamically valid zero-control hold trajectory."""
    traj = [np.asarray(state, dtype=float).copy()]
    current = traj[0].copy()
    zero_u = np.zeros(2, dtype=float)

    for _ in range(steps):
        current = rk4_step_np(current, zero_u, dt, dynamics_config)
        traj.append(current.copy())

    return np.asarray(traj, dtype=float)


def hold_controls(steps: int) -> np.ndarray:
    """Return zero controls for a dynamic hold trajectory."""
    return np.zeros((steps, 2), dtype=float)


def build_staggered_safe_sets(
    scenario: Scenario,
    *,
    dt: float,
    apf_config: APFConfig = APFConfig(),
    dynamics_config: MantaDynamicsConfig = MantaDynamicsConfig(),
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Build staged APF safe sets and their corresponding controls."""
    starts = np.asarray(scenario.starts, dtype=float)
    goals = np.asarray(scenario.goals, dtype=float)
    num_agents = len(starts)
    if goals.shape[0] != num_agents:
        raise ValueError("scenario starts and goals must have the same agent count")
    if num_agents < 2:
        raise ValueError("manta LMPC requires at least two agents")

    candidates = []
    for radius_scale in _STATIC_AGENT_RADIUS_SCALES:
        for order in permutations(range(num_agents)):
            safe_sets, safe_controls, staged_trajs = _build_ordered_safe_sets(
                scenario,
                starts,
                goals,
                order,
                radius_scale,
                dt=dt,
                apf_config=apf_config,
                dynamics_config=dynamics_config,
            )
            score = _score_safe_sets(
                safe_sets,
                goals,
                scenario.safety_distance,
                scenario.obstacle.center,
                scenario.obstacle.radius,
                apf_config.goal_tolerance,
            )
            candidates.append((score, safe_sets, safe_controls, staged_trajs))

    _, safe_sets, safe_controls, _ = min(candidates, key=_score_key)
    return safe_sets, safe_controls


def _build_ordered_safe_sets(
    scenario: Scenario,
    starts: np.ndarray,
    goals: np.ndarray,
    order: tuple[int, ...],
    static_agent_radius_scale: float,
    *,
    dt: float,
    apf_config: APFConfig,
    dynamics_config: MantaDynamicsConfig,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Generate staged APF paths while treating waiting agents as obstacles."""
    staged_trajs: dict[int, np.ndarray] = {}
    staged_controls: dict[int, np.ndarray] = {}
    start_steps: dict[int, int] = {}
    elapsed_steps = 0

    for order_index, agent in enumerate(order):
        start_steps[agent] = elapsed_steps
        active_start = hold_trajectory(
            starts[agent],
            elapsed_steps,
            dt=dt,
            dynamics_config=dynamics_config,
        )[-1]

        extra_obstacles: list[StaticObstacle] = []
        for other in order[:order_index]:
            extra_obstacles.append(
                StaticObstacle(
                    center=tuple(staged_trajs[other][-1, :2]),
                    radius=scenario.safety_distance * static_agent_radius_scale,
                )
            )
        for other in order[order_index + 1 :]:
            waiting_state = hold_trajectory(
                starts[other],
                elapsed_steps,
                dt=dt,
                dynamics_config=dynamics_config,
            )[-1]
            extra_obstacles.append(
                StaticObstacle(
                    center=tuple(waiting_state[:2]),
                    radius=scenario.safety_distance * static_agent_radius_scale,
                )
            )

        (
            staged_trajs[agent],
            staged_controls[agent],
        ) = simulate_manta_autopilot_with_controls(
            active_start,
            goals[agent],
            dt=dt,
            obstacle=scenario.obstacle,
            extra_obstacles=extra_obstacles,
            apf_config=apf_config,
            dynamics_config=dynamics_config,
        )
        elapsed_steps += len(staged_trajs[agent]) - 1

    total_steps = elapsed_steps
    safe_sets: dict[int, np.ndarray] = {}
    safe_controls: dict[int, np.ndarray] = {}
    for agent in order:
        prefix_steps = start_steps[agent]
        solo = staged_trajs[agent]
        solo_steps = len(solo) - 1
        suffix_steps = total_steps - prefix_steps - solo_steps

        prefix = hold_trajectory(
            starts[agent],
            prefix_steps,
            dt=dt,
            dynamics_config=dynamics_config,
        )
        suffix = hold_trajectory(
            solo[-1],
            suffix_steps,
            dt=dt,
            dynamics_config=dynamics_config,
        )
        safe_sets[agent] = np.vstack((prefix[:-1], solo, suffix[1:]))
        safe_controls[agent] = np.vstack(
            (
                hold_controls(prefix_steps),
                staged_controls[agent],
                hold_controls(suffix_steps),
            )
        )

    return safe_sets, safe_controls, staged_trajs


def _score_safe_sets(
    safe_sets: dict[int, np.ndarray],
    goals: np.ndarray,
    safety_distance: float,
    obstacle_center: tuple[float, float],
    obstacle_radius: float,
    goal_tolerance: float,
) -> _SafeSetScore:
    """Score a staged APF candidate without importing the LMPC runner."""
    agents = sorted(safe_sets)
    max_len = max(len(safe_sets[agent]) for agent in agents)
    state_dim = np.asarray(safe_sets[agents[0]], dtype=float).shape[1]
    states = np.zeros((max_len, len(agents), state_dim), dtype=float)
    for out_agent, agent in enumerate(agents):
        traj = np.asarray(safe_sets[agent], dtype=float)
        states[: len(traj), out_agent] = traj
        if len(traj) < max_len:
            states[len(traj) :, out_agent] = traj[-1]

    pos = states[:, :, :2]
    goal_pos = np.asarray(goals, dtype=float)[agents, :2]
    goal_errors = np.linalg.norm(pos - goal_pos[None, :, :], axis=2)
    final_goal_errors = goal_errors[-1]
    all_goals_reached = bool(np.all(final_goal_errors <= goal_tolerance))
    first_hit_steps = []
    for agent_idx in range(len(agents)):
        reached = np.where(goal_errors[:, agent_idx] <= goal_tolerance)[0]
        first_hit_steps.append(int(reached[0]) if len(reached) else max_len)

    pairwise = []
    for i in range(pos.shape[1]):
        for j in range(i + 1, pos.shape[1]):
            pairwise.append(np.linalg.norm(pos[:, i] - pos[:, j], axis=1))
    pairwise_distances = np.vstack(pairwise).T
    pairwise_violation_count = int(np.sum(pairwise_distances < safety_distance))
    min_pairwise_distance = float(np.min(pairwise_distances))

    center = np.asarray(obstacle_center, dtype=float)
    obstacle_clearance = np.linalg.norm(pos - center[None, None, :], axis=2)
    obstacle_clearance -= obstacle_radius
    obstacle_violation_count = int(np.sum(obstacle_clearance < 0.0))
    min_obstacle_clearance = float(np.min(obstacle_clearance))

    return _SafeSetScore(
        all_goals_reached=all_goals_reached,
        usable_for_learning=(
            pairwise_violation_count == 0 and obstacle_violation_count == 0
        ),
        pairwise_violation_count=pairwise_violation_count,
        obstacle_violation_count=obstacle_violation_count,
        total_first_hit_steps=sum(first_hit_steps),
        final_goal_error=float(np.sum(final_goal_errors)),
        min_pairwise_distance=min_pairwise_distance,
        min_obstacle_clearance=min_obstacle_clearance,
    )


def _score_key(
    candidate: tuple[
        _SafeSetScore,
        dict[int, np.ndarray],
        dict[int, np.ndarray],
        dict[int, np.ndarray],
    ],
) -> tuple[bool, bool, int, int, int, float, float, float]:
    """Prefer safe, complete, short-error APF seeds with larger margins."""
    score = candidate[0]
    return (
        not score.usable_for_learning,
        not score.all_goals_reached,
        score.pairwise_violation_count,
        score.obstacle_violation_count,
        score.total_first_hit_steps,
        score.final_goal_error,
        -score.min_pairwise_distance,
        -score.min_obstacle_clearance,
    )


def sample_terminal_safe_set(
    safe_set: list[np.ndarray] | np.ndarray,
    target_index: int,
    k_hull: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample future terminal safe states and time-to-go costs for the LMPC hull."""
    states = np.asarray(safe_set, dtype=float)
    safe_states = np.zeros((7, k_hull), dtype=float)
    safe_costs = np.zeros((k_hull, 1), dtype=float)
    traj_len = len(states)

    for k_idx in range(k_hull):
        state_idx = min(target_index + k_idx, traj_len - 1)
        safe_states[:, k_idx] = states[state_idx]
        safe_costs[k_idx, 0] = (traj_len - 1) - state_idx

    return safe_states, safe_costs
