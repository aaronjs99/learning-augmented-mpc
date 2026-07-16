"""Safe-set initialization and terminal sampling utilities for manta LMPC."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np

from .apf import APFConfig, simulate_manta_autopilot_with_controls
from scripts.dynamics import MantaDynamicsConfig, rk4_step_np
from scripts.metrics import history_to_tensor, validate_trajectory
from scripts.metrics.geometry import swept_pairwise_distances
from scripts.simulation import Scenario, StaticObstacle


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
    for radius_scale in apf_config.static_agent_radius_scales:
        for order in permutations(range(num_agents)):
            (
                safe_sets,
                safe_controls,
                staged_trajs,
                staged_controls,
            ) = _build_ordered_safe_sets(
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
            candidates.append(
                (score, safe_sets, safe_controls, staged_trajs, staged_controls)
            )

    candidates.sort(key=_score_key)
    if (
        apf_config.compact_staging
        and num_agents <= apf_config.compact_staging_max_agents
    ):
        compact_candidates = []
        for candidate in candidates[: apf_config.compact_staging_candidates]:
            score, _, _, staged_trajs, staged_controls = candidate
            if not score.usable_for_learning:
                continue
            compact = compact_staged_safe_sets(
                starts,
                staged_trajs,
                staged_controls,
                scenario.safety_distance,
                dt=dt,
                dynamics_config=dynamics_config,
            )
            if compact is None:
                continue
            compact_sets, compact_controls = compact
            compact_score = _score_safe_sets(
                compact_sets,
                goals,
                scenario.safety_distance,
                scenario.obstacle.center,
                scenario.obstacle.radius,
                apf_config.goal_tolerance,
            )
            compact_candidates.append(
                (
                    compact_score,
                    compact_sets,
                    compact_controls,
                    staged_trajs,
                    staged_controls,
                )
            )
        candidates.extend(compact_candidates)

    _, safe_sets, safe_controls, _, _ = min(candidates, key=_score_key)
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
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
]:
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

    return safe_sets, safe_controls, staged_trajs, staged_controls


def compact_staged_safe_sets(
    starts: np.ndarray,
    staged_trajs: dict[int, np.ndarray],
    staged_controls: dict[int, np.ndarray],
    safety_distance: float,
    *,
    dt: float,
    dynamics_config: MantaDynamicsConfig,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]] | None:
    """Overlap staged routes using the shortest pairwise-safe start delays.

    The APF route geometry is unchanged. Controls are replayed after each delay,
    which preserves the full nonlinear state evolution instead of shifting only
    the plotted positions.
    """
    agents = sorted(staged_trajs)
    if agents != list(range(len(agents))) or len(agents) not in (2, 3):
        return None
    if set(staged_controls) != set(agents):
        raise ValueError("staged trajectories and controls must share agent keys")

    delays = _find_compact_start_delays(staged_trajs, safety_distance)
    if delays is None:
        return None

    total_steps = max(
        delays[agent] + len(staged_controls[agent]) for agent in agents
    )
    safe_sets: dict[int, np.ndarray] = {}
    safe_controls: dict[int, np.ndarray] = {}
    for agent in agents:
        prefix_steps = delays[agent]
        active_controls = np.asarray(staged_controls[agent], dtype=float)
        suffix_steps = total_steps - prefix_steps - len(active_controls)
        prefix = hold_trajectory(
            starts[agent],
            prefix_steps,
            dt=dt,
            dynamics_config=dynamics_config,
        )
        active = _replay_controls(
            prefix[-1], active_controls, dt=dt, dynamics_config=dynamics_config
        )
        suffix = hold_trajectory(
            active[-1],
            suffix_steps,
            dt=dt,
            dynamics_config=dynamics_config,
        )
        safe_sets[agent] = np.vstack((prefix[:-1], active, suffix[1:]))
        safe_controls[agent] = np.vstack(
            (
                hold_controls(prefix_steps),
                active_controls,
                hold_controls(suffix_steps),
            )
        )
    return safe_sets, safe_controls


def _find_compact_start_delays(
    staged_trajs: dict[int, np.ndarray], safety_distance: float
) -> dict[int, int] | None:
    """Return minimum-makespan start delays for two or three fixed routes."""
    agents = sorted(staged_trajs)
    if agents != list(range(len(agents))) or len(agents) not in (2, 3):
        return None
    durations = {agent: len(staged_trajs[agent]) - 1 for agent in agents}
    max_delay = sum(durations.values())
    relative_safe = {
        (first, second): _safe_relative_delays(
            staged_trajs[first][:, :2],
            staged_trajs[second][:, :2],
            max_delay,
            safety_distance,
        )
        for first in agents
        for second in agents
        if first < second
    }

    best: tuple[tuple[int, int, tuple[int, ...]], dict[int, int]] | None = None
    for anchor in agents:
        others = [agent for agent in agents if agent != anchor]
        delay_ranges = [range(max_delay + 1) for _ in others]
        if len(others) == 1:
            delay_pairs = ((delay,) for delay in delay_ranges[0])
        else:
            delay_pairs = (
                (first_delay, second_delay)
                for first_delay in delay_ranges[0]
                for second_delay in delay_ranges[1]
            )
        for values in delay_pairs:
            delays = {anchor: 0, **dict(zip(others, values, strict=True))}
            if not all(
                relative_safe[(first, second)][
                    delays[second] - delays[first] + max_delay
                ]
                for first in agents
                for second in agents
                if first < second
            ):
                continue
            makespan = max(delays[a] + durations[a] for a in agents)
            key = (
                makespan,
                sum(delays.values()),
                tuple(delays[a] for a in agents),
            )
            if best is None or key < best[0]:
                best = (key, delays)
    return None if best is None else best[1]


def _safe_relative_delays(
    first: np.ndarray,
    second: np.ndarray,
    max_delay: int,
    safety_distance: float,
) -> np.ndarray:
    """Return safety flags indexed by relative delay plus ``max_delay``."""
    flags = np.zeros(2 * max_delay + 1, dtype=bool)
    first_steps = len(first) - 1
    second_steps = len(second) - 1
    for relative_delay in range(-max_delay, max_delay + 1):
        first_delay = max(0, -relative_delay)
        second_delay = max(0, relative_delay)
        total_steps = max(
            first_delay + first_steps, second_delay + second_steps
        )
        pair = np.stack(
            (
                _shift_positions(first, first_delay, total_steps),
                _shift_positions(second, second_delay, total_steps),
            ),
            axis=1,
        )
        flags[relative_delay + max_delay] = bool(
            np.min(swept_pairwise_distances(pair)) >= safety_distance
        )
    return flags


def _shift_positions(
    positions: np.ndarray, delay: int, total_steps: int
) -> np.ndarray:
    """Delay a fixed path and hold its endpoints to ``total_steps``."""
    values = np.asarray(positions, dtype=float)
    steps = len(values) - 1
    shifted = np.empty((total_steps + 1, 2), dtype=float)
    shifted[: delay + 1] = values[0]
    shifted[delay : delay + steps + 1] = values
    shifted[delay + steps :] = values[-1]
    return shifted


def _replay_controls(
    start: np.ndarray,
    controls: np.ndarray,
    *,
    dt: float,
    dynamics_config: MantaDynamicsConfig,
) -> np.ndarray:
    """Integrate a fixed control sequence from a supplied full state."""
    history = [np.asarray(start, dtype=float).copy()]
    for control in np.asarray(controls, dtype=float):
        history.append(rk4_step_np(history[-1], control, dt, dynamics_config))
    return np.asarray(history, dtype=float)


def _score_safe_sets(
    safe_sets: dict[int, np.ndarray],
    goals: np.ndarray,
    safety_distance: float,
    obstacle_center: tuple[float, float],
    obstacle_radius: float,
    goal_tolerance: float,
) -> _SafeSetScore:
    """Score a staged APF candidate with the shared admission validator."""
    agents, states = history_to_tensor(safe_sets)
    pos = states[:, :, :2]
    goal_pos = np.asarray(goals, dtype=float)[agents, :2]
    goal_errors = np.linalg.norm(pos - goal_pos[None, :, :], axis=2)
    final_goal_errors = goal_errors[-1]
    first_hit_steps = []
    for agent_idx in range(len(agents)):
        reached = np.where(goal_errors[:, agent_idx] <= goal_tolerance)[0]
        first_hit_steps.append(int(reached[0]) if len(reached) else len(states))

    validation = validate_trajectory(
        safe_sets,
        goals,
        safety_distance,
        obstacle_center,
        obstacle_radius,
        goal_tolerance,
        statuses=None,
    )

    return _SafeSetScore(
        all_goals_reached=validation.all_goals_reached,
        usable_for_learning=validation.usable_for_learning,
        pairwise_violation_count=validation.pairwise_violation_count,
        obstacle_violation_count=validation.obstacle_violation_count,
        total_first_hit_steps=sum(first_hit_steps),
        final_goal_error=float(np.sum(final_goal_errors)),
        min_pairwise_distance=validation.min_pairwise_distance,
        min_obstacle_clearance=validation.min_obstacle_clearance,
    )


def _score_key(
    candidate: tuple[
        _SafeSetScore,
        dict[int, np.ndarray],
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
