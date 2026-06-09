"""Safe-set initialization and terminal sampling utilities for manta LMPC."""

from __future__ import annotations

import numpy as np

from .apf import APFConfig, simulate_manta_autopilot
from scripts.dynamics import MantaDynamicsConfig, rk4_step_np
from scripts.simulation import Scenario


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


def build_staggered_safe_sets(
    scenario: Scenario,
    *,
    dt: float,
    apf_config: APFConfig = APFConfig(),
    dynamics_config: MantaDynamicsConfig = MantaDynamicsConfig(),
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Build dynamically valid staggered APF iteration-0 safe sets."""
    starts = np.asarray(scenario.starts, dtype=float)
    goals = np.asarray(scenario.goals, dtype=float)

    solo0 = simulate_manta_autopilot(
        starts[0],
        goals[0],
        dt=dt,
        obstacle=scenario.obstacle,
        apf_config=apf_config,
        dynamics_config=dynamics_config,
    )
    _validate_apf_goal_reached(0, solo0, goals[0], apf_config)

    hold1_prefix = hold_trajectory(
        starts[1],
        len(solo0) - 1,
        dt=dt,
        dynamics_config=dynamics_config,
    )
    solo1 = simulate_manta_autopilot(
        hold1_prefix[-1],
        goals[1],
        dt=dt,
        obstacle=scenario.obstacle,
        apf_config=apf_config,
        dynamics_config=dynamics_config,
    )
    _validate_apf_goal_reached(1, solo1, goals[1], apf_config)

    hold2_prefix = hold_trajectory(
        starts[2],
        (len(solo0) - 1) + (len(solo1) - 1),
        dt=dt,
        dynamics_config=dynamics_config,
    )
    solo2 = simulate_manta_autopilot(
        hold2_prefix[-1],
        goals[2],
        dt=dt,
        obstacle=scenario.obstacle,
        apf_config=apf_config,
        dynamics_config=dynamics_config,
    )
    _validate_apf_goal_reached(2, solo2, goals[2], apf_config)

    hold0_suffix = hold_trajectory(
        solo0[-1],
        (len(solo1) - 1) + (len(solo2) - 1),
        dt=dt,
        dynamics_config=dynamics_config,
    )
    hold1_suffix = hold_trajectory(
        solo1[-1],
        len(solo2) - 1,
        dt=dt,
        dynamics_config=dynamics_config,
    )

    safe_sets = {
        0: np.vstack((solo0, hold0_suffix[1:])),
        1: np.vstack((hold1_prefix[:-1], solo1, hold1_suffix[1:])),
        2: np.vstack((hold2_prefix[:-1], solo2)),
    }
    solo_trajs = {0: solo0, 1: solo1, 2: solo2}
    return safe_sets, solo_trajs


def _validate_apf_goal_reached(
    agent: int,
    trajectory: np.ndarray,
    goal: np.ndarray,
    apf_config: APFConfig,
) -> None:
    distance = float(np.linalg.norm(trajectory[-1, :2] - goal[:2]))
    if distance > apf_config.goal_tolerance:
        raise RuntimeError(
            f"APF trajectory for agent {agent} ended {distance:.3f} from goal, "
            f"exceeding tolerance {apf_config.goal_tolerance:.3f}."
        )


def sample_terminal_safe_set(
    safe_set: list[np.ndarray] | np.ndarray,
    target_index: int,
    k_hull: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample terminal safe states and time-to-go costs for the LMPC hull."""
    states = np.asarray(safe_set, dtype=float)
    safe_states = np.zeros((7, k_hull), dtype=float)
    safe_costs = np.zeros((k_hull, 1), dtype=float)
    traj_len = len(states)

    for k_idx in range(k_hull):
        state_idx = min(target_index + k_idx, traj_len - 1)
        safe_states[:, k_idx] = states[state_idx]
        safe_costs[k_idx, 0] = (traj_len - 1) - state_idx

    return safe_states, safe_costs
