"""Safe-set initialization and terminal sampling utilities for manta LMPC."""

from __future__ import annotations

import numpy as np

from .apf import APFConfig, simulate_manta_autopilot
from scripts.dynamics import MantaDynamicsConfig
from scripts.simulation import Scenario


def build_staggered_safe_sets(
    scenario: Scenario,
    *,
    dt: float,
    apf_config: APFConfig = APFConfig(),
    dynamics_config: MantaDynamicsConfig = MantaDynamicsConfig(),
) -> tuple[dict[int, list[np.ndarray]], dict[int, np.ndarray]]:
    """Build the staggered APF iteration-0 safe sets."""
    starts = np.asarray(scenario.starts, dtype=float)
    goals = np.asarray(scenario.goals, dtype=float)

    solo_trajs = {
        agent: simulate_manta_autopilot(
            starts[agent],
            goals[agent],
            dt=dt,
            obstacle=scenario.obstacle,
            apf_config=apf_config,
            dynamics_config=dynamics_config,
        )
        for agent in range(3)
    }

    len_0 = len(solo_trajs[0])
    len_1 = len(solo_trajs[1])
    len_2 = len(solo_trajs[2])

    safe_sets = {
        0: list(
            np.vstack(
                (
                    solo_trajs[0],
                    np.tile(goals[0], (len_1 + len_2, 1)),
                )
            )
        ),
        1: list(
            np.vstack(
                (
                    np.tile(starts[1], (len_0, 1)),
                    solo_trajs[1],
                    np.tile(goals[1], (len_2, 1)),
                )
            )
        ),
        2: list(
            np.vstack(
                (
                    np.tile(starts[2], (len_0 + len_1, 1)),
                    solo_trajs[2],
                )
            )
        ),
    }
    return safe_sets, solo_trajs


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
