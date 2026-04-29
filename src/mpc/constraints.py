"""Shared convex constraints for decentralized MPC."""

from __future__ import annotations

from collections.abc import Iterable

import cvxpy as cp
import numpy as np


def input_bounds(controls: cp.Expression, u_max: float) -> list[cp.Constraint]:
    """Return componentwise input bounds equivalent to ``||u_k||_inf <= u_max``."""
    return [controls <= u_max, controls >= -u_max]


def linearized_collision_constraints(
    positions: cp.Expression,
    agent_index: int,
    reference_trajectories: np.ndarray,
    safety_distance: float,
    current_states: np.ndarray,
    other_agent_indices: Iterable[int] | None = None,
    min_norm: float = 1e-6,
) -> list[cp.Constraint]:
    """Return linearized pairwise collision-avoidance constraints.

    The nonconvex constraint ``||p_i - p_j|| >= d_min`` is replaced by its
    first-order supporting half-space around the previous predicted relative
    position. With ``n`` as the reference separation direction, the convex
    constraint is ``n.T @ (p_i - p_j_ref) >= d_min``.
    """
    refs = np.asarray(reference_trajectories, dtype=float)
    current = np.asarray(current_states, dtype=float)
    if refs.ndim != 3 or refs.shape[1:] != (3, 2):
        raise ValueError(f"reference_trajectories must have shape (N+1, 3, 2), got {refs.shape}")
    if current.shape != (3, 2):
        raise ValueError(f"current_states must have shape (3, 2), got {current.shape}")
    if positions.shape != (refs.shape[0], 2):
        raise ValueError(f"positions must have shape {(refs.shape[0], 2)}, got {positions.shape}")

    others = range(3) if other_agent_indices is None else other_agent_indices
    constraints: list[cp.Constraint] = []
    for k in range(1, refs.shape[0]):
        for other_index in others:
            if other_index == agent_index:
                continue
            delta = refs[k, agent_index] - refs[k, other_index]
            norm = float(np.linalg.norm(delta))
            if norm < min_norm:
                delta = current[agent_index] - current[other_index]
                norm = float(np.linalg.norm(delta))
            if norm < min_norm:
                continue
            normal = delta / norm
            constraints.append(normal @ (positions[k] - refs[k, other_index]) >= safety_distance)
    return constraints
