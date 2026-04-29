"""Minimal decentralized MPC controller solved as a convex QP."""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from .constraints import input_bounds, linearized_collision_constraints


class MPCController:
    """Shared single-agent MPC solver used independently for each agent."""

    def __init__(
        self,
        dt: float,
        horizon: int = 20,
        u_max: float = 1.0,
        goal_weight: float = 1.0,
        control_weight: float = 0.05,
        terminal_weight: float = 8.0,
        solver: str = "OSQP",
    ) -> None:
        """Initialize QP weights, horizon, input limit, and CVXPY solver name."""
        self.dt = dt
        self.horizon = horizon
        self.u_max = u_max
        self.goal_weight = goal_weight
        self.control_weight = control_weight
        self.terminal_weight = terminal_weight
        self.solver = solver
        self.last_prediction: np.ndarray | None = None
        self.last_status = "not_solved"

    def solve(
        self,
        agent_index: int,
        current_states: np.ndarray,
        goals: np.ndarray,
        reference_trajectories: np.ndarray,
        safety_distance: float,
    ) -> np.ndarray:
        """Solve one agent's convex MPC problem and return the first control input."""
        current = np.asarray(current_states, dtype=float)
        target = np.asarray(goals, dtype=float)
        refs = np.asarray(reference_trajectories, dtype=float)
        if current.shape != (3, 2):
            raise ValueError(f"current_states must have shape (3, 2), got {current.shape}")
        if target.shape != (3, 2):
            raise ValueError(f"goals must have shape (3, 2), got {target.shape}")
        if refs.shape != (self.horizon + 1, 3, 2):
            raise ValueError(f"reference_trajectories must have shape {(self.horizon + 1, 3, 2)}, got {refs.shape}")

        x = cp.Variable((self.horizon + 1, 2))
        u = cp.Variable((self.horizon, 2))

        constraints: list[cp.Constraint] = [x[0] == current[agent_index]]
        constraints += input_bounds(u, self.u_max)
        for k in range(self.horizon):
            constraints.append(x[k + 1] == x[k] + self.dt * u[k])
        constraints += linearized_collision_constraints(
            positions=x,
            agent_index=agent_index,
            reference_trajectories=refs,
            safety_distance=safety_distance,
            current_states=current,
        )

        # Each decentralized agent tracks only its own goal while treating the
        # other agents' previous predictions as fixed collision references.
        goal = target[agent_index]
        objective = 0
        for k in range(self.horizon):
            objective += self.goal_weight * cp.sum_squares(x[k] - goal)
            objective += self.control_weight * cp.sum_squares(u[k])
        objective += self.terminal_weight * cp.sum_squares(x[self.horizon] - goal)

        problem = cp.Problem(cp.Minimize(objective), constraints)
        try:
            problem.solve(solver=self.solver, warm_start=True, verbose=False)
        except cp.SolverError:
            self.last_status = "solver_error"
            self.last_prediction = refs[:, agent_index].copy()
            return np.zeros(2, dtype=float)

        self.last_status = problem.status
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or u.value is None or x.value is None:
            self.last_prediction = refs[:, agent_index].copy()
            return np.zeros(2, dtype=float)

        self.last_prediction = np.asarray(x.value, dtype=float)
        return np.asarray(u.value[0], dtype=float)
