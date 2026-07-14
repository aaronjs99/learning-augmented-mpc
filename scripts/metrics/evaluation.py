"""Reusable validation and cost evaluation for multi-agent trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


History = dict[int, list[np.ndarray] | np.ndarray]


@dataclass(frozen=True)
class TrajectoryValidation:
    """Safety, completion, and solver-status checks for one trajectory."""

    all_goals_reached: bool
    min_pairwise_distance: float
    pairwise_violation_count: int
    min_obstacle_clearance: float
    obstacle_violation_count: int
    fallback_count: int

    @property
    def safe(self) -> bool:
        """Return whether pairwise and obstacle constraints were satisfied."""
        return (
            self.pairwise_violation_count == 0
            and self.obstacle_violation_count == 0
        )

    @property
    def valid(self) -> bool:
        """Return whether the trajectory is complete and safe."""
        return self.all_goals_reached and self.safe

    @property
    def usable_for_learning(self) -> bool:
        """Return whether the trajectory may be admitted to the safe set."""
        return self.valid

    @property
    def solver_clean(self) -> bool:
        """Return whether no fallback controller was used."""
        return self.fallback_count == 0

    def to_dict(self) -> dict[str, bool | float | int]:
        """Serialize validation metrics for run summaries."""
        return {
            "valid": self.valid,
            "safe": self.safe,
            "solver_clean": self.solver_clean,
            "usable_for_learning": self.usable_for_learning,
            **asdict(self),
        }


def history_to_tensor(history: History) -> tuple[list[int], np.ndarray]:
    """Pad per-agent histories by holding their final states.

    Returns the sorted agent identifiers and an array shaped ``(T, A, D)``.
    """
    if not history:
        raise ValueError("history must contain at least one agent")

    agents = sorted(history)
    trajectories = [np.asarray(history[agent], dtype=float) for agent in agents]
    if any(traj.ndim != 2 or len(traj) == 0 for traj in trajectories):
        raise ValueError("each agent history must have shape (T>=1, D)")
    state_dims = {traj.shape[1] for traj in trajectories}
    if len(state_dims) != 1:
        raise ValueError("all agent histories must use the same state dimension")

    max_len = max(len(traj) for traj in trajectories)
    states = np.empty((max_len, len(agents), trajectories[0].shape[1]), dtype=float)
    for output_agent, trajectory in enumerate(trajectories):
        states[: len(trajectory), output_agent] = trajectory
        states[len(trajectory) :, output_agent] = trajectory[-1]
    return agents, states


def validate_trajectory(
    history: History,
    goals: np.ndarray,
    safety_distance: float,
    obstacle_center: tuple[float, float],
    obstacle_radius: float,
    goal_tolerance: float,
    *,
    statuses: list[dict[int, str]] | None,
) -> TrajectoryValidation:
    """Validate goal completion, pairwise separation, and obstacle clearance."""
    agents, states = history_to_tensor(history)
    goals_array = np.asarray(goals, dtype=float)
    if goals_array.ndim != 2 or goals_array.shape[1] < 2:
        raise ValueError("goals must have shape (A, D>=2)")
    if max(agents) >= len(goals_array):
        raise ValueError("history agent identifiers must index rows in goals")

    positions = states[:, :, :2]
    goal_positions = goals_array[agents, :2]
    goal_error = np.linalg.norm(positions - goal_positions[None, :, :], axis=2)
    all_goals_reached = bool(np.all(goal_error[-1] <= goal_tolerance))

    pairwise_distances = [
        np.linalg.norm(positions[:, i] - positions[:, j], axis=1)
        for i in range(positions.shape[1])
        for j in range(i + 1, positions.shape[1])
    ]
    if pairwise_distances:
        pairwise = np.stack(pairwise_distances, axis=1)
        min_pairwise = float(np.min(pairwise))
        pairwise_violations = int(np.sum(pairwise < safety_distance))
    else:
        min_pairwise = float("inf")
        pairwise_violations = 0

    center = np.asarray(obstacle_center, dtype=float)
    clearance = np.linalg.norm(positions - center[None, None, :], axis=2)
    clearance -= obstacle_radius

    fallback_count = sum(
        status.startswith("fallback")
        for step_statuses in statuses or []
        for status in step_statuses.values()
    )
    return TrajectoryValidation(
        all_goals_reached=all_goals_reached,
        min_pairwise_distance=min_pairwise,
        pairwise_violation_count=pairwise_violations,
        min_obstacle_clearance=float(np.min(clearance)),
        obstacle_violation_count=int(np.sum(clearance < 0.0)),
        fallback_count=fallback_count,
    )


def cost_by_iteration(
    histories: list[History],
    goals: np.ndarray,
    *,
    goal_tolerance: float = 0.5,
) -> dict[int, list[int]]:
    """Return each agent's first goal-tolerance hit step per iteration."""
    goals_array = np.asarray(goals, dtype=float)
    costs: dict[int, list[int]] = {
        agent: [] for agent in range(goals_array.shape[0])
    }
    for history in histories:
        for agent in costs:
            trajectory = np.asarray(history[agent], dtype=float)
            distances = np.linalg.norm(
                trajectory[:, :2] - goals_array[agent, :2], axis=1
            )
            reached = np.flatnonzero(distances <= goal_tolerance)
            costs[agent].append(int(reached[0]) if len(reached) else len(trajectory))
    return costs
