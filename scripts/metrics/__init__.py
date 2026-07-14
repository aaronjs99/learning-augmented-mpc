"""Metrics and validation helpers for rollout evaluation."""

from .core import RolloutMetrics, compute_rollout_metrics, pairwise_distances
from .evaluation import (
    TrajectoryValidation,
    cost_by_iteration,
    history_to_tensor,
    validate_trajectory,
)

__all__ = [
    "RolloutMetrics",
    "TrajectoryValidation",
    "compute_rollout_metrics",
    "cost_by_iteration",
    "history_to_tensor",
    "pairwise_distances",
    "validate_trajectory",
]
