"""Plotting package for rollout diagnostics."""

from .trajectories import plot_pairwise_distances, plot_trajectories
from .animations import save_rollout_animation
from .manta import plot_cost_decrease, plot_learning_progression, save_manta_animation

__all__ = [
    "plot_cost_decrease",
    "plot_learning_progression",
    "plot_pairwise_distances",
    "plot_trajectories",
    "save_manta_animation",
    "save_rollout_animation",
]
