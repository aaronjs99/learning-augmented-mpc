"""Swept-segment distance geometry for trajectory safety checks."""

from __future__ import annotations

import numpy as np


def segment_point_distances(
    trajectory: np.ndarray,
    point: np.ndarray | tuple[float, float],
) -> np.ndarray:
    """Return minimum point distance along every trajectory segment.

    A one-state trajectory is treated as one degenerate segment so static
    validation still checks its only position.
    """
    positions = _positions(trajectory, "trajectory")
    target = np.asarray(point, dtype=float)
    if target.shape != (2,):
        raise ValueError(f"point must have shape (2,), got {target.shape}")
    relative = positions - target
    return _segment_origin_distances(relative)


def segment_pairwise_distances(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    """Return minimum synchronous separation along each shared segment."""
    first_positions = _positions(first, "first trajectory")
    second_positions = _positions(second, "second trajectory")
    if len(first_positions) != len(second_positions):
        raise ValueError("paired trajectories must have equal lengths")
    return _segment_origin_distances(first_positions - second_positions)


def swept_pairwise_distances(states: np.ndarray) -> np.ndarray:
    """Return swept minimum distances for every agent pair and time interval.

    Args:
        states: array shaped ``(T, A, D)`` with position in the first two dims.

    Returns:
        Array shaped ``(max(1, T - 1), A * (A - 1) / 2)``.
    """
    values = np.asarray(states, dtype=float)
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError(
            f"states must have shape (T>=1, A>=2, D>=2), got {values.shape}"
        )
    if values.shape[2] < 2:
        raise ValueError(
            f"states must have shape (T>=1, A>=2, D>=2), got {values.shape}"
        )

    positions = values[:, :, :2]
    distances = [
        segment_pairwise_distances(positions[:, i], positions[:, j])
        for i in range(positions.shape[1])
        for j in range(i + 1, positions.shape[1])
    ]
    return np.stack(distances, axis=1)


def _segment_origin_distances(relative: np.ndarray) -> np.ndarray:
    if len(relative) == 1:
        return np.linalg.norm(relative, axis=1)

    starts = relative[:-1]
    deltas = relative[1:] - starts
    squared_speed = np.sum(deltas * deltas, axis=1)
    projection = np.zeros(len(starts), dtype=float)
    moving = squared_speed > np.finfo(float).eps
    projection[moving] = -np.sum(starts[moving] * deltas[moving], axis=1)
    projection[moving] /= squared_speed[moving]
    projection = np.clip(projection, 0.0, 1.0)
    closest = starts + projection[:, None] * deltas
    return np.linalg.norm(closest, axis=1)


def _positions(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or len(array) < 1 or array.shape[1] < 2:
        raise ValueError(f"{name} must have shape (T>=1, D>=2), got {array.shape}")
    return array[:, :2]
