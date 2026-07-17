"""Bounded active-observability utilities for harbor localization."""

from __future__ import annotations

import numpy as np


def range_information(position: np.ndarray, beacons: np.ndarray,
                      noise_std: float = 0.1) -> np.ndarray:
    """Return the position Fisher information from range geometry."""
    position = np.asarray(position, dtype=float)
    beacons = np.asarray(beacons, dtype=float)
    rows = []
    for beacon in beacons:
        delta = position - beacon[:position.size]
        norm = np.linalg.norm(delta)
        if norm > 1.0e-9:
            rows.append(delta / norm)
    if not rows:
        return np.zeros((position.size, position.size))
    jacobian = np.asarray(rows)
    return jacobian.T @ jacobian / max(noise_std, 1.0e-9) ** 2


def information_score(position: np.ndarray, beacons: np.ndarray,
                      noise_std: float = 0.1) -> float:
    """Score geometry using a bounded log-determinant information measure."""
    matrix = range_information(position, beacons, noise_std)
    eigenvalues = np.linalg.eigvalsh(matrix + 1.0e-9 * np.eye(matrix.shape[0]))
    return float(np.sum(np.log1p(np.maximum(eigenvalues, 0.0))))


def choose_information_waypoint(current: np.ndarray, candidates: np.ndarray,
                                beacons: np.ndarray, task_cost_weight: float = 1.0,
                                information_weight: float = 0.25,
                                max_step: float = 1.0) -> tuple[np.ndarray, float]:
    """Choose a bounded candidate balancing task distance and information gain."""
    current = np.asarray(current, dtype=float)
    candidates = np.asarray(candidates, dtype=float)
    baseline = information_score(current, beacons)
    best = current.copy()
    best_value = float("inf")
    for candidate in candidates:
        candidate = candidate[:current.size]
        if np.linalg.norm(candidate - current) > max_step + 1.0e-9:
            continue
        task_cost = np.linalg.norm(candidate - current)
        gain = information_score(candidate, beacons) - baseline
        value = task_cost_weight * task_cost - information_weight * np.clip(gain, -10.0, 10.0)
        if value < best_value:
            best_value, best = value, candidate.copy()
    return best, float(baseline if np.array_equal(best, current) else information_score(best, beacons) - baseline)

