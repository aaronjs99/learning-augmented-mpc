"""Named 3-agent scenario definitions for simulation and testing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scenario:
    """Reusable scenario with starts/goals and safety threshold."""

    name: str
    starts: np.ndarray
    goals: np.ndarray
    safety_distance: float


def get_scenario(name: str) -> Scenario:
    """Return one named scenario for 3-agent testing."""
    scenarios = _scenario_map()
    if name not in scenarios:
        raise KeyError(f"unknown scenario '{name}'. valid: {sorted(scenarios)}")
    return scenarios[name]


def list_scenarios() -> list[str]:
    """Return available scenario names."""
    return sorted(_scenario_map())


def _scenario_map() -> dict[str, Scenario]:
    """Create the small fixed set of project scenarios."""
    return {
        "nominal_triangle_rotation": Scenario(
            name="nominal_triangle_rotation",
            starts=np.array([[-1.0, 0.0], [0.5, -0.866], [0.5, 0.866]], dtype=float),
            goals=np.array([[0.5, -0.866], [0.5, 0.866], [-1.0, 0.0]], dtype=float),
            safety_distance=0.35,
        ),
        "crossing_paths": Scenario(
            name="crossing_paths",
            starts=np.array([[-1.2, 0.0], [1.2, 0.0], [0.0, -1.2]], dtype=float),
            goals=np.array([[1.2, 0.0], [-1.2, 0.0], [0.0, 1.2]], dtype=float),
            safety_distance=0.35,
        ),
        "shifted_starts_failure_candidate": Scenario(
            name="shifted_starts_failure_candidate",
            starts=np.array([[-0.3, 0.0], [0.0, -0.25], [0.25, 0.0]], dtype=float),
            goals=np.array([[1.0, 0.8], [-1.0, 0.8], [0.0, -1.2]], dtype=float),
            safety_distance=0.40,
        ),
    }
