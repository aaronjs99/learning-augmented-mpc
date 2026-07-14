"""Scenario dataclasses and YAML-backed lookup helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StaticObstacle:
    """Circular obstacle metadata used by APF, LMPC, and plotting."""

    center: tuple[float, float]
    # Inflated radius used by LMPC/APF constraints.
    radius: float
    # Physical obstacle radius, when it is smaller than the inflated constraint.
    physical_radius: float | None = None

    def __post_init__(self) -> None:
        """Validate obstacle geometry and physical/inflated radius ordering."""
        center = np.asarray(self.center, dtype=float)
        if center.shape != (2,) or not np.all(np.isfinite(center)):
            raise ValueError("obstacle center must contain two finite coordinates")
        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("obstacle radius must be positive and finite")
        if self.physical_radius is not None and (
            not np.isfinite(self.physical_radius)
            or self.physical_radius <= 0.0
            or self.physical_radius > self.radius
        ):
            raise ValueError(
                "obstacle physical_radius must be positive and no larger than radius"
            )


@dataclass(frozen=True)
class Scenario:
    """Reusable multi-agent manta scenario.

    State rows are ``[x, y, theta, p_L, q_L, p_R, q_R]`` for starts and goals.
    """

    name: str
    starts: np.ndarray
    goals: np.ndarray
    safety_distance: float
    obstacle: StaticObstacle

    def __post_init__(self) -> None:
        """Validate multi-agent state dimensions and finite scenario geometry."""
        starts = np.asarray(self.starts, dtype=float)
        goals = np.asarray(self.goals, dtype=float)
        if not self.name.strip():
            raise ValueError("scenario name must not be empty")
        if starts.ndim != 2 or starts.shape[0] < 1 or starts.shape[1] != 7:
            raise ValueError(f"scenario starts must have shape (A>=1, 7), got {starts.shape}")
        if goals.shape != starts.shape:
            raise ValueError(
                "scenario goals must match starts shape, got "
                f"{goals.shape} and {starts.shape}"
            )
        if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(goals)):
            raise ValueError("scenario starts and goals must contain finite values")
        if not np.isfinite(self.safety_distance) or self.safety_distance <= 0.0:
            raise ValueError("scenario safety_distance must be positive and finite")
        object.__setattr__(self, "starts", starts)
        object.__setattr__(self, "goals", goals)


def get_scenario(name: str, config_path: str | None = None) -> Scenario:
    """Load one scenario from the YAML config.

    New code should usually call ``scripts.config.load_project_config`` so all
    runtime settings come from one place. This helper remains for small scripts
    or notebooks that only need scenario geometry.
    """
    from scripts.config import DEFAULT_CONFIG_PATH, load_project_config

    return load_project_config(
        config_path or DEFAULT_CONFIG_PATH, scenario_name=name
    ).scenario


def list_scenarios(config_path: str | None = None) -> list[str]:
    """List scenario names from the YAML config."""
    from scripts.config import DEFAULT_CONFIG_PATH, list_config_scenarios

    return list_config_scenarios(config_path or DEFAULT_CONFIG_PATH)
