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


@dataclass(frozen=True)
class Scenario:
    """Reusable 3-agent manta scenario.

    State rows are ``[x, y, theta, p_L, q_L, p_R, q_R]`` for starts and goals.
    """

    name: str
    starts: np.ndarray
    goals: np.ndarray
    safety_distance: float
    obstacle: StaticObstacle


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
