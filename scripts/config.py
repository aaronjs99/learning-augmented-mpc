"""YAML-backed configuration loading for manta LMPC runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.dynamics import MantaDynamicsConfig
from scripts.learning.apf import APFConfig
from scripts.mpc import MantaLMPCConfig
from scripts.simulation.scenarios import Scenario, StaticObstacle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "manta.yaml"


@dataclass(frozen=True)
class OutputConfig:
    """Filesystem defaults for generated run artifacts."""

    root_dir: Path
    run_prefix: str


@dataclass(frozen=True)
class PlotConfig:
    """Plot and animation defaults."""

    cost_goal_tolerance: float
    animation_fps: int


@dataclass(frozen=True)
class ProjectConfig:
    """Fully materialized configuration used by run scripts."""

    scenario: Scenario
    scenario_name: str
    dynamics: MantaDynamicsConfig
    apf: APFConfig
    lmpc: MantaLMPCConfig
    output: OutputConfig
    plots: PlotConfig
    make_video: bool
    quiet: bool
    raw: dict[str, Any]


def load_project_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    scenario_name: str | None = None,
) -> ProjectConfig:
    """Load YAML config and convert it into runtime dataclasses."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    run_data = raw.get("run", {})
    selected_scenario = scenario_name or run_data.get("scenario")
    if not selected_scenario:
        raise ValueError("config must set run.scenario or pass --scenario")

    dynamics = MantaDynamicsConfig(**raw.get("dynamics", {}))
    apf = APFConfig(**raw.get("apf", {}))
    lmpc = MantaLMPCConfig(**raw.get("lmpc", {}))
    scenario = _scenario_from_config(raw, selected_scenario)

    output_data = raw.get("output", {})
    output = OutputConfig(
        root_dir=_project_path(output_data.get("root_dir", "results/manta_lmpc")),
        run_prefix=str(output_data.get("run_prefix", "lmpc")),
    )
    plot_data = raw.get("plots", {})
    plots = PlotConfig(
        cost_goal_tolerance=float(plot_data.get("cost_goal_tolerance", 0.5)),
        animation_fps=int(plot_data.get("animation_fps", 20)),
    )

    return ProjectConfig(
        scenario=scenario,
        scenario_name=selected_scenario,
        dynamics=dynamics,
        apf=apf,
        lmpc=lmpc,
        output=output,
        plots=plots,
        make_video=bool(run_data.get("make_video", False)),
        quiet=bool(run_data.get("quiet", False)),
        raw=raw,
    )


def list_config_scenarios(config_path: str | Path = DEFAULT_CONFIG_PATH) -> list[str]:
    """Return scenario names available in a YAML config file."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return sorted((raw.get("scenario") or {}).keys())


def _scenario_from_config(raw: dict[str, Any], name: str) -> Scenario:
    scenario_data = raw.get("scenario", {})
    if name not in scenario_data:
        valid = sorted(scenario_data)
        raise KeyError(f"unknown scenario '{name}'. valid: {valid}")

    data = scenario_data[name]
    obstacle_data = data["obstacle"]
    obstacle = StaticObstacle(
        center=tuple(float(value) for value in obstacle_data["center"]),
        radius=float(obstacle_data["radius"]),
        physical_radius=(
            float(obstacle_data["physical_radius"])
            if "physical_radius" in obstacle_data
            else None
        ),
    )
    return Scenario(
        name=name,
        starts=np.asarray(data["starts"], dtype=float),
        goals=np.asarray(data["goals"], dtype=float),
        safety_distance=float(data["safety_distance"]),
        obstacle=obstacle,
    )


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path
