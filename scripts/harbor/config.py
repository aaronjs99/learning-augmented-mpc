"""YAML loading for heterogeneous harbor experiments."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .communication import LinkConfig
from .models import make_platform_model
from .simulation import (
    HarborAgent,
    HarborDisturbanceConfig,
    HarborObservationNoiseConfig,
    HarborSimulationConfig,
    OperatingDomain,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HARBOR_CONFIG = PROJECT_ROOT / "config" / "harbor.yaml"


@dataclass(frozen=True)
class HarborFaultStudyConfig:
    """Optimizer overrides used only by the asymmetric actuator-fault study."""

    prediction_horizon: int = 15
    terminal_goal_weight: float = 500.0
    terminal_slack_bound: float = 2.0
    terminal_slack_weight: float = 500.0

    def __post_init__(self) -> None:
        if self.prediction_horizon <= 0:
            raise ValueError("fault-study prediction_horizon must be positive")
        if min(
            self.terminal_goal_weight,
            self.terminal_slack_bound,
            self.terminal_slack_weight,
        ) < 0.0:
            raise ValueError("fault-study MPC weights and bounds must be nonnegative")


@dataclass(frozen=True)
class HarborFaultEnsembleConfig:
    """Seeded actuator-loss ensemble for fault-identification generalization."""

    seeds: tuple[int, ...] = (11, 23, 37, 53, 71)
    effectiveness_min: float = 0.55
    effectiveness_max: float = 0.98
    bootstrap_samples: int = 5000

    def __post_init__(self) -> None:
        seeds = tuple(int(seed) for seed in self.seeds)
        if (
            not seeds
            or len(set(seeds)) != len(seeds)
            or any(seed < 0 for seed in seeds)
        ):
            raise ValueError(
                "fault-ensemble seeds must be nonnegative, nonempty, and unique"
            )
        if not all(0.0 < value <= 1.0 for value in (
            self.effectiveness_min,
            self.effectiveness_max,
        )):
            raise ValueError("fault-ensemble effectiveness bounds must be in (0, 1]")
        if self.effectiveness_min >= self.effectiveness_max:
            raise ValueError("fault-ensemble effectiveness bounds must increase")
        if self.bootstrap_samples <= 0:
            raise ValueError("fault-ensemble bootstrap_samples must be positive")
        object.__setattr__(self, "seeds", seeds)


def load_harbor_config(
    path: str | Path = DEFAULT_HARBOR_CONFIG,
) -> tuple[list[HarborAgent], HarborSimulationConfig, LinkConfig]:
    """Load platform models, independent domains, and communication settings."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}

    simulation = _dataclass_from_mapping(
        HarborSimulationConfig, raw.get("simulation", {}), "simulation"
    )
    communication = _dataclass_from_mapping(
        LinkConfig, raw.get("communication", {}), "communication"
    )
    model_parameters = raw.get("platform_models", {})
    platform_profiles = raw.get("platform_profiles", {})
    agents = []
    for entry in raw.get("agents", []):
        profile_name = entry.get("profile")
        if profile_name is not None:
            if profile_name not in platform_profiles:
                raise ValueError(
                    f"agent {entry['name']} references unknown profile {profile_name}"
                )
            profile = platform_profiles[profile_name]
            kind = str(profile["kind"]).lower()
            if "kind" in entry and str(entry["kind"]).lower() != kind:
                raise ValueError(
                    f"agent {entry['name']} kind disagrees with profile {profile_name}"
                )
            parameters = dict(profile.get("parameters", {}))
            display_name = str(profile.get("label", profile_name))
        else:
            kind = str(entry["kind"]).lower()
            parameters = dict(model_parameters.get(kind, {}))
            display_name = entry.get("display_name")
        model = make_platform_model(kind, parameters)
        start = np.asarray(entry["start"], dtype=float)
        goal = np.asarray(entry["goal"], dtype=float)
        if start.shape != (model.state_dim,):
            raise ValueError(
                f"agent {entry['name']} start must have shape ({model.state_dim},)"
            )
        if goal.shape != (model.pose_dim,):
            raise ValueError(
                f"agent {entry['name']} goal must have shape ({model.pose_dim},)"
            )
        domain_data = entry["domain"]
        domain = OperatingDomain(
            x_bounds=_bounds(domain_data["x"], f"{entry['name']}.domain.x"),
            y_bounds=_bounds(domain_data["y"], f"{entry['name']}.domain.y"),
            z_bounds=_bounds(domain_data["z"], f"{entry['name']}.domain.z"),
        )
        radius = float(entry["radius"])
        if radius <= 0.0:
            raise ValueError(f"agent {entry['name']} radius must be positive")
        agents.append(
            HarborAgent(
                name=str(entry["name"]),
                model=model,
                start=start,
                goal=goal,
                radius=radius,
                domain=domain,
                waypoints=(
                    np.asarray(entry["waypoints"], dtype=float)
                    if "waypoints" in entry
                    else None
                ),
                profile=(str(profile_name) if profile_name is not None else None),
                display_name=(
                    str(entry.get("display_name", display_name))
                    if entry.get("display_name", display_name) is not None
                    else None
                ),
            )
        )
    if len(agents) < 2:
        raise ValueError("harbor config requires at least two agents")
    for agent in agents:
        if agent.model.kind == "ugv" and agent.domain.y_bounds[0] < simulation.shoreline_y:
            raise ValueError("UGV domain must remain on land above the shoreline")
        if agent.model.kind in {"usv", "rov"} and agent.domain.y_bounds[1] > simulation.shoreline_y:
            raise ValueError("USV/ROV domains must remain in harbor water")
        if agent.model.kind == "rov" and agent.domain.z_bounds[1] >= 0.0:
            raise ValueError("ROV domain must remain below the water surface")
    return agents, simulation, communication


def load_harbor_disturbance_config(
    path: str | Path = DEFAULT_HARBOR_CONFIG,
) -> HarborDisturbanceConfig:
    """Load the strict optional ``disturbance_study`` YAML section."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    return _disturbance_from_section(raw, "disturbance_study")


def load_harbor_fault_config(
    path: str | Path = DEFAULT_HARBOR_CONFIG,
) -> HarborDisturbanceConfig:
    """Load the strict optional ``actuator_fault_study`` YAML section."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    return _disturbance_from_section(raw, "actuator_fault_study")


def load_harbor_fault_study_config(
    path: str | Path = DEFAULT_HARBOR_CONFIG,
) -> HarborFaultStudyConfig:
    """Load strict optimizer overrides for the actuator-fault experiment."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    return _dataclass_from_mapping(
        HarborFaultStudyConfig,
        raw.get("actuator_fault_mpc", {}),
        "actuator_fault_mpc",
    )


def load_harbor_fault_ensemble_config(
    path: str | Path = DEFAULT_HARBOR_CONFIG,
) -> HarborFaultEnsembleConfig:
    """Load the strict seeded actuator-fault ensemble section."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    return _dataclass_from_mapping(
        HarborFaultEnsembleConfig,
        raw.get("actuator_fault_ensemble", {}),
        "actuator_fault_ensemble",
    )


def load_harbor_observation_noise_config(
    path: str | Path = DEFAULT_HARBOR_CONFIG,
) -> HarborObservationNoiseConfig:
    """Load strict platform-aware onboard observation noise settings."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    return _dataclass_from_mapping(
        HarborObservationNoiseConfig,
        raw.get("observation_noise", {}),
        "observation_noise",
    )


def _disturbance_from_section(raw: dict[str, Any], section: str):
    return _dataclass_from_mapping(
        HarborDisturbanceConfig,
        raw.get(section, {}),
        section,
    )


def _dataclass_from_mapping(cls, data: dict[str, Any], section: str):
    known = {field.name for field in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"unknown {section} field(s): {', '.join(unknown)}")
    return cls(**data)


def _bounds(values, name: str) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.shape != (2,) or not array[0] < array[1]:
        raise ValueError(f"{name} must contain increasing [min, max] values")
    return float(array[0]), float(array[1])
