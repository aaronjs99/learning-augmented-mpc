"""YAML loading for heterogeneous harbor experiments."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .communication import LinkConfig
from .models import make_platform_model
from .simulation import HarborAgent, HarborSimulationConfig, OperatingDomain

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HARBOR_CONFIG = PROJECT_ROOT / "config" / "harbor.yaml"


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
    agents = []
    for entry in raw.get("agents", []):
        kind = str(entry["kind"]).lower()
        model = make_platform_model(kind, dict(model_parameters.get(kind, {})))
        start = np.asarray(entry["start"], dtype=float)
        goal = np.asarray(entry["goal"], dtype=float)
        if start.shape != (model.state_dim,):
            raise ValueError(
                f"agent {entry['name']} start must have shape ({model.state_dim},)"
            )
        if goal.shape != (3,):
            raise ValueError(f"agent {entry['name']} goal must have shape (3,)")
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
