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


@dataclass(frozen=True)
class HarborTimeVaryingFaultConfig:
    """Seeds and adaptive-estimator settings for scheduled fault studies."""

    observation_seeds: tuple[int, ...] = (131, 197, 263)
    change_threshold: float = 2.5
    covariance_inflation: float = 12.0
    event_detection_window_steps: int = 16
    change_warmup_steps: int = 8
    change_cooldown_steps: int = 10
    recovery_prior_gain: float = 0.20

    def __post_init__(self) -> None:
        seeds = tuple(int(seed) for seed in self.observation_seeds)
        if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
            raise ValueError(
                "time-varying-fault seeds must be nonnegative, nonempty, and unique"
            )
        if self.change_threshold <= 0.0 or self.covariance_inflation < 1.0:
            raise ValueError(
                "change threshold must be positive and covariance inflation at least 1"
            )
        if not 0.0 <= self.recovery_prior_gain <= 1.0:
            raise ValueError("recovery prior gain must lie in [0, 1]")
        integer_steps = (
            self.event_detection_window_steps,
            self.change_warmup_steps,
            self.change_cooldown_steps,
        )
        if any(int(value) != value for value in integer_steps):
            raise ValueError("time-varying-fault step settings must be integers")
        if self.event_detection_window_steps <= 0:
            raise ValueError("event detection window must be positive")
        if self.change_warmup_steps < 0 or self.change_cooldown_steps <= 0:
            raise ValueError(
                "change warmup must be nonnegative and cooldown must be positive"
            )
        object.__setattr__(self, "observation_seeds", seeds)


@dataclass(frozen=True)
class HarborTemporaryFaultEnsembleConfig:
    """Stratified temporary-fault cases for out-of-schedule evaluation."""

    seeds: tuple[int, ...] = (149, 211, 307, 401, 503)
    observation_seed_offset: int = 1000
    effectiveness_min: float = 0.55
    effectiveness_max: float = 0.92
    onset_step_min: int = 8
    onset_step_max: int = 18
    duration_step_min: int = 18
    duration_step_max: int = 28
    bootstrap_samples: int = 5000
    water_current_min: tuple[float, float, float] | None = None
    water_current_max: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        seeds = tuple(int(seed) for seed in self.seeds)
        if (
            not seeds
            or len(set(seeds)) != len(seeds)
            or any(seed < 0 for seed in seeds)
        ):
            raise ValueError(
                "temporary-fault seeds must be nonnegative, nonempty, and unique"
            )
        integer_values = (
            self.observation_seed_offset,
            self.onset_step_min,
            self.onset_step_max,
            self.duration_step_min,
            self.duration_step_max,
            self.bootstrap_samples,
        )
        if any(int(value) != value for value in integer_values):
            raise ValueError("temporary-fault step, seed, and count fields must be integers")
        if self.observation_seed_offset < 0:
            raise ValueError("observation_seed_offset must be nonnegative")
        if not 0.0 < self.effectiveness_min < self.effectiveness_max <= 1.0:
            raise ValueError(
                "temporary-fault effectiveness bounds must increase within (0, 1]"
            )
        if not 0 <= self.onset_step_min <= self.onset_step_max:
            raise ValueError("temporary-fault onset bounds must be ordered")
        if not 1 <= self.duration_step_min <= self.duration_step_max:
            raise ValueError(
                "temporary-fault duration bounds must be positive and ordered"
            )
        if self.bootstrap_samples <= 0:
            raise ValueError("temporary-fault bootstrap_samples must be positive")
        current_bounds = (self.water_current_min, self.water_current_max)
        if (current_bounds[0] is None) != (current_bounds[1] is None):
            raise ValueError(
                "temporary-fault water-current bounds must both be set or omitted"
            )
        if current_bounds[0] is not None:
            lower = tuple(float(value) for value in current_bounds[0])
            upper = tuple(float(value) for value in current_bounds[1])
            if len(lower) != 3 or len(upper) != 3:
                raise ValueError(
                    "temporary-fault water-current bounds must contain x, y, z"
                )
            if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
                raise ValueError("temporary-fault water-current bounds must be finite")
            if any(low > high for low, high in zip(lower, upper)):
                raise ValueError(
                    "temporary-fault water-current bounds must be ordered"
                )
            object.__setattr__(self, "water_current_min", lower)
            object.__setattr__(self, "water_current_max", upper)
        object.__setattr__(self, "seeds", seeds)


@dataclass(frozen=True)
class HarborConfirmationCriteriaConfig:
    """Predeclared closed-loop gates for temporary-fault confirmation."""

    controller_labels: tuple[str, ...] = (
        "Fixed-covariance RLS",
        "Innovation-threshold RLS",
    )
    minimum_adaptive_win_rate: float = 0.8
    require_positive_bootstrap_lower_bound: bool = True
    minimum_completion_rate: float = 1.0
    minimum_safety_rate: float = 1.0
    minimum_fallback_free_rate: float = 1.0
    maximum_mean_completion_cost_delta: float = 0.0

    def __post_init__(self) -> None:
        labels = tuple(str(label) for label in self.controller_labels)
        expected = (
            "Fixed-covariance RLS",
            "Innovation-threshold RLS",
        )
        if labels != expected:
            raise ValueError(
                "confirmation controllers must be fixed-covariance then "
                "innovation-threshold RLS"
            )
        rates = (
            self.minimum_adaptive_win_rate,
            self.minimum_completion_rate,
            self.minimum_safety_rate,
            self.minimum_fallback_free_rate,
        )
        if any(not 0.0 <= value <= 1.0 for value in rates):
            raise ValueError("confirmation rates must lie in [0, 1]")
        if not np.isfinite(self.maximum_mean_completion_cost_delta):
            raise ValueError("confirmation completion-cost bound must be finite")
        object.__setattr__(self, "controller_labels", labels)


@dataclass(frozen=True)
class HarborRecoveryConfirmationCriteriaConfig:
    """Predeclared gates for recovery-prior confirmation."""

    controller_labels: tuple[str, ...] = (
        "Fixed-covariance RLS",
        "Innovation-threshold RLS",
        "Recovery-prior threshold RLS",
    )
    minimum_recovery_win_rate: float = 0.8
    require_positive_bootstrap_lower_bound: bool = True
    minimum_completion_rate: float = 1.0
    minimum_safety_rate: float = 1.0
    minimum_fallback_free_rate: float = 1.0
    maximum_mean_fault_interval_rmse_delta: float = 0.0
    maximum_mean_final_rmse_delta: float = 0.0
    maximum_mean_completion_cost_delta: float = 0.0
    maximum_mean_current_rmse_delta: float = 0.0
    maximum_mean_final_current_rmse_delta: float = 0.0

    def __post_init__(self) -> None:
        labels = tuple(str(label) for label in self.controller_labels)
        valid_candidates = {
            "Recovery-prior threshold RLS",
            "Transient-offset threshold RLS",
        }
        if (
            labels[:2]
            != ("Fixed-covariance RLS", "Innovation-threshold RLS")
            or len(labels) != 3
            or labels[2] not in valid_candidates
        ):
            raise ValueError("recovery confirmation controllers are invalid")
        rates = (
            self.minimum_recovery_win_rate,
            self.minimum_completion_rate,
            self.minimum_safety_rate,
            self.minimum_fallback_free_rate,
        )
        if any(not 0.0 <= value <= 1.0 for value in rates):
            raise ValueError("recovery confirmation rates must lie in [0, 1]")
        bounds = (
            self.maximum_mean_fault_interval_rmse_delta,
            self.maximum_mean_final_rmse_delta,
            self.maximum_mean_completion_cost_delta,
            self.maximum_mean_current_rmse_delta,
            self.maximum_mean_final_current_rmse_delta,
        )
        if any(not np.isfinite(value) for value in bounds):
            raise ValueError("recovery confirmation bounds must be finite")
        object.__setattr__(self, "controller_labels", labels)


@dataclass(frozen=True)
class HarborJointUncertaintyCriteriaConfig:
    """Predeclared control-performance gates under joint hidden uncertainty."""

    controller_labels: tuple[str, ...] = (
        "Innovation-threshold RLS",
        "Transient-offset threshold RLS",
    )
    minimum_candidate_completion_rate: float = 1.0
    minimum_completion_rescue_rate: float = 0.1
    maximum_completion_regression_rate: float = 0.0
    minimum_safety_rate: float = 1.0
    minimum_fallback_free_rate: float = 1.0
    minimum_recovery_win_rate: float = 0.6
    maximum_mean_completion_cost_delta: float = 0.0
    maximum_mean_final_effectiveness_rmse_delta: float = 0.0
    maximum_mean_current_rmse_delta: float = 0.001
    maximum_mean_final_current_rmse_delta: float = 0.001

    def __post_init__(self) -> None:
        labels = tuple(str(label) for label in self.controller_labels)
        if labels != (
            "Innovation-threshold RLS",
            "Transient-offset threshold RLS",
        ):
            raise ValueError("joint-uncertainty controllers are fixed")
        rates = (
            self.minimum_candidate_completion_rate,
            self.minimum_completion_rescue_rate,
            self.maximum_completion_regression_rate,
            self.minimum_safety_rate,
            self.minimum_fallback_free_rate,
            self.minimum_recovery_win_rate,
        )
        if any(not 0.0 <= value <= 1.0 for value in rates):
            raise ValueError("joint-uncertainty rates must lie in [0, 1]")
        bounds = (
            self.maximum_mean_completion_cost_delta,
            self.maximum_mean_final_effectiveness_rmse_delta,
            self.maximum_mean_current_rmse_delta,
            self.maximum_mean_final_current_rmse_delta,
        )
        if any(not np.isfinite(value) for value in bounds):
            raise ValueError("joint-uncertainty bounds must be finite")
        object.__setattr__(self, "controller_labels", labels)


@dataclass(frozen=True)
class HarborProjectedResidualCriteriaConfig:
    """Frozen gates for actuation-subspace disturbance injection."""

    controller_labels: tuple[str, ...] = (
        "Transient-offset threshold RLS",
        "Projected transient-offset RLS",
    )
    minimum_candidate_completion_rate: float = 1.0
    minimum_completion_rescue_rate: float = 0.1
    maximum_completion_regression_rate: float = 0.0
    minimum_safety_rate: float = 1.0
    minimum_fallback_free_rate: float = 1.0
    maximum_mean_completion_cost_delta: float = 0.0
    maximum_mean_recovery_rmse_delta: float = 0.001
    maximum_mean_current_rmse_delta: float = 0.001
    maximum_mean_control_current_rmse_delta: float = 0.0
    maximum_mean_final_control_current_rmse_delta: float = 0.0

    def __post_init__(self) -> None:
        labels = tuple(str(label) for label in self.controller_labels)
        if labels != (
            "Transient-offset threshold RLS",
            "Projected transient-offset RLS",
        ):
            raise ValueError("projected-residual controllers are fixed")
        rates = (
            self.minimum_candidate_completion_rate,
            self.minimum_completion_rescue_rate,
            self.maximum_completion_regression_rate,
            self.minimum_safety_rate,
            self.minimum_fallback_free_rate,
        )
        if any(not 0.0 <= value <= 1.0 for value in rates):
            raise ValueError("projected-residual rates must lie in [0, 1]")
        bounds = (
            self.maximum_mean_completion_cost_delta,
            self.maximum_mean_recovery_rmse_delta,
            self.maximum_mean_current_rmse_delta,
            self.maximum_mean_control_current_rmse_delta,
            self.maximum_mean_final_control_current_rmse_delta,
        )
        if any(not np.isfinite(value) for value in bounds):
            raise ValueError("projected-residual bounds must be finite")
        object.__setattr__(self, "controller_labels", labels)


@dataclass(frozen=True)
class HarborDynamicEnvelopeCriteriaConfig:
    """Frozen gates for elastic versus hard dynamic-state envelopes."""

    controller_labels: tuple[str, ...] = (
        "Hard-envelope transient-offset RLS",
        "Retry-elastic transient-offset RLS",
    )
    minimum_candidate_completion_rate: float = 1.0
    minimum_completion_rescue_rate: float = 0.1
    maximum_completion_regression_rate: float = 0.0
    minimum_safety_rate: float = 1.0
    minimum_fallback_free_rate: float = 1.0
    maximum_mean_completion_cost_delta: float = 5.0
    maximum_mean_recovery_rmse_delta: float = 0.001
    maximum_mean_current_rmse_delta: float = 0.001
    maximum_dynamic_state_slack: float = 0.02

    def __post_init__(self) -> None:
        labels = tuple(str(label) for label in self.controller_labels)
        if labels != (
            "Hard-envelope transient-offset RLS",
            "Retry-elastic transient-offset RLS",
        ):
            raise ValueError("dynamic-envelope controllers are fixed")
        rates = (
            self.minimum_candidate_completion_rate,
            self.minimum_completion_rescue_rate,
            self.maximum_completion_regression_rate,
            self.minimum_safety_rate,
            self.minimum_fallback_free_rate,
        )
        if any(not 0.0 <= value <= 1.0 for value in rates):
            raise ValueError("dynamic-envelope rates must lie in [0, 1]")
        bounds = (
            self.maximum_mean_completion_cost_delta,
            self.maximum_mean_recovery_rmse_delta,
            self.maximum_mean_current_rmse_delta,
            self.maximum_dynamic_state_slack,
        )
        if any(not np.isfinite(value) for value in bounds):
            raise ValueError("dynamic-envelope bounds must be finite")
        if self.maximum_dynamic_state_slack < 0.0:
            raise ValueError("maximum dynamic-state slack must be nonnegative")
        object.__setattr__(self, "controller_labels", labels)


@dataclass(frozen=True)
class HarborStationKeepingCriteriaConfig:
    """Frozen gates for actuator-decoupled marine current observation."""

    controller_labels: tuple[str, ...] = (
        "Hard-envelope transient-offset RLS",
        "Kinematic-current RLS transient-offset RLS",
    )
    minimum_candidate_completion_rate: float = 1.0
    minimum_usv_yaw_win_rate: float = 0.8
    minimum_safety_rate: float = 1.0
    minimum_fallback_free_rate: float = 1.0
    maximum_mean_usv_yaw_error_delta: float = -0.01
    maximum_mean_usv_position_error_delta: float = 0.03
    maximum_mean_usv_current_rmse_delta: float = -0.01
    maximum_mean_recovery_rmse_delta: float = 0.002
    maximum_mean_completion_cost_delta: float = 15.0

    def __post_init__(self) -> None:
        labels = tuple(str(label) for label in self.controller_labels)
        if labels != (
            "Hard-envelope transient-offset RLS",
            "Kinematic-current RLS transient-offset RLS",
        ):
            raise ValueError("station-keeping controllers are fixed")
        rates = (
            self.minimum_candidate_completion_rate,
            self.minimum_usv_yaw_win_rate,
            self.minimum_safety_rate,
            self.minimum_fallback_free_rate,
        )
        if any(not 0.0 <= value <= 1.0 for value in rates):
            raise ValueError("station-keeping rates must lie in [0, 1]")
        bounds = (
            self.maximum_mean_usv_yaw_error_delta,
            self.maximum_mean_usv_position_error_delta,
            self.maximum_mean_usv_current_rmse_delta,
            self.maximum_mean_recovery_rmse_delta,
            self.maximum_mean_completion_cost_delta,
        )
        if any(not np.isfinite(value) for value in bounds):
            raise ValueError("station-keeping bounds must be finite")
        object.__setattr__(self, "controller_labels", labels)


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


def load_harbor_time_varying_fault_config(
    path: str | Path = DEFAULT_HARBOR_CONFIG,
) -> tuple[HarborDisturbanceConfig, HarborTimeVaryingFaultConfig]:
    """Load a scheduled plant fault and its matched experiment settings."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    disturbance = _disturbance_from_section(raw, "time_varying_fault_study")
    experiment = _dataclass_from_mapping(
        HarborTimeVaryingFaultConfig,
        raw.get("time_varying_fault_experiment", {}),
        "time_varying_fault_experiment",
    )
    return disturbance, experiment


def load_harbor_temporary_fault_ensemble_config(
    path: str | Path = DEFAULT_HARBOR_CONFIG,
    section: str = "temporary_fault_ensemble",
) -> HarborTemporaryFaultEnsembleConfig:
    """Load strict stratified temporary-fault ensemble settings."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    return _dataclass_from_mapping(
        HarborTemporaryFaultEnsembleConfig,
        raw.get(section, {}),
        section,
    )


def load_harbor_confirmation_criteria_config(
    path: str | Path = DEFAULT_HARBOR_CONFIG,
) -> HarborConfirmationCriteriaConfig:
    """Load the predeclared temporary-fault confirmation gates."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    return _dataclass_from_mapping(
        HarborConfirmationCriteriaConfig,
        raw.get("temporary_fault_confirmation_criteria", {}),
        "temporary_fault_confirmation_criteria",
    )


def load_harbor_recovery_confirmation_criteria_config(
    path: str | Path = DEFAULT_HARBOR_CONFIG,
    section: str = "temporary_fault_recovery_confirmation_criteria",
) -> HarborRecoveryConfirmationCriteriaConfig:
    """Load predeclared actuator-recovery confirmation gates."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    return _dataclass_from_mapping(
        HarborRecoveryConfirmationCriteriaConfig,
        raw.get(section, {}),
        section,
    )


def load_harbor_joint_uncertainty_criteria_config(
    path: str | Path = DEFAULT_HARBOR_CONFIG,
) -> HarborJointUncertaintyCriteriaConfig:
    """Load frozen joint-current and temporary-fault confirmation gates."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    section = "joint_uncertainty_confirmation_criteria"
    return _dataclass_from_mapping(
        HarborJointUncertaintyCriteriaConfig,
        raw.get(section, {}),
        section,
    )


def load_harbor_projected_residual_criteria_config(
    path: str | Path = DEFAULT_HARBOR_CONFIG,
) -> HarborProjectedResidualCriteriaConfig:
    """Load frozen actuation-subspace projection confirmation gates."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    section = "projected_residual_confirmation_criteria"
    return _dataclass_from_mapping(
        HarborProjectedResidualCriteriaConfig,
        raw.get(section, {}),
        section,
    )


def load_harbor_dynamic_envelope_criteria_config(
    path: str | Path = DEFAULT_HARBOR_CONFIG,
) -> HarborDynamicEnvelopeCriteriaConfig:
    """Load frozen elastic dynamic-envelope confirmation gates."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    section = "dynamic_envelope_confirmation_criteria"
    return _dataclass_from_mapping(
        HarborDynamicEnvelopeCriteriaConfig,
        raw.get(section, {}),
        section,
    )


def load_harbor_station_keeping_criteria_config(
    path: str | Path = DEFAULT_HARBOR_CONFIG,
) -> HarborStationKeepingCriteriaConfig:
    """Load frozen kinematic-current observer confirmation gates."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    section = "station_keeping_confirmation_criteria"
    return _dataclass_from_mapping(
        HarborStationKeepingCriteriaConfig,
        raw.get(section, {}),
        section,
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
