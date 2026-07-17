"""Distributed receding-horizon controllers for heterogeneous harbor agents."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import yaml

from .communication import AgentMessage
from .config import DEFAULT_HARBOR_CONFIG, PROJECT_ROOT
from .models import PlatformModel
from .simulation import HarborAgent


@dataclass(frozen=True)
class HarborMPCConfig:
    """Shared optimization settings for MPC and safe-set LMPC modes."""

    prediction_horizon: int = 8
    replan_interval_steps: int = 1
    learning_iterations: int = 2
    terminal_samples: int = 8
    terminal_position_only: bool = True
    seed_learning_from_mpc: bool = True
    position_weight: float = 4.0
    orientation_weight: float = 0.8
    velocity_weight: float = 0.15
    control_weight: float = 0.04
    control_rate_weight: float = 0.25
    terminal_goal_weight: float = 20.0
    terminal_safe_cost_weight: float = 1.0
    terminal_slack_weight: float = 5000.0
    terminal_slack_bound: float = 1.0
    collision_buffer: float = 0.1
    belief_safety_enabled: bool = False
    belief_safety_quantile: float = 2.0
    belief_safety_max_margin: float = 0.5
    collision_slack_weight: float = 100000.0
    collision_slack_bound: float = 0.0
    domain_boundary_margin: float = 0.0
    domain_boundary_slack_weight: float = 100000.0
    dynamic_state_slack_bound: float = 0.10
    dynamic_state_slack_weight: float = 100000.0
    dynamic_state_slack_retry_bound: float = 0.0
    dynamic_state_slack_retry_orientation_error: float = 0.0
    dynamic_state_slack_retry_goal_radius: float = 0.0
    residual_adaptation: bool = False
    residual_adaptation_kinds: tuple[str, ...] = ("ugv", "usv", "rov")
    residual_control_projection: str = "full"
    residual_estimator_gain: float = 0.35
    residual_max_speed: float = 0.35
    residual_estimator_mode: str = "ewma"
    residual_measurement_source: str = "model_prediction"
    residual_rls_initial_std: float = 0.20
    residual_rls_measurement_noise: float = 0.25
    residual_rls_process_noise: float = 1.0e-4
    residual_rls_innovation_gate: float = 3.0
    residual_change_holdoff_steps: int = 0
    control_effectiveness_adaptation: bool = False
    effectiveness_estimator_mode: str = "scalar"
    effectiveness_estimator_gain: float = 0.25
    effectiveness_rls_forgetting_factor: float = 0.995
    effectiveness_rls_measurement_noise: float = 0.02
    effectiveness_rls_process_noise: float = 1.0e-5
    effectiveness_rls_innovation_gate: float = 3.0
    effectiveness_rls_adaptive_covariance: bool = False
    effectiveness_rls_change_detector: str = "threshold"
    effectiveness_rls_change_threshold: float = 3.0
    effectiveness_rls_covariance_inflation: float = 8.0
    effectiveness_rls_change_persistence: int = 1
    effectiveness_rls_change_cooldown_steps: int = 20
    effectiveness_rls_change_warmup_steps: int = 0
    effectiveness_recovery_prior_gain: float = 0.0
    effectiveness_recovery_direction_tolerance: float = 1.0e-6
    effectiveness_recovery_prior_mode: str = "embedded"
    effectiveness_recovery_offset_decay: float = 0.90
    effectiveness_recovery_require_full_rank: bool = True
    effectiveness_recovery_rank_tolerance: float = 1.0e-8
    effectiveness_recovery_max_episodes_per_agent: int = 1
    effectiveness_recovery_minimum_dwell_steps: int = 0
    effectiveness_rls_cusum_drift: float = 1.5
    effectiveness_rls_cusum_threshold: float = 4.0
    effectiveness_min: float = 0.5
    effectiveness_max: float = 1.2
    effectiveness_excitation_threshold: float = 0.05
    active_identification: bool = False
    identification_strategy: str = "information"
    identification_probe_fraction: float = 0.12
    identification_target_energy: float = 0.06
    identification_prior_std: float = 0.25
    identification_measurement_noise: float = 0.005
    identification_target_std: float = 0.15
    identification_fault_focus_weight: float = 2.0
    identification_reset_on_change: bool = False
    identification_arm_on_change: bool = False
    identification_arm_on_loss_only: bool = False
    identification_probe_interval_steps: int = 1
    identification_min_probes_per_channel: int = 2
    identification_max_rejections: int = 2
    identification_clearance_buffer: float = 0.4
    obstacle_prediction_mode: str = "goal_bounded_velocity"
    obstacle_prediction_alignment_threshold: float = 0.5
    ipopt_max_iter: int = 120
    ipopt_print_level: int = 0

    def __post_init__(self) -> None:
        residual_kinds = tuple(str(kind) for kind in self.residual_adaptation_kinds)
        if len(set(residual_kinds)) != len(residual_kinds) or not set(
            residual_kinds
        ) <= {"ugv", "usv", "rov"}:
            raise ValueError(
                "residual_adaptation_kinds must be unique platform kinds"
            )
        object.__setattr__(self, "residual_adaptation_kinds", residual_kinds)
        if self.residual_control_projection not in {
            "full",
            "actuation_subspace",
            "station_keeping_subspace",
        }:
            raise ValueError(
                "residual_control_projection must be full, actuation_subspace, "
                "or station_keeping_subspace"
            )
        if self.residual_estimator_mode not in {"ewma", "constant_bias_rls"}:
            raise ValueError(
                "residual_estimator_mode must be ewma or constant_bias_rls"
            )
        if self.residual_measurement_source not in {
            "model_prediction",
            "kinematic_velocity",
        }:
            raise ValueError(
                "residual_measurement_source must be model_prediction or "
                "kinematic_velocity"
            )
        integer_values = {
            "prediction_horizon": self.prediction_horizon,
            "replan_interval_steps": self.replan_interval_steps,
            "learning_iterations": self.learning_iterations,
            "terminal_samples": self.terminal_samples,
            "identification_probe_interval_steps": self.identification_probe_interval_steps,
            "identification_min_probes_per_channel": self.identification_min_probes_per_channel,
            "identification_max_rejections": self.identification_max_rejections,
            "effectiveness_rls_change_persistence": self.effectiveness_rls_change_persistence,
            "effectiveness_rls_change_cooldown_steps": self.effectiveness_rls_change_cooldown_steps,
            "ipopt_max_iter": self.ipopt_max_iter,
        }
        if any(value <= 0 for value in integer_values.values()):
            raise ValueError("harbor_mpc integer settings must be positive")
        if int(self.effectiveness_rls_change_warmup_steps) != (
            self.effectiveness_rls_change_warmup_steps
        ):
            raise ValueError("effectiveness RLS change warmup must be an integer")
        if self.effectiveness_rls_change_warmup_steps < 0:
            raise ValueError("effectiveness RLS change warmup must be nonnegative")
        if int(self.residual_change_holdoff_steps) != self.residual_change_holdoff_steps:
            raise ValueError("residual change holdoff must be an integer")
        if self.residual_change_holdoff_steps < 0:
            raise ValueError("residual change holdoff must be nonnegative")
        nonnegative = (
            self.position_weight,
            self.orientation_weight,
            self.velocity_weight,
            self.control_weight,
            self.control_rate_weight,
            self.terminal_goal_weight,
            self.terminal_safe_cost_weight,
            self.terminal_slack_weight,
            self.terminal_slack_bound,
            self.collision_buffer,
            self.belief_safety_quantile,
            self.belief_safety_max_margin,
            self.collision_slack_weight,
            self.collision_slack_bound,
            self.domain_boundary_margin,
            self.domain_boundary_slack_weight,
            self.dynamic_state_slack_bound,
            self.dynamic_state_slack_weight,
            self.dynamic_state_slack_retry_bound,
            self.dynamic_state_slack_retry_orientation_error,
            self.dynamic_state_slack_retry_goal_radius,
            self.residual_estimator_gain,
            self.residual_max_speed,
            self.residual_rls_initial_std,
            self.residual_rls_measurement_noise,
            self.residual_rls_process_noise,
            self.residual_rls_innovation_gate,
            self.effectiveness_estimator_gain,
            self.effectiveness_rls_measurement_noise,
            self.effectiveness_rls_process_noise,
            self.effectiveness_rls_innovation_gate,
            self.effectiveness_rls_change_threshold,
            self.effectiveness_rls_covariance_inflation,
            self.effectiveness_rls_cusum_drift,
            self.effectiveness_rls_cusum_threshold,
            self.effectiveness_recovery_prior_gain,
            self.effectiveness_recovery_direction_tolerance,
            self.effectiveness_recovery_rank_tolerance,
            self.effectiveness_min,
            self.effectiveness_max,
            self.effectiveness_excitation_threshold,
            self.identification_probe_fraction,
            self.identification_target_energy,
            self.identification_prior_std,
            self.identification_measurement_noise,
            self.identification_target_std,
            self.identification_fault_focus_weight,
            self.identification_clearance_buffer,
            self.obstacle_prediction_alignment_threshold,
        )
        if any(value < 0.0 for value in nonnegative):
            raise ValueError("harbor_mpc weights and slack bounds must be nonnegative")
        if self.residual_estimator_gain > 1.0:
            raise ValueError("residual_estimator_gain must be in [0, 1]")
        if min(
            self.residual_rls_initial_std,
            self.residual_rls_measurement_noise,
            self.residual_rls_innovation_gate,
        ) <= 0.0:
            raise ValueError("residual RLS uncertainty and gate must be positive")
        if self.effectiveness_estimator_gain > 1.0:
            raise ValueError("effectiveness_estimator_gain must be in [0, 1]")
        if not 0.0 < self.effectiveness_rls_forgetting_factor <= 1.0:
            raise ValueError("effectiveness_rls_forgetting_factor must be in (0, 1]")
        if min(
            self.effectiveness_rls_measurement_noise,
            self.effectiveness_rls_innovation_gate,
            self.effectiveness_rls_change_threshold,
            self.effectiveness_rls_covariance_inflation,
        ) <= 0.0:
            raise ValueError("RLS noise, gates, and covariance inflation must be positive")
        if self.effectiveness_rls_covariance_inflation < 1.0:
            raise ValueError("effectiveness_rls_covariance_inflation must be at least 1")
        if self.effectiveness_recovery_prior_gain > 1.0:
            raise ValueError("effectiveness_recovery_prior_gain must be in [0, 1]")
        if self.effectiveness_recovery_prior_mode not in {"embedded", "transient"}:
            raise ValueError(
                "effectiveness_recovery_prior_mode must be embedded or transient"
            )
        if not 0.0 <= self.effectiveness_recovery_offset_decay < 1.0:
            raise ValueError("effectiveness_recovery_offset_decay must be in [0, 1)")
        if int(self.effectiveness_recovery_max_episodes_per_agent) != (
            self.effectiveness_recovery_max_episodes_per_agent
        ) or self.effectiveness_recovery_max_episodes_per_agent < 0:
            raise ValueError(
                "effectiveness_recovery_max_episodes_per_agent must be a "
                "nonnegative integer"
            )
        if int(self.effectiveness_recovery_minimum_dwell_steps) != (
            self.effectiveness_recovery_minimum_dwell_steps
        ) or self.effectiveness_recovery_minimum_dwell_steps < 0:
            raise ValueError(
                "effectiveness_recovery_minimum_dwell_steps must be a "
                "nonnegative integer"
            )
        if self.effectiveness_rls_change_detector not in {"threshold", "cusum"}:
            raise ValueError(
                "effectiveness_rls_change_detector must be threshold or cusum"
            )
        if not self.effectiveness_min <= 1.0 <= self.effectiveness_max:
            raise ValueError("effectiveness bounds must contain the nominal value 1")
        if self.effectiveness_estimator_mode not in {
            "scalar",
            "diagonal",
            "recursive_diagonal",
        }:
            raise ValueError(
                "effectiveness_estimator_mode must be scalar, diagonal, or "
                "recursive_diagonal"
            )
        if self.identification_probe_fraction > 1.0:
            raise ValueError("identification_probe_fraction must be in [0, 1]")
        if self.active_identification and self.identification_probe_fraction == 0.0:
            raise ValueError(
                "identification_probe_fraction must be positive when active"
            )
        if self.identification_strategy not in {"energy", "information"}:
            raise ValueError("identification_strategy must be energy or information")
        if self.obstacle_prediction_mode not in {
            "constant_velocity",
            "goal_bounded_velocity",
        }:
            raise ValueError(
                "obstacle_prediction_mode must be constant_velocity or "
                "goal_bounded_velocity"
            )
        if self.obstacle_prediction_alignment_threshold > 1.0:
            raise ValueError(
                "obstacle_prediction_alignment_threshold must be in [0, 1]"
            )
        if min(
            self.identification_prior_std,
            self.identification_measurement_noise,
            self.identification_target_std,
        ) <= 0.0:
            raise ValueError("identification uncertainty settings must be positive")


@dataclass(frozen=True)
class HarborMPCStep:
    """One optimizer result with prediction and relaxation telemetry."""

    control: np.ndarray
    predicted_states: np.ndarray
    max_collision_slack: float
    max_domain_buffer_slack: float
    max_dynamic_state_slack: float
    max_terminal_slack: float


def load_harbor_mpc_config(
    path: str | Path = DEFAULT_HARBOR_CONFIG,
) -> HarborMPCConfig:
    """Load the strict ``harbor_mpc`` YAML section."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    data = raw.get("harbor_mpc", {})
    known = {field.name for field in fields(HarborMPCConfig)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"unknown harbor_mpc field(s): {', '.join(unknown)}")
    return HarborMPCConfig(**data)


def _dynamic_state_bounds(model: PlatformModel) -> list[tuple[int, float, float]]:
    """Return nominal velocity and rate envelopes for one platform model."""
    if model.kind == "ugv":
        minimum_speed = -model.max_reverse_speed if model.variant in {
            "kinematic_bicycle",
            "dynamic_skid_steer",
        } else 0.0
        bounds = [(3, minimum_speed, model.max_speed)]
        if model.variant == "dynamic_skid_steer":
            bounds.append((4, -model.max_yaw_rate, model.max_yaw_rate))
        return bounds
    if model.kind == "usv":
        bounds = [(3, 0.0, model.max_speed)]
        if model.variant == "marine_3dof":
            bounds.extend(
                [
                    (4, -model.max_sway_speed, model.max_sway_speed),
                    (5, -model.max_yaw_rate, model.max_yaw_rate),
                ]
            )
        return bounds
    return [
        (6, -model.max_horizontal_speed, model.max_horizontal_speed),
        (7, -model.max_horizontal_speed, model.max_horizontal_speed),
        (8, -model.max_vertical_speed, model.max_vertical_speed),
        (9, -model.max_angular_rate, model.max_angular_rate),
        (10, -model.max_angular_rate, model.max_angular_rate),
        (11, -model.max_angular_rate, model.max_angular_rate),
    ]


class HarborAgentOptimizer:
    """One CasADi NLP for a single platform and communicated obstacle tracks."""

    def __init__(
        self,
        *,
        agent: HarborAgent,
        other_agents: list[HarborAgent],
        config: HarborMPCConfig,
        dt: float,
        learning: bool,
    ) -> None:
        try:
            import casadi as ca
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "CasADi is required for harbor MPC. Install requirements.txt."
            ) from exc
        self.ca = ca
        self.agent = agent
        self.other_agents = tuple(other_agents)
        self.config = config
        self.dt = dt
        self.learning = learning
        self.last_status = "not_run"
        self.last_failure_diagnostics: dict[str, Any] = {}
        self._build_problem()

    def solve(
        self,
        *,
        state: np.ndarray,
        goal: np.ndarray,
        obstacle_predictions: dict[str, np.ndarray],
        safe_states: np.ndarray,
        safe_costs: np.ndarray,
        warm_states: np.ndarray,
        warm_controls: np.ndarray,
        previous_control: np.ndarray,
        position_drift: np.ndarray,
        control_effectiveness: np.ndarray,
        identification_probe: np.ndarray,
        identification_probe_mask: np.ndarray,
        belief_collision_margin: float = 0.0,
    ) -> HarborMPCStep:
        opti = self.opti
        opti.set_value(self.p_initial, state)
        opti.set_value(self.p_goal, goal)
        opti.set_value(self.p_previous_control, previous_control)
        opti.set_value(self.p_position_drift, position_drift)
        opti.set_value(self.p_control_effectiveness, control_effectiveness)
        opti.set_value(self.p_belief_collision_margin, float(max(belief_collision_margin, 0.0)))
        if self.p_identification_probe is not None:
            opti.set_value(self.p_identification_probe, identification_probe)
            opti.set_value(
                self.p_identification_probe_mask, identification_probe_mask
            )
        for index, other in enumerate(self.other_agents):
            opti.set_value(
                self.p_obstacles[index], obstacle_predictions[other.name].T
            )
        if self.learning:
            opti.set_value(self.p_safe_pose, self._terminal_rows(safe_states))
            opti.set_value(self.p_safe_costs, safe_costs.reshape(-1, 1))
            opti.set_initial(self.lambdas, 1.0 / self.config.terminal_samples)
            opti.set_initial(self.terminal_slack, 0.0)
        opti.set_initial(self.states, warm_states.T)
        opti.set_initial(self.controls, warm_controls.T)
        opti.set_initial(self.collision_slack, 0.0)
        opti.set_initial(self.domain_buffer_slack, 0.0)
        opti.set_initial(self.dynamic_state_slack, 0.0)
        try:
            solution = opti.solve()
        except RuntimeError:
            self.last_status = str(opti.stats().get("return_status", "unknown"))
            self.last_failure_diagnostics = self._collect_failure_diagnostics()
            raise
        self.last_status = str(opti.stats().get("return_status", "unknown"))
        self.last_failure_diagnostics = {}
        states = np.asarray(solution.value(self.states), dtype=float).T
        collision = np.asarray(
            solution.value(self.collision_slack), dtype=float
        ).reshape(-1)
        domain_buffer = np.asarray(
            solution.value(self.domain_buffer_slack), dtype=float
        ).reshape(-1)
        dynamic_state = np.asarray(
            solution.value(self.dynamic_state_slack), dtype=float
        ).reshape(-1)
        terminal = (
            np.asarray(solution.value(self.terminal_slack), dtype=float).reshape(-1)
            if self.learning
            else np.zeros(1)
        )
        return HarborMPCStep(
            control=np.asarray(solution.value(self.controls[:, 0]), dtype=float).reshape(
                self.agent.model.control_dim
            ),
            predicted_states=states,
            max_collision_slack=float(np.max(collision, initial=0.0)),
            max_domain_buffer_slack=float(
                np.max(domain_buffer, initial=0.0)
            ),
            max_dynamic_state_slack=float(
                np.max(dynamic_state, initial=0.0)
            ),
            max_terminal_slack=float(np.max(np.abs(terminal), initial=0.0)),
        )

    def _collect_failure_diagnostics(self, limit: int = 6) -> dict[str, Any]:
        """Capture the largest constraint residuals at IPOPT's final iterate."""
        try:
            values = np.asarray(
                self.opti.debug.value(self.opti.g), dtype=float
            ).reshape(-1)
            lower = np.asarray(
                self.opti.debug.value(self.opti.lbg), dtype=float
            ).reshape(-1)
            upper = np.asarray(
                self.opti.debug.value(self.opti.ubg), dtype=float
            ).reshape(-1)
        except Exception:
            return {"status": self.last_status, "constraints": []}
        violations = np.maximum(lower - values, values - upper)
        indices = np.argsort(violations)[::-1]
        constraints = []
        for index in indices[:limit]:
            violation = float(max(violations[index], 0.0))
            if violation <= 1.0e-7:
                break
            constraints.append(
                {
                    "index": int(index),
                    "violation": violation,
                    "value": float(values[index]),
                    "lower": float(lower[index]),
                    "upper": float(upper[index]),
                    "description": str(self.opti.debug.g_describe(int(index))),
                }
            )
        return {"status": self.last_status, "constraints": constraints}

    def _build_problem(self) -> None:
        ca = self.ca
        model = self.agent.model
        cfg = self.config
        nx = model.state_dim
        nu = model.control_dim
        pose_dim = model.pose_dim
        horizon = cfg.prediction_horizon
        obstacle_count = len(self.other_agents)
        opti = ca.Opti()
        states = opti.variable(nx, horizon + 1)
        controls = opti.variable(nu, horizon)
        collision_slack = opti.variable(obstacle_count, horizon)
        domain_axes = 3 if model.kind == "rov" else 2
        domain_buffer_slack = opti.variable(2 * domain_axes, horizon)
        dynamic_bounds = _dynamic_state_bounds(model)
        dynamic_state_slack = opti.variable(2 * len(dynamic_bounds), horizon)
        p_initial = opti.parameter(nx, 1)
        p_goal = opti.parameter(pose_dim, 1)
        p_previous_control = opti.parameter(nu, 1)
        p_position_drift = opti.parameter(3, 1)
        p_control_effectiveness = opti.parameter(nu, 1)
        p_belief_collision_margin = opti.parameter()
        p_identification_probe = (
            opti.parameter(nu, 1) if cfg.active_identification else None
        )
        p_identification_probe_mask = (
            opti.parameter(nu, 1) if cfg.active_identification else None
        )
        p_obstacles = [opti.parameter(3, horizon) for _ in self.other_agents]

        opti.subject_to(states[:, 0] == p_initial)
        opti.subject_to(
            opti.bounded(0.0, collision_slack, cfg.collision_slack_bound)
        )
        opti.subject_to(
            opti.bounded(0.0, domain_buffer_slack, cfg.domain_boundary_margin)
        )
        opti.subject_to(
            opti.bounded(
                0.0, dynamic_state_slack, cfg.dynamic_state_slack_bound
            )
        )
        self._apply_bounds(
            opti, states, controls, domain_buffer_slack, dynamic_state_slack
        )
        if cfg.active_identification:
            opti.subject_to(
                ca.times(
                    p_identification_probe_mask,
                    controls[:, 0] - p_identification_probe,
                )
                == 0.0
            )

        cost = 0
        previous = p_previous_control
        control_scale = ca.DM(model.control_scale())
        for index in range(horizon):
            opti.subject_to(
                states[:, index + 1]
                == _apply_symbolic_position_drift(
                    ca,
                    model,
                    _symbolic_step(
                        ca,
                        model,
                        states[:, index],
                        ca.times(p_control_effectiveness, controls[:, index]),
                        self.dt,
                    ),
                    p_position_drift,
                    self.dt,
                )
            )
            cost += self._tracking_cost(states[:, index + 1], p_goal)
            normalized_control = ca.rdivide(controls[:, index], control_scale)
            cost += cfg.control_weight * ca.dot(
                normalized_control, normalized_control
            )
            delta_control = ca.rdivide(
                controls[:, index] - previous, control_scale
            )
            cost += cfg.control_rate_weight * ca.dot(delta_control, delta_control)
            previous = controls[:, index]
            own_position = _symbolic_position(ca, model, states[:, index + 1])
            for obstacle_index, other in enumerate(self.other_agents):
                relative = own_position - p_obstacles[obstacle_index][:, index]
                required = (
                    self.agent.radius + other.radius + cfg.collision_buffer
                    + p_belief_collision_margin
                )
                opti.subject_to(
                    ca.dot(relative, relative) + collision_slack[obstacle_index, index]
                    >= required * required
                )

        cost += cfg.collision_slack_weight * ca.sum1(ca.vec(collision_slack))
        cost += cfg.domain_boundary_slack_weight * ca.sumsqr(domain_buffer_slack)
        cost += cfg.dynamic_state_slack_weight * ca.sumsqr(dynamic_state_slack)
        terminal_pose = states[:pose_dim, horizon]
        if self.learning:
            sample_count = cfg.terminal_samples
            terminal_dim = (
                2 if model.pose_dim == 3 else 3
            ) if cfg.terminal_position_only else pose_dim
            p_safe_pose = opti.parameter(sample_count, terminal_dim)
            p_safe_costs = opti.parameter(sample_count, 1)
            lambdas = opti.variable(sample_count, 1)
            terminal_slack = opti.variable(terminal_dim, 1)
            opti.subject_to(opti.bounded(0.0, lambdas, 1.0))
            opti.subject_to(ca.sum1(lambdas) == 1.0)
            opti.subject_to(
                opti.bounded(
                    -cfg.terminal_slack_bound,
                    terminal_slack,
                    cfg.terminal_slack_bound,
                )
            )
            safe_terminal = ca.mtimes(p_safe_pose.T, lambdas)
            opti.subject_to(
                terminal_pose[:terminal_dim] == safe_terminal + terminal_slack
            )
            cost += cfg.terminal_slack_weight * ca.dot(
                terminal_slack, terminal_slack
            )
            cost += cfg.terminal_safe_cost_weight * ca.dot(p_safe_costs, lambdas)
            cost += cfg.terminal_goal_weight * self._pose_error_cost(
                terminal_pose, p_goal
            )
            self.p_safe_pose = p_safe_pose
            self.p_safe_costs = p_safe_costs
            self.lambdas = lambdas
            self.terminal_slack = terminal_slack
        else:
            cost += cfg.terminal_goal_weight * self._pose_error_cost(
                terminal_pose, p_goal
            )

        opti.minimize(cost)
        options = {
            "expand": True,
            "print_time": False,
            "ipopt.print_level": cfg.ipopt_print_level,
            "ipopt.max_iter": cfg.ipopt_max_iter,
            "ipopt.sb": "yes",
        }
        opti.solver("ipopt", options)
        self.opti = opti
        self.states = states
        self.controls = controls
        self.collision_slack = collision_slack
        self.domain_buffer_slack = domain_buffer_slack
        self.dynamic_state_slack = dynamic_state_slack
        self.p_initial = p_initial
        self.p_goal = p_goal
        self.p_previous_control = p_previous_control
        self.p_position_drift = p_position_drift
        self.p_control_effectiveness = p_control_effectiveness
        self.p_belief_collision_margin = p_belief_collision_margin
        self.p_identification_probe = p_identification_probe
        self.p_identification_probe_mask = p_identification_probe_mask
        self.p_obstacles = p_obstacles

    def _tracking_cost(self, state, goal):
        ca = self.ca
        cfg = self.config
        model = self.agent.model
        position = _symbolic_position(ca, model, state)
        goal_position = _symbolic_goal_position(ca, model, goal)
        velocity = _symbolic_velocity(model, state)
        return (
            cfg.position_weight * ca.sumsqr(position - goal_position)
            + cfg.orientation_weight * _symbolic_orientation_cost(ca, model, state, goal)
            + cfg.velocity_weight * ca.sumsqr(velocity)
        )

    def _pose_error_cost(self, pose, goal):
        ca = self.ca
        model = self.agent.model
        linear_dim = 2 if model.pose_dim == 3 else 3
        value = ca.sumsqr(pose[:linear_dim] - goal[:linear_dim])
        for index in range(linear_dim, model.pose_dim):
            value += 2.0 * (1.0 - ca.cos(pose[index] - goal[index]))
        return value

    def _apply_bounds(
        self,
        opti,
        states,
        controls,
        domain_buffer_slack,
        dynamic_state_slack,
    ) -> None:
        model = self.agent.model
        domain = self.agent.domain
        margin = self.config.domain_boundary_margin
        predicted_states = states[:, 1:]
        domains = [domain.x_bounds, domain.y_bounds]
        if model.kind == "rov":
            domains.append(domain.z_bounds)
        for axis, bounds in enumerate(domains):
            values = predicted_states[axis, :]
            lower_slack = domain_buffer_slack[2 * axis, :]
            upper_slack = domain_buffer_slack[2 * axis + 1, :]
            opti.subject_to(opti.bounded(bounds[0], values, bounds[1]))
            opti.subject_to(values >= bounds[0] + margin - lower_slack)
            opti.subject_to(values <= bounds[1] - margin + upper_slack)
        for bound_index, (state_index, lower, upper) in enumerate(
            _dynamic_state_bounds(model)
        ):
            values = predicted_states[state_index, :]
            opti.subject_to(
                values >= lower - dynamic_state_slack[2 * bound_index, :]
            )
            opti.subject_to(
                values <= upper + dynamic_state_slack[2 * bound_index + 1, :]
            )
        if model.kind == "ugv":
            if model.variant == "dynamic_skid_steer":
                opti.subject_to(
                    opti.bounded(
                        -model.max_side_force,
                        controls[0, :],
                        model.max_side_force,
                    )
                )
                opti.subject_to(
                    opti.bounded(
                        -model.max_side_force,
                        controls[1, :],
                        model.max_side_force,
                    )
                )
            else:
                opti.subject_to(
                    opti.bounded(
                        -model.max_acceleration,
                        controls[0, :],
                        model.max_acceleration,
                    )
                )
                steering_bound = (
                    model.max_steering_angle
                    if model.variant == "kinematic_bicycle"
                    else model.max_yaw_rate
                )
                opti.subject_to(
                    opti.bounded(-steering_bound, controls[1, :], steering_bound)
                )
        elif model.kind == "usv":
            thrust_bound = (
                model.max_jet_thrust
                if model.variant == "marine_3dof"
                else model.max_thrust
            )
            opti.subject_to(
                opti.bounded(-thrust_bound, controls[0, :], thrust_bound)
            )
            second_control_bound = (
                model.max_jet_thrust
                if model.variant == "marine_3dof"
                else model.max_yaw_rate
            )
            opti.subject_to(
                opti.bounded(-second_control_bound, controls[1, :], second_control_bound)
            )
        else:
            for index, limit in enumerate(model.control_scale()):
                opti.subject_to(
                    opti.bounded(-limit, controls[index, :], limit)
                )

    def _terminal_rows(self, states: np.ndarray) -> np.ndarray:
        terminal_dim = (
            2 if self.agent.model.pose_dim == 3 else 3
        ) if self.config.terminal_position_only else self.agent.model.pose_dim
        return np.asarray(states, dtype=float)[:, :terminal_dim]


class DistributedHarborMPC:
    """Per-agent MPC using local state, received messages, and optional safe data."""

    def __init__(
        self,
        *,
        agents: list[HarborAgent],
        config: HarborMPCConfig,
        dt: float,
        safe_states: dict[str, np.ndarray],
        safe_controls: dict[str, np.ndarray],
        learning: bool,
        initial_effectiveness_estimates: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.agents = {agent.name: agent for agent in agents}
        self.config = config
        self.dt = dt
        self.safe_states = safe_states
        self.safe_controls = safe_controls
        self.learning = learning
        self.optimizers = {
            agent.name: HarborAgentOptimizer(
                agent=agent,
                other_agents=[other for other in agents if other.name != agent.name],
                config=config,
                dt=dt,
                learning=learning,
            )
            for agent in agents
        }
        self.dynamic_state_retry_optimizers = (
            {
                agent.name: HarborAgentOptimizer(
                    agent=agent,
                    other_agents=[
                        other for other in agents if other.name != agent.name
                    ],
                    config=replace(
                        config,
                        dynamic_state_slack_bound=(
                            config.dynamic_state_slack_retry_bound
                        ),
                    ),
                    dt=dt,
                    learning=learning,
                )
                for agent in agents
            }
            if config.dynamic_state_slack_retry_bound > 0.0
            else {}
        )
        self.previous_controls = {
            agent.name: np.zeros(agent.model.control_dim) for agent in agents
        }
        self.previous_states: dict[str, np.ndarray] = {}
        self.previous_update_steps: dict[str, int] = {}
        self.position_drift_estimates = {
            agent.name: np.zeros(3) for agent in agents
        }
        self.position_drift_variances = {
            agent.name: np.full(3, config.residual_rls_initial_std**2)
            for agent in agents
        }
        self.residual_rejection_count_by_agent = {
            agent.name: 0 for agent in agents
        }
        self.residual_history = {agent.name: [] for agent in agents}
        self.control_residual_history = {agent.name: [] for agent in agents}
        self.residual_projection_retry_count_by_agent = {
            agent.name: 0 for agent in agents
        }
        self.dynamic_state_retry_count_by_agent = {
            agent.name: 0 for agent in agents
        }
        self.control_effectiveness_estimates = {
            agent.name: np.ones(agent.model.control_dim, dtype=float)
            for agent in agents
        }
        if initial_effectiveness_estimates is not None:
            unknown = set(initial_effectiveness_estimates) - set(self.agents)
            if unknown:
                raise ValueError(
                    "initial effectiveness contains unknown agent(s): "
                    + ", ".join(sorted(unknown))
                )
            for agent in agents:
                if agent.name not in initial_effectiveness_estimates:
                    continue
                estimate = np.asarray(
                    initial_effectiveness_estimates[agent.name], dtype=float
                ).reshape(-1)
                if estimate.shape != (agent.model.control_dim,):
                    raise ValueError(
                        f"{agent.name} initial effectiveness must have "
                        f"{agent.model.control_dim} entries"
                    )
                if (
                    np.any(estimate < config.effectiveness_min)
                    or np.any(estimate > config.effectiveness_max)
                ):
                    raise ValueError(
                        f"{agent.name} initial effectiveness is outside estimator bounds"
                    )
                self.control_effectiveness_estimates[agent.name] = estimate.copy()
        self.raw_effectiveness_estimates = {
            name: estimate.copy()
            for name, estimate in self.control_effectiveness_estimates.items()
        }
        self.recovery_effectiveness_offsets = {
            agent.name: np.zeros(agent.model.control_dim, dtype=float)
            for agent in agents
        }
        self.effectiveness_history = {agent.name: [] for agent in agents}
        self.raw_effectiveness_history = {agent.name: [] for agent in agents}
        self.effectiveness_covariances = {
            agent.name: np.eye(agent.model.control_dim)
            * config.identification_prior_std**2
            for agent in agents
        }
        self.effectiveness_change_steps_by_agent = {
            agent.name: [] for agent in agents
        }
        self.recovery_offset_steps_by_agent = {agent.name: [] for agent in agents}
        self.recovery_offset_rejections_by_agent = {
            agent.name: 0 for agent in agents
        }
        self.recovery_offset_unarmed_rejections_by_agent = {
            agent.name: 0 for agent in agents
        }
        self.recovery_offset_episode_limit_rejections_by_agent = {
            agent.name: 0 for agent in agents
        }
        self.recovery_offset_dwell_rejections_by_agent = {
            agent.name: 0 for agent in agents
        }
        self.recovery_offset_armed_by_agent = {
            agent.name: False for agent in agents
        }
        self.recovery_offset_armed_step_by_agent: dict[str, int | None] = {
            agent.name: None for agent in agents
        }
        self.effectiveness_change_evidence_by_agent = {
            agent.name: 0.0 for agent in agents
        }
        self.effectiveness_last_change_step_by_agent = {
            agent.name: -config.effectiveness_rls_change_cooldown_steps
            for agent in agents
        }
        self.excitation_energy = {
            agent.name: np.zeros(agent.model.control_dim, dtype=float)
            for agent in agents
        }
        self.excitation_history = {agent.name: [] for agent in agents}
        self.information_matrices = {
            agent.name: np.zeros(
                (agent.model.control_dim, agent.model.control_dim), dtype=float
            )
            for agent in agents
        }
        self.information_std_history = {agent.name: [] for agent in agents}
        self.identification_probe_count_by_agent = {
            agent.name: 0 for agent in agents
        }
        self.identification_probe_channel_counts = {
            agent.name: np.zeros(agent.model.control_dim, dtype=int)
            for agent in agents
        }
        self.identification_probe_quota_counts = {
            agent.name: np.zeros(agent.model.control_dim, dtype=int)
            for agent in agents
        }
        self.identification_probe_sequence_by_agent: dict[str, list[int]] = {
            agent.name: [] for agent in agents
        }
        self.identification_probe_rejection_counts = {
            agent.name: np.zeros(agent.model.control_dim, dtype=int)
            for agent in agents
        }
        self.identification_change_armed_by_agent = {
            agent.name: not config.identification_arm_on_change for agent in agents
        }
        self.solve_count = 0
        self.fallback_count = 0
        self.solve_time_seconds = 0.0
        self.max_collision_slack = 0.0
        self.max_domain_buffer_slack = 0.0
        self.max_dynamic_state_slack = 0.0
        self.max_terminal_slack = 0.0
        self.solve_count_by_agent = {agent.name: 0 for agent in agents}
        self.fallback_count_by_agent = {agent.name: 0 for agent in agents}
        self.failure_steps_by_agent: dict[str, list[int]] = {
            agent.name: [] for agent in agents
        }
        self.failure_status_counts: dict[str, int] = {}
        self.failure_diagnostics_by_agent: dict[str, list[dict[str, Any]]] = {
            agent.name: [] for agent in agents
        }

    def control(
        self,
        *,
        agent: HarborAgent,
        state: np.ndarray,
        navigation_goal: np.ndarray,
        desired_velocity: np.ndarray,
        inbox: dict[str, AgentMessage],
        step: int,
        dt: float,
        belief: dict | None = None,
    ) -> np.ndarray:
        self.observe(agent=agent, state=state, step=step, dt=dt)
        safe_states = self.safe_states[agent.name]
        safe_controls = self.safe_controls[agent.name]
        nearest = _nearest_reference_index(agent.model, safe_states, state)
        reference_index = min(max(step, nearest), len(safe_states) - 1)
        safe_sample_states, safe_costs = _sample_safe_states(
            safe_states,
            reference_index + self.config.prediction_horizon,
            self.config.terminal_samples,
        )
        warm_states, warm_controls = _warm_reference(
            safe_states,
            safe_controls,
            reference_index,
            self.config.prediction_horizon,
        )
        predictions = self._obstacle_predictions(agent, inbox, step)
        probe, probe_mask, probe_channel = self._identification_probe(
            agent,
            state,
            desired_velocity,
            inbox,
            step,
        )
        started = perf_counter()
        control_position_drift = _control_position_drift(
            agent.model,
            state,
            self.position_drift_estimates[agent.name],
            self.config.residual_control_projection,
            desired_velocity,
        )
        belief_margin = 0.0
        if self.config.belief_safety_enabled and belief is not None:
            covariance = np.asarray(belief.get("position_covariance", []), dtype=float)
            if covariance.ndim == 2 and covariance.size:
                belief_margin = min(
                    self.config.belief_safety_max_margin,
                    self.config.belief_safety_quantile
                    * float(np.sqrt(max(np.trace(covariance), 0.0))),
                )
        solve_arguments = dict(
            state=np.asarray(state, dtype=float),
            goal=_approach_pose_goal(
                agent.model,
                navigation_goal,
                desired_velocity,
            ),
            obstacle_predictions=predictions,
            safe_states=safe_sample_states,
            safe_costs=safe_costs,
            warm_states=warm_states,
            warm_controls=warm_controls,
            previous_control=self.previous_controls[agent.name],
            position_drift=control_position_drift,
            control_effectiveness=self.control_effectiveness_estimates[agent.name],
            identification_probe=probe,
            identification_probe_mask=probe_mask,
            belief_collision_margin=belief_margin,
        )
        used_dynamic_retry = False
        try:
            solution = self.optimizers[agent.name].solve(**solve_arguments)
        except RuntimeError:
            solution = None
            if probe_channel is not None:
                self.identification_probe_rejection_counts[agent.name][
                    probe_channel
                ] += 1
                solve_arguments["identification_probe"] = np.zeros_like(probe)
                solve_arguments["identification_probe_mask"] = np.zeros_like(
                    probe_mask
                )
                try:
                    solution = self.optimizers[agent.name].solve(**solve_arguments)
                except RuntimeError:
                    pass
            if solution is None:
                retry_optimizer = self.dynamic_state_retry_optimizers.get(
                    agent.name
                )
                if retry_optimizer is not None:
                    try:
                        solution = retry_optimizer.solve(**solve_arguments)
                    except RuntimeError:
                        pass
                    else:
                        self.dynamic_state_retry_count_by_agent[agent.name] += 1
                        used_dynamic_retry = True
            if solution is None:
                retry_projection = _control_position_drift(
                    agent.model,
                    state,
                    self.position_drift_estimates[agent.name],
                    "actuation_subspace",
                    desired_velocity,
                )
                can_retry_projection = (
                    self.config.residual_control_projection
                    == "station_keeping_subspace"
                    and agent.model.kind == "usv"
                    and not np.allclose(
                        retry_projection,
                        solve_arguments["position_drift"],
                        atol=1.0e-12,
                    )
                )
                if can_retry_projection:
                    solve_arguments["position_drift"] = retry_projection
                    try:
                        solution = self.optimizers[agent.name].solve(
                            **solve_arguments
                        )
                    except RuntimeError:
                        pass
                    else:
                        control_position_drift = retry_projection
                        self.residual_projection_retry_count_by_agent[
                            agent.name
                        ] += 1
            if solution is None:
                self.fallback_count += 1
                self.fallback_count_by_agent[agent.name] += 1
                self.failure_steps_by_agent[agent.name].append(step)
                status = self.optimizers[agent.name].last_status
                self.failure_status_counts[status] = (
                    self.failure_status_counts.get(status, 0) + 1
                )
                if not self.failure_diagnostics_by_agent[agent.name]:
                    diagnostic_optimizer = self.optimizers[agent.name]
                    diagnostic = {
                        "step": int(step),
                        "state": np.asarray(state, dtype=float).tolist(),
                        "previous_control": self.previous_controls[
                            agent.name
                        ].tolist(),
                        "control_effectiveness": self.control_effectiveness_estimates[
                            agent.name
                        ].tolist(),
                        **diagnostic_optimizer.last_failure_diagnostics,
                    }
                    self.failure_diagnostics_by_agent[agent.name].append(diagnostic)
                    print(
                        f"MPC failure diagnostic for {agent.name}: {diagnostic}",
                        flush=True,
                    )
                control = agent.model.guidance_control(
                    state, desired_velocity, dt, desired_pose=navigation_goal
                )
            else:
                control = solution.control
                probe_channel = None
        else:
            control = solution.control
        retry_optimizer = self.dynamic_state_retry_optimizers.get(agent.name)
        yaw_index = agent.model.pose_dim - 1
        measured_yaw_error = abs(
            np.arctan2(
                np.sin(state[yaw_index] - navigation_goal[yaw_index]),
                np.cos(state[yaw_index] - navigation_goal[yaw_index]),
            )
        )
        should_compare_retry = (
            solution is not None
            and not used_dynamic_retry
            and retry_optimizer is not None
            and agent.model.kind == "usv"
            and self.config.dynamic_state_slack_retry_orientation_error > 0.0
            and np.linalg.norm(state[:2] - navigation_goal[:2])
            <= self.config.dynamic_state_slack_retry_goal_radius
            and measured_yaw_error
            > self.config.dynamic_state_slack_retry_orientation_error
        )
        if should_compare_retry:
            try:
                retry_solution = retry_optimizer.solve(**solve_arguments)
            except RuntimeError:
                pass
            else:
                solution = retry_solution
                control = solution.control
                self.dynamic_state_retry_count_by_agent[agent.name] += 1
        if solution is not None:
            self.solve_count += 1
            self.solve_count_by_agent[agent.name] += 1
            self.max_collision_slack = max(
                self.max_collision_slack, solution.max_collision_slack
            )
            self.max_domain_buffer_slack = max(
                self.max_domain_buffer_slack,
                solution.max_domain_buffer_slack,
            )
            self.max_dynamic_state_slack = max(
                self.max_dynamic_state_slack,
                solution.max_dynamic_state_slack,
            )
            self.max_terminal_slack = max(
                self.max_terminal_slack, solution.max_terminal_slack
            )
            if probe_channel is not None:
                self.identification_probe_count_by_agent[agent.name] += 1
                self.identification_probe_channel_counts[agent.name][
                    probe_channel
                ] += 1
                self.identification_probe_quota_counts[agent.name][probe_channel] += 1
                self.identification_probe_sequence_by_agent[agent.name].append(
                    probe_channel
                )
        self.solve_time_seconds += perf_counter() - started
        self.previous_controls[agent.name] = np.asarray(control, dtype=float)
        self.control_residual_history[agent.name].append(
            control_position_drift.copy()
        )
        return self.previous_controls[agent.name]

    def observe(
        self,
        *,
        agent: HarborAgent,
        state: np.ndarray,
        step: int,
        dt: float,
    ) -> None:
        """Update local estimators at plant rate without forcing an NLP solve."""
        if not np.isclose(dt, self.dt):
            raise ValueError("observer and MPC sample times must match")
        if self.previous_update_steps.get(agent.name) == step:
            return
        self._update_model_estimates(agent, state, step)
        self.previous_states[agent.name] = np.asarray(state, dtype=float).copy()
        self.previous_update_steps[agent.name] = step
        self.residual_history[agent.name].append(
            self.position_drift_estimates[agent.name].copy()
        )
        self.effectiveness_history[agent.name].append(
            self.control_effectiveness_estimates[agent.name].copy()
        )
        self.raw_effectiveness_history[agent.name].append(
            self.raw_effectiveness_estimates[agent.name].copy()
        )
        self.excitation_history[agent.name].append(
            self.excitation_energy[agent.name].copy()
        )
        self.information_std_history[agent.name].append(
            _posterior_effectiveness_std(
                self.information_matrices[agent.name], self.config
            )
        )

    def _update_model_estimates(
        self, agent: HarborAgent, current_state: np.ndarray, current_step: int
    ) -> None:
        """Update local actuator and position-residual estimates in sequence."""
        if agent.name not in self.previous_states:
            return
        previous_state = self.previous_states[agent.name]
        previous_control = self.previous_controls[agent.name]
        elapsed_steps = current_step - self.previous_update_steps[agent.name]
        if elapsed_steps <= 0:
            raise ValueError("MPC estimator updates must advance in time")
        elapsed_dt = elapsed_steps * self.dt
        normalized = previous_control / agent.model.control_scale()
        self.excitation_energy[agent.name] += normalized * normalized
        recursive_prior = (
            self.raw_effectiveness_estimates[agent.name]
            if self.config.effectiveness_recovery_prior_mode == "transient"
            else self.control_effectiveness_estimates[agent.name]
        )
        normalized_sensitivity = _effectiveness_sensitivity_matrix(
            agent.model,
            previous_state,
            previous_control,
            elapsed_dt,
            recursive_prior,
            self.config,
            normalize_dynamic_state=True,
        )
        jacobian = (
            normalized_sensitivity / self.config.identification_measurement_noise
        )
        self.information_matrices[agent.name] += jacobian.T @ jacobian
        if self.config.control_effectiveness_adaptation:
            if self.config.effectiveness_estimator_mode == "recursive_diagonal":
                previous_effectiveness = recursive_prior.copy()
                allow_change = (
                    current_step >= self.config.effectiveness_rls_change_warmup_steps
                    and current_step
                    - self.effectiveness_last_change_step_by_agent[agent.name]
                    >= self.config.effectiveness_rls_change_cooldown_steps
                )
                estimate, covariance, change_detected, evidence = (
                    _recursive_diagonal_effectiveness_update(
                        agent.model,
                        previous_state,
                        previous_control,
                        np.asarray(current_state, dtype=float),
                        elapsed_dt,
                        recursive_prior,
                        self.effectiveness_covariances[agent.name],
                        self.config,
                        change_evidence=(
                            self.effectiveness_change_evidence_by_agent[agent.name]
                        ),
                        allow_change=allow_change,
                    )
                )
                self.effectiveness_change_evidence_by_agent[agent.name] = evidence
                if change_detected:
                    self.effectiveness_change_steps_by_agent[agent.name].append(
                        current_step
                    )
                    self.effectiveness_last_change_step_by_agent[agent.name] = (
                        current_step
                    )
                    loss_detected = _effectiveness_change_is_loss(
                        previous_effectiveness, estimate
                    )
                    recovery_detected = _effectiveness_change_is_recovery(
                        previous_effectiveness,
                        estimate,
                        self.config.effectiveness_recovery_direction_tolerance,
                    )
                    if (
                        recovery_detected
                        and self.config.effectiveness_recovery_prior_mode == "embedded"
                    ):
                        estimate = _apply_nominal_recovery_prior(
                            previous_effectiveness,
                            estimate,
                            self.config.effectiveness_recovery_prior_gain,
                            self.config.effectiveness_recovery_direction_tolerance,
                            self.config.effectiveness_min,
                            self.config.effectiveness_max,
                        )
                    if self.config.effectiveness_recovery_prior_mode == "transient":
                        if loss_detected:
                            self.recovery_effectiveness_offsets[agent.name].fill(0.0)
                            episode_count = len(
                                self.recovery_offset_steps_by_agent[agent.name]
                            )
                            episode_limit = (
                                self.config.effectiveness_recovery_max_episodes_per_agent
                            )
                            within_budget = (
                                episode_limit == 0 or episode_count < episode_limit
                            )
                            if (
                                within_budget
                                and not self.recovery_offset_armed_by_agent[agent.name]
                            ):
                                self.recovery_offset_armed_by_agent[agent.name] = True
                                self.recovery_offset_armed_step_by_agent[agent.name] = (
                                    current_step
                                )
                            if not within_budget:
                                self.recovery_offset_armed_by_agent[agent.name] = False
                                self.recovery_offset_armed_step_by_agent[agent.name] = None
                                self.recovery_offset_episode_limit_rejections_by_agent[
                                    agent.name
                                ] += 1
                        elif recovery_detected:
                            armed = self.recovery_offset_armed_by_agent[agent.name]
                            armed_step = self.recovery_offset_armed_step_by_agent[
                                agent.name
                            ]
                            if not armed:
                                self.recovery_offset_unarmed_rejections_by_agent[
                                    agent.name
                                ] += 1
                            elif (
                                armed_step is not None
                                and current_step - armed_step
                                < self.config.effectiveness_recovery_minimum_dwell_steps
                            ):
                                self.recovery_offset_dwell_rejections_by_agent[
                                    agent.name
                                ] += 1
                            else:
                                identifiable = (
                                    not self.config.effectiveness_recovery_require_full_rank
                                    or _has_full_column_rank(
                                        normalized_sensitivity,
                                        self.config.effectiveness_recovery_rank_tolerance,
                                    )
                                )
                                self.recovery_offset_armed_by_agent[agent.name] = False
                                self.recovery_offset_armed_step_by_agent[agent.name] = None
                                if identifiable:
                                    self.recovery_effectiveness_offsets[agent.name] = (
                                        _channel_selective_recovery_offset(
                                            previous_effectiveness,
                                            estimate,
                                            self.config.effectiveness_recovery_prior_gain,
                                            self.config.effectiveness_recovery_direction_tolerance,
                                        )
                                    )
                                    self.recovery_offset_steps_by_agent[agent.name].append(
                                        current_step
                                    )
                                else:
                                    self.recovery_effectiveness_offsets[agent.name].fill(
                                        0.0
                                    )
                                    self.recovery_offset_rejections_by_agent[
                                        agent.name
                                    ] += 1
                    permit_identification = (
                        not self.config.identification_arm_on_loss_only
                        or loss_detected
                    )
                    if (
                        self.config.identification_reset_on_change
                        and permit_identification
                    ):
                        self.excitation_energy[agent.name].fill(0.0)
                        self.information_matrices[agent.name].fill(0.0)
                        self.identification_probe_quota_counts[agent.name].fill(0)
                        self.identification_probe_rejection_counts[agent.name].fill(0)
                    if (
                        self.config.identification_arm_on_change
                        and permit_identification
                    ):
                        self.identification_change_armed_by_agent[agent.name] = True
                if self.config.effectiveness_recovery_prior_mode == "transient":
                    self.raw_effectiveness_estimates[agent.name] = estimate
                    offset = self.recovery_effectiveness_offsets[agent.name]
                    self.control_effectiveness_estimates[agent.name] = np.clip(
                        estimate + offset,
                        self.config.effectiveness_min,
                        self.config.effectiveness_max,
                    )
                    self.recovery_effectiveness_offsets[agent.name] = (
                        self.config.effectiveness_recovery_offset_decay * offset
                    )
                else:
                    self.control_effectiveness_estimates[agent.name] = estimate
                    self.raw_effectiveness_estimates[agent.name] = estimate.copy()
                self.effectiveness_covariances[agent.name] = covariance
            else:
                self.control_effectiveness_estimates[agent.name] = (
                    _estimate_effectiveness_vector(
                        agent.model,
                        previous_state,
                        previous_control,
                        np.asarray(current_state, dtype=float),
                        elapsed_dt,
                        self.control_effectiveness_estimates[agent.name],
                        self.config,
                    )
                )
                self.raw_effectiveness_estimates[agent.name] = (
                    self.control_effectiveness_estimates[agent.name].copy()
                )
        if (
            not self.config.residual_adaptation
            or agent.model.kind not in self.config.residual_adaptation_kinds
        ):
            return
        effectiveness = (
            self.raw_effectiveness_estimates[agent.name]
            if self.config.effectiveness_recovery_prior_mode == "transient"
            else self.control_effectiveness_estimates[agent.name]
        )
        if self.config.residual_measurement_source == "kinematic_velocity":
            measured_velocity_error = _kinematic_position_drift_measurement(
                agent.model,
                previous_state,
                current_state,
                elapsed_dt,
                elapsed_steps,
            )
        else:
            predicted = agent.model.step(
                previous_state,
                effectiveness * previous_control,
                elapsed_dt,
            )
            measured_velocity_error = (
                agent.model.position(current_state)
                - agent.model.position(predicted)
            ) / elapsed_dt
        if self.config.residual_estimator_mode == "constant_bias_rls":
            elapsed_since_change = (
                current_step
                - self.effectiveness_last_change_step_by_agent[agent.name]
            )
            if (
                0 <= elapsed_since_change
                < self.config.residual_change_holdoff_steps
            ):
                self.position_drift_variances[agent.name] += (
                    self.config.residual_rls_process_noise
                )
                return
            estimate, variance, rejected = _constant_bias_residual_update(
                self.position_drift_estimates[agent.name],
                self.position_drift_variances[agent.name],
                measured_velocity_error,
                self.config,
            )
            self.position_drift_variances[agent.name] = variance
            self.residual_rejection_count_by_agent[agent.name] += rejected
        else:
            gain = self.config.residual_estimator_gain
            estimate = (
                (1.0 - gain) * self.position_drift_estimates[agent.name]
                + gain * measured_velocity_error
            )
        norm = float(np.linalg.norm(estimate))
        if norm > self.config.residual_max_speed:
            estimate *= self.config.residual_max_speed / norm
        self.position_drift_estimates[agent.name] = estimate

    def _identification_probe(
        self,
        agent: HarborAgent,
        state: np.ndarray,
        desired_velocity: np.ndarray,
        inbox: dict[str, AgentMessage],
        step: int,
    ) -> tuple[np.ndarray, np.ndarray, int | None]:
        """Select one direct, constraint-aware local identification pulse."""
        model = agent.model
        empty = np.zeros(model.control_dim, dtype=float)
        if (
            not self.config.active_identification
            or not self.config.control_effectiveness_adaptation
            or not self.identification_change_armed_by_agent[agent.name]
            or self.config.effectiveness_estimator_mode
            not in {"diagonal", "recursive_diagonal"}
            or np.linalg.norm(desired_velocity) <= 1e-6
            or step % self.config.identification_probe_interval_steps != 0
        ):
            return empty, empty.copy(), None
        own_position = model.position(np.asarray(state, dtype=float))
        for other_name, message in inbox.items():
            other = self.agents.get(other_name)
            if other is None:
                continue
            clearance = float(np.linalg.norm(own_position - message.position))
            required = (
                agent.radius
                + other.radius
                + self.config.collision_buffer
                + self.config.identification_clearance_buffer
            )
            if clearance <= required:
                return empty, empty.copy(), None
        if self.config.identification_strategy == "information":
            increments = _candidate_probe_information(
                model,
                np.asarray(state, dtype=float),
                self.control_effectiveness_estimates[agent.name],
                self.dt,
                self.config,
            )
            channel = _information_identification_channel(
                self.information_matrices[agent.name],
                increments,
                self.identification_probe_quota_counts[agent.name],
                self.identification_probe_rejection_counts[agent.name],
                self.config,
                effectiveness_estimate=self.control_effectiveness_estimates[
                    agent.name
                ],
            )
        else:
            channel = _identification_channel(
                self.excitation_energy[agent.name],
                self.identification_probe_quota_counts[agent.name],
                self.identification_probe_rejection_counts[agent.name],
                self.config,
            )
        if channel is None:
            return empty, empty.copy(), None
        count = self.identification_probe_channel_counts[agent.name][channel]
        sign = 1.0 if count % 2 == 0 else -1.0
        probe = empty.copy()
        probe[channel] = (
            sign
            * self.config.identification_probe_fraction
            * model.control_scale()[channel]
        )
        mask = empty.copy()
        mask[channel] = 1.0
        return probe, mask, channel


    def _obstacle_predictions(
        self,
        agent: HarborAgent,
        inbox: dict[str, AgentMessage],
        current_step: int,
    ) -> dict[str, np.ndarray]:
        horizon = self.config.prediction_horizon
        predictions = {}
        for other in self.optimizers[agent.name].other_agents:
            message = inbox.get(other.name)
            if message is None:
                predictions[other.name] = np.full((horizon, 3), 1.0e3)
                continue
            age = max(0, current_step - message.sent_step)
            times = (age + np.arange(1, horizon + 1))[:, None] * self.dt
            if self.config.obstacle_prediction_mode == "goal_bounded_velocity":
                predictions[other.name] = _goal_bounded_velocity_prediction(
                    message,
                    times,
                    self.config.obstacle_prediction_alignment_threshold,
                )
            else:
                predictions[other.name] = message.position + times * message.velocity
        return predictions


def _goal_bounded_velocity_prediction(
    message: AgentMessage,
    times: np.ndarray,
    alignment_threshold: float,
) -> np.ndarray:
    """Extrapolate velocity without carrying an aligned agent past its intent."""
    position = np.asarray(message.position, dtype=float).reshape(3)
    velocity = np.asarray(message.velocity, dtype=float).reshape(3)
    prediction_times = np.asarray(times, dtype=float).reshape(-1, 1)
    speed = float(np.linalg.norm(velocity))
    goal_delta = np.asarray(message.goal, dtype=float).reshape(3) - position
    goal_distance = float(np.linalg.norm(goal_delta))
    if speed <= 1e-9 or goal_distance <= 1e-9:
        return np.repeat(position[None, :], len(prediction_times), axis=0)
    direction = velocity / speed
    alignment = float(np.dot(direction, goal_delta / goal_distance))
    if alignment < alignment_threshold:
        return position + prediction_times * velocity
    along_track_remaining = max(float(np.dot(goal_delta, direction)), 0.0)
    travel = np.minimum(speed * prediction_times, along_track_remaining)
    return position + travel * direction


def _estimate_control_effectiveness(
    model: PlatformModel,
    previous_state: np.ndarray,
    previous_control: np.ndarray,
    current_state: np.ndarray,
    dt: float,
    prior: float,
    config: HarborMPCConfig,
) -> float:
    """Fit scalar actuator effectiveness from locally measured dynamic states."""
    command = np.asarray(previous_control, dtype=float)
    normalized_excitation = float(np.linalg.norm(command / model.control_scale()))
    if normalized_excitation < config.effectiveness_excitation_threshold:
        return prior
    epsilon = 0.05
    lower = max(config.effectiveness_min, prior - epsilon)
    upper = min(config.effectiveness_max, prior + epsilon)
    if upper - lower <= np.finfo(float).eps:
        return prior
    dynamic = slice(model.pose_dim, model.state_dim)
    low_state = model.step(previous_state, lower * command, dt)[dynamic]
    high_state = model.step(previous_state, upper * command, dt)[dynamic]
    sensitivity = (high_state - low_state) / (upper - lower)
    denominator = float(np.dot(sensitivity, sensitivity))
    if denominator <= 1e-12:
        return prior
    predicted = model.step(previous_state, prior * command, dt)[dynamic]
    measured = np.asarray(current_state, dtype=float)[dynamic]
    instantaneous = prior + float(np.dot(sensitivity, measured - predicted)) / denominator
    instantaneous = float(
        np.clip(
            instantaneous,
            config.effectiveness_min,
            config.effectiveness_max,
        )
    )
    gain = config.effectiveness_estimator_gain
    return float((1.0 - gain) * prior + gain * instantaneous)


def _effectiveness_change_is_loss(
    previous: np.ndarray, updated: np.ndarray, tolerance: float = 1.0e-6
) -> bool:
    """Classify a detected aggregate actuator change without plant truth."""
    previous = np.asarray(previous, dtype=float)
    updated = np.asarray(updated, dtype=float)
    return bool(np.mean(updated - previous) < -tolerance)


def _effectiveness_change_is_recovery(
    previous: np.ndarray, updated: np.ndarray, tolerance: float = 1.0e-6
) -> bool:
    """Classify an aggregate positive actuator change without plant truth."""
    previous = np.asarray(previous, dtype=float)
    updated = np.asarray(updated, dtype=float)
    return bool(np.mean(updated - previous) > tolerance)


def _apply_nominal_recovery_prior(
    previous: np.ndarray,
    updated: np.ndarray,
    gain: float,
    direction_tolerance: float,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Pull only positively changing actuator channels toward nominal health."""
    previous = np.asarray(previous, dtype=float)
    result = np.asarray(updated, dtype=float).copy()
    result += _channel_selective_recovery_offset(
        previous, result, gain, direction_tolerance
    )
    return np.clip(result, lower, upper)


def _channel_selective_recovery_offset(
    previous: np.ndarray,
    updated: np.ndarray,
    gain: float,
    direction_tolerance: float,
) -> np.ndarray:
    """Return a nominal-health offset only for positively changing channels."""
    previous = np.asarray(previous, dtype=float)
    updated = np.asarray(updated, dtype=float)
    offset = np.zeros_like(updated)
    recovering = updated - previous > direction_tolerance
    offset[recovering] = gain * (1.0 - updated[recovering])
    return offset


def _has_full_column_rank(matrix: np.ndarray, tolerance: float) -> bool:
    """Return whether local measurements identify every actuator direction."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return False
    return bool(np.linalg.matrix_rank(matrix, tol=tolerance) == matrix.shape[1])


def _dynamic_state_scale(model: PlatformModel) -> np.ndarray:
    """Return characteristic velocity/rate scales for local information."""
    if model.kind == "ugv":
        values = [model.max_speed]
        if model.state_dim - model.pose_dim > 1:
            values.append(model.max_yaw_rate)
    elif model.kind == "usv":
        values = [model.max_speed]
        if model.state_dim - model.pose_dim > 1:
            values.extend((model.max_sway_speed, model.max_yaw_rate))
    else:
        values = [
            model.max_horizontal_speed,
            model.max_horizontal_speed,
            model.max_vertical_speed,
            model.max_angular_rate,
            model.max_angular_rate,
            model.max_angular_rate,
        ]
    dynamic_dim = model.state_dim - model.pose_dim
    return np.maximum(np.asarray(values[:dynamic_dim], dtype=float), 1e-9)


def _effectiveness_sensitivity_matrix(
    model: PlatformModel,
    previous_state: np.ndarray,
    previous_control: np.ndarray,
    dt: float,
    prior: np.ndarray,
    config: HarborMPCConfig,
    *,
    normalize_dynamic_state: bool,
) -> np.ndarray:
    """Linearize one-step dynamic response with respect to actuator gains."""
    command = np.asarray(previous_control, dtype=float).reshape(model.control_dim)
    prior = np.asarray(prior, dtype=float).reshape(model.control_dim)
    dynamic = slice(model.pose_dim, model.state_dim)
    sensitivity = np.zeros(
        (model.state_dim - model.pose_dim, model.control_dim), dtype=float
    )
    epsilon = 0.05
    for channel in np.flatnonzero(np.abs(command) > 1e-12):
        lower = prior.copy()
        upper = prior.copy()
        lower[channel] = max(config.effectiveness_min, prior[channel] - epsilon)
        upper[channel] = min(config.effectiveness_max, prior[channel] + epsilon)
        span = upper[channel] - lower[channel]
        if span <= np.finfo(float).eps:
            continue
        low_state = model.step(previous_state, lower * command, dt)[dynamic]
        high_state = model.step(previous_state, upper * command, dt)[dynamic]
        sensitivity[:, channel] = (high_state - low_state) / span
    if normalize_dynamic_state:
        sensitivity /= _dynamic_state_scale(model)[:, None]
    return sensitivity


def _recursive_diagonal_effectiveness_update(
    model: PlatformModel,
    previous_state: np.ndarray,
    previous_control: np.ndarray,
    current_state: np.ndarray,
    dt: float,
    prior: np.ndarray,
    covariance: np.ndarray,
    config: HarborMPCConfig,
    *,
    change_evidence: float = 0.0,
    allow_change: bool = True,
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    """Update per-actuator gains with robust covariance-form recursive LS."""
    prior = np.asarray(prior, dtype=float).reshape(model.control_dim)
    covariance = np.asarray(covariance, dtype=float).reshape(
        model.control_dim, model.control_dim
    )
    dynamic = slice(model.pose_dim, model.state_dim)
    scale = _dynamic_state_scale(model)
    predicted = model.step(
        previous_state, prior * np.asarray(previous_control, dtype=float), dt
    )[dynamic]
    innovation = (np.asarray(current_state, dtype=float)[dynamic] - predicted) / scale
    sensitivity = _effectiveness_sensitivity_matrix(
        model,
        previous_state,
        previous_control,
        dt,
        prior,
        config,
        normalize_dynamic_state=True,
    )

    identity = np.eye(model.control_dim)
    predicted_covariance = (
        covariance / config.effectiveness_rls_forgetting_factor
        + config.effectiveness_rls_process_noise * identity
    )
    measurement_covariance = (
        config.effectiveness_rls_measurement_noise**2
        * np.eye(model.state_dim - model.pose_dim)
    )

    def innovation_statistics(candidate_covariance: np.ndarray):
        innovation_covariance = (
            sensitivity @ candidate_covariance @ sensitivity.T
            + measurement_covariance
        )
        inverse = np.linalg.pinv(innovation_covariance, hermitian=True)
        distance = float(
            np.sqrt(max(float(innovation.T @ inverse @ innovation), 0.0))
        )
        return inverse, distance

    inverse_innovation, mahalanobis = innovation_statistics(predicted_covariance)
    excited = bool(
        np.linalg.norm(previous_control / model.control_scale())
        >= config.effectiveness_excitation_threshold
    )
    change_detected = False
    if not config.effectiveness_rls_adaptive_covariance:
        change_evidence = 0.0
    elif config.effectiveness_rls_change_detector == "threshold":
        change_candidate = bool(
            excited and mahalanobis > config.effectiveness_rls_change_threshold
        )
        change_evidence = change_evidence + 1.0 if change_candidate else 0.0
        change_detected = bool(
            change_candidate
            and change_evidence >= config.effectiveness_rls_change_persistence
            and allow_change
        )
    elif allow_change:
        normalized_innovation_squared = mahalanobis**2 / len(innovation)
        change_evidence = (
            max(
                0.0,
                change_evidence
                + normalized_innovation_squared
                - config.effectiveness_rls_cusum_drift,
            )
            if excited
            else 0.0
        )
        change_detected = bool(
            change_evidence >= config.effectiveness_rls_cusum_threshold
        )
    else:
        change_evidence = 0.0
    if change_detected:
        change_evidence = 0.0
        predicted_covariance *= config.effectiveness_rls_covariance_inflation
        inverse_innovation, mahalanobis = innovation_statistics(predicted_covariance)
    if mahalanobis > config.effectiveness_rls_innovation_gate:
        innovation *= config.effectiveness_rls_innovation_gate / mahalanobis

    gain = predicted_covariance @ sensitivity.T @ inverse_innovation
    estimate = np.clip(
        prior + gain @ innovation,
        config.effectiveness_min,
        config.effectiveness_max,
    )
    correction = identity - gain @ sensitivity
    updated_covariance = (
        correction @ predicted_covariance @ correction.T
        + gain @ measurement_covariance @ gain.T
    )
    updated_covariance = 0.5 * (updated_covariance + updated_covariance.T)
    return estimate, updated_covariance, change_detected, change_evidence


def _posterior_effectiveness_std(
    information_matrix: np.ndarray, config: HarborMPCConfig
) -> np.ndarray:
    """Approximate local actuator-gain posterior standard deviations."""
    information = np.asarray(information_matrix, dtype=float)
    precision = information + np.eye(len(information)) / (
        config.identification_prior_std**2
    )
    covariance = np.linalg.pinv(precision, hermitian=True)
    return np.sqrt(np.maximum(np.diag(covariance), 0.0))


def _candidate_probe_information(
    model: PlatformModel,
    state: np.ndarray,
    prior: np.ndarray,
    dt: float,
    config: HarborMPCConfig,
) -> np.ndarray:
    """Predict Fisher-information increments for isolated channel probes."""
    increments = np.zeros(
        (model.control_dim, model.control_dim, model.control_dim), dtype=float
    )
    for channel in range(model.control_dim):
        command = np.zeros(model.control_dim, dtype=float)
        command[channel] = (
            config.identification_probe_fraction * model.control_scale()[channel]
        )
        sensitivity = _effectiveness_sensitivity_matrix(
            model,
            state,
            command,
            dt,
            prior,
            config,
            normalize_dynamic_state=True,
        )
        jacobian = sensitivity / config.identification_measurement_noise
        increments[channel] = jacobian.T @ jacobian
    return increments


def _information_identification_channel(
    information_matrix: np.ndarray,
    candidate_increments: np.ndarray,
    probe_counts: np.ndarray,
    rejection_counts: np.ndarray,
    config: HarborMPCConfig,
    effectiveness_estimate: np.ndarray | None = None,
) -> int | None:
    """Select a safe probe by fault-focused expected information gain."""
    information = np.asarray(information_matrix, dtype=float)
    dimension = len(information)
    increments = np.asarray(candidate_increments, dtype=float)
    if information.shape != (dimension, dimension) or increments.shape != (
        dimension,
        dimension,
        dimension,
    ):
        raise ValueError("actuator information matrices have inconsistent shapes")
    counts = np.asarray(probe_counts, dtype=int).reshape(dimension)
    rejections = np.asarray(rejection_counts, dtype=int).reshape(dimension)
    available = rejections < config.identification_max_rejections
    posterior_std = _posterior_effectiveness_std(information, config)
    estimate = (
        np.ones(dimension, dtype=float)
        if effectiveness_estimate is None
        else np.asarray(effectiveness_estimate, dtype=float).reshape(dimension)
    )
    fault_evidence = np.abs(1.0 - estimate) / config.identification_prior_std
    calibration = np.flatnonzero(
        available & (counts < config.identification_min_probes_per_channel)
    )
    if len(calibration):
        minimum_count = int(np.min(counts[calibration]))
        candidates = calibration[counts[calibration] == minimum_count]
    else:
        candidates = np.flatnonzero(
            available & (posterior_std > config.identification_target_std)
        )
    if len(candidates) == 0:
        return None
    precision = information + np.eye(dimension) / (
        config.identification_prior_std**2
    )
    base_sign, base_logdet = np.linalg.slogdet(precision)
    if base_sign <= 0.0:
        raise ValueError("actuator information precision must be positive definite")
    scores = []
    for channel in candidates:
        sign, logdet = np.linalg.slogdet(precision + increments[channel])
        gain = -np.inf if sign <= 0.0 else logdet - base_logdet
        focus = 1.0 + config.identification_fault_focus_weight * fault_evidence[
            channel
        ]
        scores.append(gain * posterior_std[channel] * focus)
    return int(candidates[int(np.argmax(scores))])


def _estimate_effectiveness_vector(
    model: PlatformModel,
    previous_state: np.ndarray,
    previous_control: np.ndarray,
    current_state: np.ndarray,
    dt: float,
    prior: np.ndarray,
    config: HarborMPCConfig,
) -> np.ndarray:
    """Estimate scalar-shared or diagonal local actuator effectiveness."""
    prior_vector = np.asarray(prior, dtype=float).reshape(model.control_dim)
    if config.effectiveness_estimator_mode == "scalar":
        estimate = _estimate_control_effectiveness(
            model,
            previous_state,
            previous_control,
            current_state,
            dt,
            float(np.mean(prior_vector)),
            config,
        )
        return np.full(model.control_dim, estimate, dtype=float)
    return _estimate_diagonal_control_effectiveness(
        model,
        previous_state,
        previous_control,
        current_state,
        dt,
        prior_vector,
        config,
    )


def _least_excited_channel(
    excitation_energy: np.ndarray, target_energy: float
) -> int | None:
    """Return the least-observed channel below a common information target."""
    energy = np.asarray(excitation_energy, dtype=float).reshape(-1)
    candidates = np.flatnonzero(energy < target_energy)
    if len(candidates) == 0:
        return None
    return int(candidates[np.argmin(energy[candidates])])


def _identification_channel(
    excitation_energy: np.ndarray,
    probe_counts: np.ndarray,
    rejection_counts: np.ndarray,
    config: HarborMPCConfig,
) -> int | None:
    """Prioritize calibration quotas, then remaining low-energy channels."""
    energy = np.asarray(excitation_energy, dtype=float).reshape(-1)
    counts = np.asarray(probe_counts, dtype=int).reshape(energy.shape)
    rejections = np.asarray(rejection_counts, dtype=int).reshape(energy.shape)
    available = rejections < config.identification_max_rejections
    calibration = np.flatnonzero(
        available & (counts < config.identification_min_probes_per_channel)
    )
    if len(calibration):
        ordering = np.lexsort((energy[calibration], counts[calibration]))
        return int(calibration[ordering[0]])
    eligible_energy = energy.copy()
    eligible_energy[~available] = config.identification_target_energy
    return _least_excited_channel(
        eligible_energy,
        config.identification_target_energy,
    )


def _estimate_diagonal_control_effectiveness(
    model: PlatformModel,
    previous_state: np.ndarray,
    previous_control: np.ndarray,
    current_state: np.ndarray,
    dt: float,
    prior: np.ndarray,
    config: HarborMPCConfig,
) -> np.ndarray:
    """Fit independent channel gains from one-step local dynamic response."""
    command = np.asarray(previous_control, dtype=float).reshape(model.control_dim)
    scales = np.asarray(model.control_scale(), dtype=float)
    active = np.abs(command / scales) >= config.effectiveness_excitation_threshold
    if not np.any(active):
        return np.asarray(prior, dtype=float).copy()
    prior = np.asarray(prior, dtype=float).reshape(model.control_dim)
    dynamic = slice(model.pose_dim, model.state_dim)
    predicted = model.step(previous_state, prior * command, dt)[dynamic]
    measured = np.asarray(current_state, dtype=float)[dynamic]
    active_indices = np.flatnonzero(active)
    sensitivity = np.zeros((len(predicted), len(active_indices)), dtype=float)
    epsilon = 0.05
    for column, channel in enumerate(active_indices):
        lower = prior.copy()
        upper = prior.copy()
        lower[channel] = max(config.effectiveness_min, prior[channel] - epsilon)
        upper[channel] = min(config.effectiveness_max, prior[channel] + epsilon)
        span = upper[channel] - lower[channel]
        if span <= np.finfo(float).eps:
            continue
        low_state = model.step(previous_state, lower * command, dt)[dynamic]
        high_state = model.step(previous_state, upper * command, dt)[dynamic]
        sensitivity[:, column] = (high_state - low_state) / span
    if float(np.linalg.norm(sensitivity)) <= 1e-12:
        return prior.copy()
    correction = np.linalg.lstsq(
        sensitivity,
        measured - predicted,
        rcond=1e-8,
    )[0]
    instantaneous = prior.copy()
    instantaneous[active_indices] += correction
    instantaneous = np.clip(
        instantaneous,
        config.effectiveness_min,
        config.effectiveness_max,
    )
    gain = config.effectiveness_estimator_gain
    return (1.0 - gain) * prior + gain * instantaneous


def _constant_bias_residual_update(
    prior: np.ndarray,
    variance: np.ndarray,
    measurement: np.ndarray,
    config: HarborMPCConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Update independent constant-current states with innovation gating."""
    estimate = np.asarray(prior, dtype=float).copy()
    predicted_variance = (
        np.asarray(variance, dtype=float)
        + config.residual_rls_process_noise
    )
    innovation = np.asarray(measurement, dtype=float) - estimate
    innovation_variance = (
        predicted_variance + config.residual_rls_measurement_noise**2
    )
    normalized = np.abs(innovation) / np.sqrt(innovation_variance)
    accepted = normalized <= config.residual_rls_innovation_gate
    gain = predicted_variance / innovation_variance
    estimate[accepted] += gain[accepted] * innovation[accepted]
    posterior_variance = predicted_variance.copy()
    posterior_variance[accepted] *= 1.0 - gain[accepted]
    return estimate, posterior_variance, int(np.count_nonzero(~accepted))


def _kinematic_position_drift_measurement(
    model: PlatformModel,
    previous_state: np.ndarray,
    current_state: np.ndarray,
    dt: float,
    elapsed_steps: int,
) -> np.ndarray:
    """Infer world-frame drift without using controls or actuator estimates."""
    prior = np.asarray(previous_state, dtype=float)
    current = np.asarray(current_state, dtype=float)
    ground_velocity = (
        model.position(current) - model.position(prior)
    ) / dt
    if elapsed_steps == 1:
        velocity_state = current.copy()
        if model.kind == "usv":
            velocity_state[2] = prior[2]
        elif model.kind == "rov":
            velocity_state[3:6] = prior[3:6]
        through_water_velocity = model.velocity(velocity_state)
    else:
        through_water_velocity = 0.5 * (
            model.velocity(prior) + model.velocity(current)
        )
    return ground_velocity - through_water_velocity


def _approach_pose_goal(
    model: PlatformModel,
    navigation_goal: np.ndarray,
    desired_velocity: np.ndarray,
) -> np.ndarray:
    """Use line-of-sight yaw while a planar underactuated agent is moving."""
    goal = np.asarray(navigation_goal, dtype=float).copy()
    desired = np.asarray(desired_velocity, dtype=float)
    if model.kind in {"ugv", "usv"} and np.linalg.norm(desired[:2]) > 1e-6:
        goal[2] = np.arctan2(desired[1], desired[0])
    return goal


def _control_position_drift(
    model: PlatformModel,
    state: np.ndarray,
    estimate: np.ndarray,
    projection: str,
    desired_velocity: np.ndarray | None = None,
) -> np.ndarray:
    """Project USV drift onto its surge-controllable local subspace."""
    drift = np.asarray(estimate, dtype=float).copy()
    moving = (
        desired_velocity is not None
        and np.linalg.norm(np.asarray(desired_velocity, dtype=float)[:2]) > 1.0e-6
    )
    if (
        projection == "full"
        or model.kind != "usv"
        or (projection == "station_keeping_subspace" and moving)
    ):
        return drift
    yaw = float(np.asarray(state, dtype=float)[2])
    surge = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    return float(np.dot(drift, surge)) * surge


def _nearest_reference_index(
    model: PlatformModel, states: np.ndarray, current_state: np.ndarray
) -> int:
    positions = np.asarray([model.position(state) for state in states])
    current = model.position(current_state)
    return int(np.argmin(np.linalg.norm(positions - current, axis=1)))


def _sample_safe_states(
    states: np.ndarray, target_index: int, sample_count: int
) -> tuple[np.ndarray, np.ndarray]:
    trajectory = np.asarray(states, dtype=float)
    indices = np.minimum(
        np.arange(target_index, target_index + sample_count), len(trajectory) - 1
    )
    costs = (len(trajectory) - 1 - indices).astype(float)
    return trajectory[indices], costs


def _warm_reference(
    states: np.ndarray,
    controls: np.ndarray,
    start_index: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    state_indices = np.minimum(
        np.arange(start_index, start_index + horizon + 1), len(states) - 1
    )
    if len(controls):
        control_indices = np.minimum(
            np.arange(start_index, start_index + horizon), len(controls) - 1
        )
        warm_controls = np.asarray(controls, dtype=float)[control_indices]
    else:
        warm_controls = np.zeros((horizon, 0))
    return np.asarray(states, dtype=float)[state_indices], warm_controls


def _symbolic_step(ca, model: PlatformModel, state, control, dt: float):
    return model.symbolic_step(ca, state, control, dt)


def _apply_symbolic_position_drift(ca, model, state, drift, dt: float):
    """Add the estimated world-velocity residual to predicted position only."""
    dimensions = 3 if model.kind == "rov" else 2
    return ca.vertcat(state[:dimensions] + dt * drift[:dimensions], state[dimensions:])


def _symbolic_position(ca, model: PlatformModel, state):
    if model.kind == "ugv":
        return ca.vertcat(state[0], state[1], model.ground_z)
    if model.kind == "usv":
        return ca.vertcat(state[0], state[1], model.surface_z)
    return state[:3]


def _symbolic_goal_position(ca, model: PlatformModel, goal):
    if model.kind == "ugv":
        return ca.vertcat(goal[0], goal[1], model.ground_z)
    if model.kind == "usv":
        return ca.vertcat(goal[0], goal[1], model.surface_z)
    return goal[:3]


def _symbolic_velocity(model: PlatformModel, state):
    return model.symbolic_velocity(None, state)


def _symbolic_orientation_cost(ca, model: PlatformModel, state, goal):
    if model.pose_dim == 3:
        return 2.0 * (1.0 - ca.cos(state[2] - goal[2]))
    return sum(2.0 * (1.0 - ca.cos(state[index] - goal[index])) for index in range(3, 6))
