"""Distributed receding-horizon controllers for heterogeneous harbor agents."""

from __future__ import annotations

from dataclasses import dataclass, fields
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
    collision_slack_weight: float = 100000.0
    collision_slack_bound: float = 0.0
    residual_adaptation: bool = False
    residual_estimator_gain: float = 0.35
    residual_max_speed: float = 0.35
    control_effectiveness_adaptation: bool = False
    effectiveness_estimator_mode: str = "scalar"
    effectiveness_estimator_gain: float = 0.25
    effectiveness_rls_forgetting_factor: float = 0.995
    effectiveness_rls_measurement_noise: float = 0.02
    effectiveness_rls_process_noise: float = 1.0e-5
    effectiveness_rls_innovation_gate: float = 3.0
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
    identification_probe_interval_steps: int = 1
    identification_min_probes_per_channel: int = 2
    identification_max_rejections: int = 2
    identification_clearance_buffer: float = 0.4
    obstacle_prediction_mode: str = "goal_bounded_velocity"
    obstacle_prediction_alignment_threshold: float = 0.5
    ipopt_max_iter: int = 120
    ipopt_print_level: int = 0

    def __post_init__(self) -> None:
        integer_values = {
            "prediction_horizon": self.prediction_horizon,
            "replan_interval_steps": self.replan_interval_steps,
            "learning_iterations": self.learning_iterations,
            "terminal_samples": self.terminal_samples,
            "identification_probe_interval_steps": self.identification_probe_interval_steps,
            "identification_min_probes_per_channel": self.identification_min_probes_per_channel,
            "identification_max_rejections": self.identification_max_rejections,
            "ipopt_max_iter": self.ipopt_max_iter,
        }
        if any(value <= 0 for value in integer_values.values()):
            raise ValueError("harbor_mpc integer settings must be positive")
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
            self.collision_slack_weight,
            self.collision_slack_bound,
            self.residual_estimator_gain,
            self.residual_max_speed,
            self.effectiveness_estimator_gain,
            self.effectiveness_rls_measurement_noise,
            self.effectiveness_rls_process_noise,
            self.effectiveness_rls_innovation_gate,
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
        if self.effectiveness_estimator_gain > 1.0:
            raise ValueError("effectiveness_estimator_gain must be in [0, 1]")
        if not 0.0 < self.effectiveness_rls_forgetting_factor <= 1.0:
            raise ValueError("effectiveness_rls_forgetting_factor must be in (0, 1]")
        if min(
            self.effectiveness_rls_measurement_noise,
            self.effectiveness_rls_innovation_gate,
        ) <= 0.0:
            raise ValueError("RLS measurement noise and innovation gate must be positive")
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
    ) -> HarborMPCStep:
        opti = self.opti
        opti.set_value(self.p_initial, state)
        opti.set_value(self.p_goal, goal)
        opti.set_value(self.p_previous_control, previous_control)
        opti.set_value(self.p_position_drift, position_drift)
        opti.set_value(self.p_control_effectiveness, control_effectiveness)
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
        try:
            solution = opti.solve()
        except RuntimeError:
            self.last_status = str(opti.stats().get("return_status", "unknown"))
            raise
        self.last_status = str(opti.stats().get("return_status", "unknown"))
        states = np.asarray(solution.value(self.states), dtype=float).T
        collision = np.asarray(
            solution.value(self.collision_slack), dtype=float
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
            max_terminal_slack=float(np.max(np.abs(terminal), initial=0.0)),
        )

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
        p_initial = opti.parameter(nx, 1)
        p_goal = opti.parameter(pose_dim, 1)
        p_previous_control = opti.parameter(nu, 1)
        p_position_drift = opti.parameter(3, 1)
        p_control_effectiveness = opti.parameter(nu, 1)
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
        self._apply_bounds(opti, states, controls)
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
                )
                opti.subject_to(
                    ca.dot(relative, relative) + collision_slack[obstacle_index, index]
                    >= required * required
                )

        cost += cfg.collision_slack_weight * ca.sum1(ca.vec(collision_slack))
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
        self.p_initial = p_initial
        self.p_goal = p_goal
        self.p_previous_control = p_previous_control
        self.p_position_drift = p_position_drift
        self.p_control_effectiveness = p_control_effectiveness
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

    def _apply_bounds(self, opti, states, controls) -> None:
        model = self.agent.model
        domain = self.agent.domain
        opti.subject_to(
            opti.bounded(domain.x_bounds[0], states[0, :], domain.x_bounds[1])
        )
        opti.subject_to(
            opti.bounded(domain.y_bounds[0], states[1, :], domain.y_bounds[1])
        )
        if model.kind == "ugv":
            minimum_speed = -model.max_reverse_speed if model.variant in {
                "kinematic_bicycle",
                "dynamic_skid_steer",
            } else 0.0
            opti.subject_to(
                opti.bounded(minimum_speed, states[3, :], model.max_speed)
            )
            if model.variant == "dynamic_skid_steer":
                opti.subject_to(
                    opti.bounded(-model.max_yaw_rate, states[4, :], model.max_yaw_rate)
                )
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
            opti.subject_to(opti.bounded(0.0, states[3, :], model.max_speed))
            if model.variant == "marine_3dof":
                opti.subject_to(
                    opti.bounded(
                        -model.max_sway_speed,
                        states[4, :],
                        model.max_sway_speed,
                    )
                )
                opti.subject_to(
                    opti.bounded(
                        -model.max_yaw_rate, states[5, :], model.max_yaw_rate
                    )
                )
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
            opti.subject_to(
                opti.bounded(domain.z_bounds[0], states[2, :], domain.z_bounds[1])
            )
            opti.subject_to(
                opti.bounded(
                    -model.max_horizontal_speed,
                    states[6:8, :],
                    model.max_horizontal_speed,
                )
            )
            opti.subject_to(
                opti.bounded(
                    -model.max_vertical_speed,
                    states[8, :],
                    model.max_vertical_speed,
                )
            )
            opti.subject_to(
                opti.bounded(
                    -model.max_angular_rate,
                    states[9:12, :],
                    model.max_angular_rate,
                )
            )
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
        self.previous_controls = {
            agent.name: np.zeros(agent.model.control_dim) for agent in agents
        }
        self.previous_states: dict[str, np.ndarray] = {}
        self.position_drift_estimates = {
            agent.name: np.zeros(3) for agent in agents
        }
        self.residual_history = {agent.name: [] for agent in agents}
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
        self.effectiveness_history = {agent.name: [] for agent in agents}
        self.effectiveness_covariances = {
            agent.name: np.eye(agent.model.control_dim)
            * config.identification_prior_std**2
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
        self.identification_probe_sequence_by_agent: dict[str, list[int]] = {
            agent.name: [] for agent in agents
        }
        self.identification_probe_rejection_counts = {
            agent.name: np.zeros(agent.model.control_dim, dtype=int)
            for agent in agents
        }
        self.solve_count = 0
        self.fallback_count = 0
        self.solve_time_seconds = 0.0
        self.max_collision_slack = 0.0
        self.max_terminal_slack = 0.0
        self.solve_count_by_agent = {agent.name: 0 for agent in agents}
        self.fallback_count_by_agent = {agent.name: 0 for agent in agents}
        self.failure_steps_by_agent: dict[str, list[int]] = {
            agent.name: [] for agent in agents
        }
        self.failure_status_counts: dict[str, int] = {}

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
    ) -> np.ndarray:
        self._update_model_estimates(agent, state)
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
            position_drift=self.position_drift_estimates[agent.name],
            control_effectiveness=self.control_effectiveness_estimates[agent.name],
            identification_probe=probe,
            identification_probe_mask=probe_mask,
        )
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
                self.fallback_count += 1
                self.fallback_count_by_agent[agent.name] += 1
                self.failure_steps_by_agent[agent.name].append(step)
                status = self.optimizers[agent.name].last_status
                self.failure_status_counts[status] = (
                    self.failure_status_counts.get(status, 0) + 1
                )
                control = agent.model.guidance_control(
                    state, desired_velocity, dt, desired_pose=navigation_goal
                )
            else:
                control = solution.control
                probe_channel = None
        else:
            control = solution.control
        if solution is not None:
            self.solve_count += 1
            self.solve_count_by_agent[agent.name] += 1
            self.max_collision_slack = max(
                self.max_collision_slack, solution.max_collision_slack
            )
            self.max_terminal_slack = max(
                self.max_terminal_slack, solution.max_terminal_slack
            )
            if probe_channel is not None:
                self.identification_probe_count_by_agent[agent.name] += 1
                self.identification_probe_channel_counts[agent.name][
                    probe_channel
                ] += 1
                self.identification_probe_sequence_by_agent[agent.name].append(
                    probe_channel
                )
        self.solve_time_seconds += perf_counter() - started
        self.previous_controls[agent.name] = np.asarray(control, dtype=float)
        self.previous_states[agent.name] = np.asarray(state, dtype=float).copy()
        self.residual_history[agent.name].append(
            self.position_drift_estimates[agent.name].copy()
        )
        self.effectiveness_history[agent.name].append(
            self.control_effectiveness_estimates[agent.name].copy()
        )
        self.excitation_history[agent.name].append(
            self.excitation_energy[agent.name].copy()
        )
        self.information_std_history[agent.name].append(
            _posterior_effectiveness_std(
                self.information_matrices[agent.name], self.config
            )
        )
        return self.previous_controls[agent.name]

    def _update_model_estimates(
        self, agent: HarborAgent, current_state: np.ndarray
    ) -> None:
        """Update local actuator and position-residual estimates in sequence."""
        if agent.name not in self.previous_states:
            return
        previous_state = self.previous_states[agent.name]
        previous_control = self.previous_controls[agent.name]
        normalized = previous_control / agent.model.control_scale()
        self.excitation_energy[agent.name] += normalized * normalized
        normalized_sensitivity = _effectiveness_sensitivity_matrix(
            agent.model,
            previous_state,
            previous_control,
            self.dt,
            self.control_effectiveness_estimates[agent.name],
            self.config,
            normalize_dynamic_state=True,
        )
        jacobian = (
            normalized_sensitivity / self.config.identification_measurement_noise
        )
        self.information_matrices[agent.name] += jacobian.T @ jacobian
        if self.config.control_effectiveness_adaptation:
            if self.config.effectiveness_estimator_mode == "recursive_diagonal":
                estimate, covariance = _recursive_diagonal_effectiveness_update(
                    agent.model,
                    previous_state,
                    previous_control,
                    np.asarray(current_state, dtype=float),
                    self.dt,
                    self.control_effectiveness_estimates[agent.name],
                    self.effectiveness_covariances[agent.name],
                    self.config,
                )
                self.control_effectiveness_estimates[agent.name] = estimate
                self.effectiveness_covariances[agent.name] = covariance
            else:
                self.control_effectiveness_estimates[agent.name] = (
                    _estimate_effectiveness_vector(
                        agent.model,
                        previous_state,
                        previous_control,
                        np.asarray(current_state, dtype=float),
                        self.dt,
                        self.control_effectiveness_estimates[agent.name],
                        self.config,
                    )
                )
        if not self.config.residual_adaptation:
            return
        effectiveness = self.control_effectiveness_estimates[agent.name]
        predicted = agent.model.step(
            previous_state,
            effectiveness * previous_control,
            self.dt,
        )
        measured_velocity_error = (
            agent.model.position(current_state) - agent.model.position(predicted)
        ) / self.dt
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
            or self.config.effectiveness_estimator_mode != "diagonal"
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
                self.identification_probe_channel_counts[agent.name],
                self.identification_probe_rejection_counts[agent.name],
                self.config,
                effectiveness_estimate=self.control_effectiveness_estimates[
                    agent.name
                ],
            )
        else:
            channel = _identification_channel(
                self.excitation_energy[agent.name],
                self.identification_probe_channel_counts[agent.name],
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
) -> tuple[np.ndarray, np.ndarray]:
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
    innovation_covariance = (
        sensitivity @ predicted_covariance @ sensitivity.T
        + measurement_covariance
    )
    inverse_innovation = np.linalg.pinv(innovation_covariance, hermitian=True)
    mahalanobis = float(
        np.sqrt(max(float(innovation.T @ inverse_innovation @ innovation), 0.0))
    )
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
    return estimate, updated_covariance


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
