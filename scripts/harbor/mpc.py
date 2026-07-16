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
    ipopt_max_iter: int = 120
    ipopt_print_level: int = 0

    def __post_init__(self) -> None:
        integer_values = {
            "prediction_horizon": self.prediction_horizon,
            "replan_interval_steps": self.replan_interval_steps,
            "learning_iterations": self.learning_iterations,
            "terminal_samples": self.terminal_samples,
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
        )
        if any(value < 0.0 for value in nonnegative):
            raise ValueError("harbor_mpc weights and slack bounds must be nonnegative")


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
    ) -> HarborMPCStep:
        opti = self.opti
        opti.set_value(self.p_initial, state)
        opti.set_value(self.p_goal, goal)
        opti.set_value(self.p_previous_control, previous_control)
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
        p_obstacles = [opti.parameter(3, horizon) for _ in self.other_agents]

        opti.subject_to(states[:, 0] == p_initial)
        opti.subject_to(
            opti.bounded(0.0, collision_slack, cfg.collision_slack_bound)
        )
        self._apply_bounds(opti, states, controls)

        cost = 0
        previous = p_previous_control
        for index in range(horizon):
            opti.subject_to(
                states[:, index + 1]
                == _symbolic_step(ca, model, states[:, index], controls[:, index], self.dt)
            )
            cost += self._tracking_cost(states[:, index + 1], p_goal)
            cost += cfg.control_weight * ca.dot(controls[:, index], controls[:, index])
            delta_control = controls[:, index] - previous
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
            opti.subject_to(opti.bounded(0.0, states[3, :], model.max_speed))
            opti.subject_to(
                opti.bounded(-model.max_acceleration, controls[0, :], model.max_acceleration)
            )
            opti.subject_to(
                opti.bounded(-model.max_yaw_rate, controls[1, :], model.max_yaw_rate)
            )
        elif model.kind == "usv":
            opti.subject_to(opti.bounded(0.0, states[3, :], model.max_speed))
            opti.subject_to(
                opti.bounded(-model.max_thrust, controls[0, :], model.max_thrust)
            )
            opti.subject_to(
                opti.bounded(-model.max_yaw_rate, controls[1, :], model.max_yaw_rate)
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
            opti.subject_to(
                opti.bounded(-model.max_force, controls[:3, :], model.max_force)
            )
            opti.subject_to(
                opti.bounded(-model.max_torque, controls[3:6, :], model.max_torque)
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
        self.solve_count = 0
        self.fallback_count = 0
        self.solve_time_seconds = 0.0
        self.max_collision_slack = 0.0
        self.max_terminal_slack = 0.0
        self.solve_count_by_agent = {agent.name: 0 for agent in agents}
        self.fallback_count_by_agent = {agent.name: 0 for agent in agents}
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
        started = perf_counter()
        try:
            solution = self.optimizers[agent.name].solve(
                state=np.asarray(state, dtype=float),
                goal=np.asarray(navigation_goal, dtype=float),
                obstacle_predictions=predictions,
                safe_states=safe_sample_states,
                safe_costs=safe_costs,
                warm_states=warm_states,
                warm_controls=warm_controls,
                previous_control=self.previous_controls[agent.name],
            )
        except RuntimeError:
            self.fallback_count += 1
            self.fallback_count_by_agent[agent.name] += 1
            status = self.optimizers[agent.name].last_status
            self.failure_status_counts[status] = (
                self.failure_status_counts.get(status, 0) + 1
            )
            control = agent.model.guidance_control(
                state, desired_velocity, dt, desired_pose=navigation_goal
            )
        else:
            self.solve_count += 1
            self.solve_count_by_agent[agent.name] += 1
            self.max_collision_slack = max(
                self.max_collision_slack, solution.max_collision_slack
            )
            self.max_terminal_slack = max(
                self.max_terminal_slack, solution.max_terminal_slack
            )
            control = solution.control
        finally:
            self.solve_time_seconds += perf_counter() - started
        self.previous_controls[agent.name] = np.asarray(control, dtype=float)
        return self.previous_controls[agent.name]

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
            predictions[other.name] = message.position + times * message.velocity
        return predictions


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
    if model.kind == "ugv":
        speed = state[3] + dt * control[0]
        heading = state[2] + dt * control[1]
        return ca.vertcat(
            state[0] + dt * speed * ca.cos(heading),
            state[1] + dt * speed * ca.sin(heading),
            heading,
            speed,
        )
    if model.kind == "usv":
        speed = state[3] + dt * (control[0] - model.drag * state[3])
        heading = state[2] + dt * control[1]
        return ca.vertcat(
            state[0] + dt * speed * ca.cos(heading),
            state[1] + dt * speed * ca.sin(heading),
            heading,
            speed,
        )
    velocity = state[6:9] + dt * (control[:3] - model.linear_drag * state[6:9])
    angular_rate = state[9:12] + dt * (
        control[3:6] - model.angular_drag * state[9:12]
    )
    return ca.vertcat(
        state[:3] + dt * velocity,
        state[3:6] + dt * angular_rate,
        velocity,
        angular_rate,
    )


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
    if model.kind in {"ugv", "usv"}:
        return state[3]
    return state[6:9]


def _symbolic_orientation_cost(ca, model: PlatformModel, state, goal):
    if model.pose_dim == 3:
        return 2.0 * (1.0 - ca.cos(state[2] - goal[2]))
    return sum(2.0 * (1.0 - ca.cos(state[index] - goal[index])) for index in range(3, 6))
