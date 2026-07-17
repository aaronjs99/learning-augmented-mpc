"""Heterogeneous, untethered UGV/USV/ROV harbor simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .communication import CommunicationNetwork, LinkConfig
from .models import PlatformModel, wrap_angle


@dataclass(frozen=True)
class OperatingDomain:
    """Independent workspace bounds for one platform."""

    x_bounds: tuple[float, float]
    y_bounds: tuple[float, float]
    z_bounds: tuple[float, float]

    def __post_init__(self) -> None:
        for name, bounds in (
            ("x", self.x_bounds),
            ("y", self.y_bounds),
            ("z", self.z_bounds),
        ):
            if len(bounds) != 2 or not bounds[0] < bounds[1]:
                raise ValueError(f"domain {name}_bounds must be increasing")

    def project(self, model: PlatformModel, state: np.ndarray) -> np.ndarray:
        """Project only the platform's own position into its configured domain."""
        value = np.asarray(state, dtype=float).copy()
        value[0] = np.clip(value[0], *self.x_bounds)
        value[1] = np.clip(value[1], *self.y_bounds)
        if model.kind == "rov":
            value[2] = np.clip(value[2], *self.z_bounds)
        return value

    def contains(self, position: np.ndarray) -> bool:
        """Return whether a world position belongs to this platform domain."""
        value = np.asarray(position, dtype=float)
        return bool(
            self.x_bounds[0] <= value[0] <= self.x_bounds[1]
            and self.y_bounds[0] <= value[1] <= self.y_bounds[1]
            and self.z_bounds[0] <= value[2] <= self.z_bounds[1]
        )


@dataclass(frozen=True)
class HarborAgent:
    """One independently actuated platform and its task metadata."""

    name: str
    model: PlatformModel
    start: np.ndarray
    goal: np.ndarray
    radius: float
    domain: OperatingDomain
    waypoints: np.ndarray | None = None
    profile: str | None = None
    display_name: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("harbor agent name must not be empty")
        if self.profile is not None and not self.profile.strip():
            raise ValueError("harbor agent profile must not be empty")
        if self.display_name is not None and not self.display_name.strip():
            raise ValueError("harbor agent display_name must not be empty")
        if np.asarray(self.start).shape != (self.model.state_dim,):
            raise ValueError("harbor agent start does not match model state dimension")
        if np.asarray(self.goal).shape != (self.model.pose_dim,):
            raise ValueError(
                f"harbor agent goal must have shape ({self.model.pose_dim},)"
            )
        if self.radius <= 0.0:
            raise ValueError("harbor agent radius must be positive")
        if not self.domain.contains(self.model.position(self.start)):
            raise ValueError("harbor agent start lies outside its operating domain")
        if not self.domain.contains(self.model.goal_position(self.goal)):
            raise ValueError("harbor agent goal lies outside its operating domain")
        if self.waypoints is not None:
            waypoints = np.asarray(self.waypoints, dtype=float)
            if waypoints.ndim != 2 or waypoints.shape[1] != self.model.pose_dim:
                raise ValueError(
                    f"harbor agent waypoints must have shape (N, {self.model.pose_dim})"
                )
            if not all(
                self.domain.contains(self.model.goal_position(point))
                for point in waypoints
            ):
                raise ValueError("harbor agent waypoint lies outside its operating domain")

    @property
    def route(self) -> np.ndarray:
        """Return intermediate waypoints followed by the final goal."""
        intermediate = (
            np.empty((0, self.model.pose_dim))
            if self.waypoints is None
            else np.asarray(self.waypoints, dtype=float)
        )
        return np.vstack((intermediate, self.goal))


@dataclass(frozen=True)
class HarborSimulationConfig:
    """Platform-neutral coordination and rollout settings."""

    dt: float = 0.2
    horizon: int = 160
    goal_hold_steps: int = 1
    guidance_update_interval_steps: int = 1
    goal_tolerance: float = 0.25
    orientation_tolerance: float = 0.15
    approach_speed_gain: float = 1.0
    coordination_distance: float = 2.0
    avoidance_gain: float = 1.5
    yielding_speed_scale: float = 0.15
    coordination_policy: str = "eta_priority"
    priority_response_scale: float = 0.5
    predict_delayed_messages: bool = True
    world_x_bounds: tuple[float, float] = (-5.0, 5.0)
    world_y_bounds: tuple[float, float] = (-5.0, 5.0)
    shoreline_y: float = 3.0
    seabed_z: float = -4.0

    def __post_init__(self) -> None:
        if (
            self.dt <= 0.0
            or self.horizon <= 0
            or self.goal_hold_steps <= 0
            or self.guidance_update_interval_steps <= 0
        ):
            raise ValueError("harbor dt and horizon must be positive")
        if (
            self.goal_tolerance <= 0.0
            or self.orientation_tolerance <= 0.0
            or self.approach_speed_gain <= 0.0
            or self.coordination_distance <= 0.0
        ):
            raise ValueError("harbor goal and coordination distances must be positive")
        if self.avoidance_gain < 0.0:
            raise ValueError("harbor avoidance_gain must be nonnegative")
        if not 0.0 <= self.yielding_speed_scale <= 1.0:
            raise ValueError("harbor yielding_speed_scale must be in [0, 1]")
        if self.coordination_policy not in {"reciprocal", "eta_priority"}:
            raise ValueError(
                "harbor coordination_policy must be reciprocal or eta_priority"
            )
        if not 0.0 <= self.priority_response_scale <= 1.0:
            raise ValueError("harbor priority_response_scale must be in [0, 1]")
        if not self.world_x_bounds[0] < self.world_x_bounds[1]:
            raise ValueError("harbor world_x_bounds must be increasing")
        if not self.world_y_bounds[0] < self.shoreline_y < self.world_y_bounds[1]:
            raise ValueError("harbor shoreline_y must lie inside world_y_bounds")
        if self.seabed_z >= 0.0:
            raise ValueError("harbor seabed_z must be below the water surface")


@dataclass(frozen=True)
class HarborDisturbanceConfig:
    """Unknown execution-plant effects used for robustness experiments."""

    water_current: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ugv_control_effectiveness: float | tuple[float, ...] = 1.0
    usv_control_effectiveness: float | tuple[float, ...] = 1.0
    rov_control_effectiveness: float | tuple[float, ...] = 1.0
    agent_control_effectiveness: dict[
        str, float | tuple[float, ...]
    ] = field(default_factory=dict)
    agent_control_effectiveness_schedule: dict[
        str, tuple[dict[str, object], ...]
    ] = field(default_factory=dict)
    evaluation_hold_steps: int = 12

    def __post_init__(self) -> None:
        if len(self.water_current) != 3:
            raise ValueError("water_current must contain [x, y, z] velocity")
        if not np.all(np.isfinite(self.water_current)):
            raise ValueError("water_current must be finite")
        for name in (
            "ugv_control_effectiveness",
            "usv_control_effectiveness",
            "rov_control_effectiveness",
        ):
            value = _normalize_effectiveness(getattr(self, name), name)
            object.__setattr__(self, name, value)
        normalized_agents = {
            str(name): _normalize_effectiveness(value, f"agent {name}")
            for name, value in self.agent_control_effectiveness.items()
        }
        if any(not name.strip() for name in normalized_agents):
            raise ValueError("agent effectiveness names must not be empty")
        object.__setattr__(self, "agent_control_effectiveness", normalized_agents)
        normalized_schedule = {}
        for agent_name, configured_events in self.agent_control_effectiveness_schedule.items():
            name = str(agent_name).strip()
            if not name:
                raise ValueError("scheduled effectiveness names must not be empty")
            events = []
            for configured in configured_events:
                if isinstance(configured, dict) and set(configured) == {
                    "step",
                    "effectiveness",
                }:
                    configured_step = configured["step"]
                    configured_effectiveness = configured["effectiveness"]
                elif isinstance(configured, (tuple, list)) and len(configured) == 2:
                    configured_step, configured_effectiveness = configured
                else:
                    raise ValueError(
                        "scheduled effectiveness events require step and effectiveness"
                    )
                step = int(configured_step)
                if step < 0 or step != configured_step:
                    raise ValueError(
                        "scheduled effectiveness steps must be nonnegative integers"
                    )
                events.append(
                    (
                        step,
                        _normalize_effectiveness(
                            configured_effectiveness, f"scheduled agent {name}"
                        ),
                    )
                )
            events.sort(key=lambda event: event[0])
            if len({step for step, _ in events}) != len(events):
                raise ValueError(
                    "scheduled effectiveness steps must be unique per agent"
                )
            normalized_schedule[name] = tuple(events)
        object.__setattr__(
            self, "agent_control_effectiveness_schedule", normalized_schedule
        )
        if self.evaluation_hold_steps <= 0:
            raise ValueError("evaluation_hold_steps must be positive")

    def effectiveness(
        self,
        model: PlatformModel,
        agent_name: str | None = None,
        step: int | None = None,
    ) -> np.ndarray:
        """Return one hidden effectiveness value per platform control channel."""
        configured = (
            self.agent_control_effectiveness[agent_name]
            if agent_name in self.agent_control_effectiveness
            else getattr(self, f"{model.kind}_control_effectiveness")
        )
        if step is not None and agent_name in self.agent_control_effectiveness_schedule:
            for event_step, event_value in self.agent_control_effectiveness_schedule[
                agent_name
            ]:
                if event_step > step:
                    break
                configured = event_value
        values = np.asarray(configured, dtype=float).reshape(-1)
        if len(values) == 1:
            return np.full(model.control_dim, values[0], dtype=float)
        if len(values) != model.control_dim:
            raise ValueError(
                f"{agent_name or model.kind} effectiveness must have "
                f"1 or {model.control_dim} entries"
            )
        return values.copy()

    def current(self, model: PlatformModel) -> np.ndarray:
        """Return the environmental world velocity acting on a platform."""
        if model.kind == "ugv":
            return np.zeros(3)
        value = np.asarray(self.water_current, dtype=float).copy()
        if model.kind == "usv":
            value[2] = 0.0
        return value


@dataclass(frozen=True)
class HarborObservationNoiseConfig:
    """Seeded onboard state-measurement noise, separate from plant truth."""

    enabled: bool = False
    seed: int = 0
    kind_state_std: dict[str, tuple[float, ...]] = field(default_factory=dict)
    agent_state_std: dict[str, tuple[float, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("observation-noise seed must be nonnegative")
        for field_name in ("kind_state_std", "agent_state_std"):
            values = {}
            for name, configured in getattr(self, field_name).items():
                key = str(name).strip()
                array = np.asarray(configured, dtype=float).reshape(-1)
                if not key or len(array) == 0 or not np.all(np.isfinite(array)):
                    raise ValueError(
                        f"{field_name} entries must be named finite vectors"
                    )
                if np.any(array < 0.0):
                    raise ValueError(
                        f"{field_name} standard deviations must be nonnegative"
                    )
                values[key] = tuple(float(value) for value in array)
            object.__setattr__(self, field_name, values)

    def state_std(self, agent: HarborAgent) -> np.ndarray:
        """Return the configured standard deviation for one local state."""
        configured = self.agent_state_std.get(
            agent.name,
            self.kind_state_std.get(agent.model.kind, (0.0,) * agent.model.state_dim),
        )
        values = np.asarray(configured, dtype=float)
        if values.shape != (agent.model.state_dim,):
            raise ValueError(
                f"{agent.name} observation noise must have "
                f"{agent.model.state_dim} entries"
            )
        return values

    def measure(
        self,
        agent: HarborAgent,
        state: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate one noisy local state while preserving angle topology."""
        truth = np.asarray(state, dtype=float)
        if not self.enabled:
            return truth.copy()
        measured = truth + rng.normal(0.0, self.state_std(agent))
        angle_indices = (2,) if agent.model.pose_dim == 3 else (3, 4, 5)
        for index in angle_indices:
            measured[index] = wrap_angle(measured[index])
        measured = agent.domain.project(agent.model, measured)
        model = agent.model
        if model.kind == "ugv":
            measured[3] = np.clip(
                measured[3], -model.max_reverse_speed, model.max_speed
            )
            if model.variant == "dynamic_skid_steer":
                measured[4] = np.clip(
                    measured[4], -model.max_yaw_rate, model.max_yaw_rate
                )
        elif model.kind == "usv":
            measured[3] = np.clip(measured[3], 0.0, model.max_speed)
            if model.variant == "marine_3dof":
                measured[4] = np.clip(
                    measured[4], -model.max_sway_speed, model.max_sway_speed
                )
                measured[5] = np.clip(
                    measured[5], -model.max_yaw_rate, model.max_yaw_rate
                )
        else:
            measured[6:8] = np.clip(
                measured[6:8], -model.max_horizontal_speed, model.max_horizontal_speed
            )
            measured[8] = np.clip(
                measured[8], -model.max_vertical_speed, model.max_vertical_speed
            )
            measured[9:12] = np.clip(
                measured[9:12], -model.max_angular_rate, model.max_angular_rate
            )
        return measured


@dataclass(frozen=True)
class HarborResult:
    """Trajectories and communication telemetry for one rollout."""

    states: dict[str, np.ndarray]
    observed_states: dict[str, np.ndarray]
    positions: dict[str, np.ndarray]
    controls: dict[str, np.ndarray]
    applied_controls: dict[str, np.ndarray]
    applied_effectiveness: dict[str, np.ndarray]
    first_goal_steps: dict[str, int | None]
    final_goal_errors: dict[str, float]
    final_orientation_errors: dict[str, float]
    all_goals_reached: bool
    min_pairwise_distance: float
    pairwise_violation_count: int
    messages_sent: int
    messages_delivered: int
    messages_dropped: int
    guidance_update_count: int


class HarborControlProvider(Protocol):
    """Controller boundary that exposes only local state, intent, and inbox."""

    def control(
        self,
        *,
        agent: HarborAgent,
        state: np.ndarray,
        navigation_goal: np.ndarray,
        desired_velocity: np.ndarray,
        inbox: dict,
        step: int,
        dt: float,
    ) -> np.ndarray:
        """Return one bounded platform control for the current update."""

def run_harbor_simulation(
    agents: list[HarborAgent],
    config: HarborSimulationConfig,
    communication: LinkConfig,
    control_provider: HarborControlProvider | None = None,
    disturbance: HarborDisturbanceConfig | None = None,
    observation_noise: HarborObservationNoiseConfig | None = None,
) -> HarborResult:
    """Simulate independent platforms coordinated only by received messages."""
    if len(agents) < 2:
        raise ValueError("harbor simulation requires at least two platforms")
    names = [agent.name for agent in agents]
    if len(set(names)) != len(names):
        raise ValueError("harbor platform names must be unique")

    by_name = {agent.name: agent for agent in agents}
    current = {agent.name: np.asarray(agent.start, dtype=float).copy() for agent in agents}
    histories = {name: [state.copy()] for name, state in current.items()}
    sensor = observation_noise or HarborObservationNoiseConfig()
    seed_sequence = np.random.SeedSequence(sensor.seed)
    sensor_rngs = {
        agent.name: np.random.default_rng(child)
        for agent, child in zip(agents, seed_sequence.spawn(len(agents)), strict=True)
    }
    observed_histories = {
        agent.name: [
            sensor.measure(agent, current[agent.name], sensor_rngs[agent.name])
        ]
        for agent in agents
    }
    controls = {name: [] for name in names}
    applied_controls = {name: [] for name in names}
    applied_effectiveness = {name: [] for name in names}
    plant = disturbance or HarborDisturbanceConfig()
    last_controls: dict[str, np.ndarray] = {}
    guidance_update_count = 0
    first_goal_steps: dict[str, int | None] = {name: None for name in names}
    route_indices = {name: 0 for name in names}
    stopped_at_goal: set[str] = set()
    goal_hold_counts = {
        name: int(_goal_reached(by_name[name], current[name], by_name[name].goal, config))
        for name in names
    }
    network = CommunicationNetwork(names, communication)

    for step in range(config.horizon):
        measured = {
            name: observed_histories[name][-1] for name in names
        }
        observations = {
            name: (
                agent.model.position(measured[name]),
                agent.model.velocity(measured[name]) + plant.current(agent.model),
                agent.model.goal_position(agent.goal),
                _platform_speed(agent.model),
            )
            for name, agent in by_name.items()
        }
        if all(
            count >= config.goal_hold_steps for count in goal_hold_counts.values()
        ):
            for name in names:
                if first_goal_steps[name] is None:
                    first_goal_steps[name] = step
            break
        inboxes = network.exchange(step, observations)
        next_states = {}
        for name, agent in by_name.items():
            position = observations[name][0]
            route = agent.route
            while (
                route_indices[name] < len(route) - 1
                and _intermediate_waypoint_complete(
                    agent,
                    measured[name],
                    route,
                    route_indices[name],
                    config,
                )
            ):
                route_indices[name] += 1
            navigation_goal = route[route_indices[name]]
            navigation_position = agent.model.goal_position(navigation_goal)
            goal_delta = navigation_position - position
            goal_distance = float(np.linalg.norm(goal_delta))
            at_final_goal = route_indices[name] == len(route) - 1
            reached_navigation_goal = _goal_reached(
                agent, measured[name], navigation_goal, config
            )
            if (
                at_final_goal
                and first_goal_steps[name] is None
                and _goal_reached(agent, current[name], agent.goal, config)
            ):
                first_goal_steps[name] = step
            if at_final_goal and reached_navigation_goal:
                desired_velocity = np.zeros(3)
            else:
                desired_velocity = np.zeros(3)
                if goal_distance > config.goal_tolerance:
                    desired_velocity = goal_delta / max(goal_distance, 1e-9)
                    desired_velocity *= min(
                        _platform_speed(agent.model),
                        config.approach_speed_gain * goal_distance,
                    )
                    desired_velocity = _coordinate_velocity(
                        name,
                        position,
                        navigation_position,
                        _platform_speed(agent.model),
                        desired_velocity,
                        inboxes[name],
                        config,
                        step,
                    )
            at_goal = at_final_goal and reached_navigation_goal
            left_goal = name in stopped_at_goal and not at_goal
            if left_goal:
                stopped_at_goal.remove(name)
            continuous_station_keeping = control_provider is not None or (
                agent.model.variant in {"marine_3dof", "marine_6dof"}
            )
            update_guidance = name not in last_controls or (
                at_goal and name not in stopped_at_goal
            )
            if at_goal and continuous_station_keeping:
                update_guidance = True
            if left_goal:
                update_guidance = True
            if not at_goal and step % config.guidance_update_interval_steps == 0:
                update_guidance = True
            if update_guidance:
                if control_provider is None:
                    proposed_control = agent.model.guidance_control(
                        measured[name],
                        desired_velocity,
                        config.dt,
                        desired_pose=navigation_goal,
                    )
                    smoothing = float(
                        getattr(agent.model, "control_smoothing", 1.0)
                    )
                else:
                    proposed_control = control_provider.control(
                        agent=agent,
                        state=measured[name],
                        navigation_goal=navigation_goal,
                        desired_velocity=desired_velocity,
                        inbox=inboxes[name],
                        step=step,
                        dt=config.dt,
                    )
                    smoothing = 1.0
                control = (
                    proposed_control
                    if name not in last_controls
                    else smoothing * proposed_control
                    + (1.0 - smoothing) * last_controls[name]
                )
                last_controls[name] = control
                guidance_update_count += 1
                if at_goal:
                    stopped_at_goal.add(name)
            else:
                control = last_controls[name]
            effectiveness = plant.effectiveness(agent.model, agent.name, step)
            applied_control = effectiveness * control
            next_state = agent.model.step(current[name], applied_control, config.dt)
            current_velocity = plant.current(agent.model)
            next_state = _advect_position(agent.model, next_state, current_velocity, config.dt)
            next_states[name] = agent.domain.project(agent.model, next_state)
            controls[name].append(control)
            applied_controls[name].append(applied_control)
            applied_effectiveness[name].append(effectiveness)

        current = next_states
        for name in names:
            histories[name].append(current[name].copy())
            observed_histories[name].append(
                sensor.measure(
                    by_name[name], current[name], sensor_rngs[name]
                )
            )
            goal_hold_counts[name] = (
                goal_hold_counts[name] + 1
                if _goal_reached(by_name[name], current[name], by_name[name].goal, config)
                else 0
            )

    for name, agent in by_name.items():
        if first_goal_steps[name] is None:
            if _goal_reached(agent, current[name], agent.goal, config):
                first_goal_steps[name] = config.horizon

    state_arrays = {name: np.asarray(values) for name, values in histories.items()}
    position_arrays = {
        name: np.asarray([by_name[name].model.position(state) for state in values])
        for name, values in state_arrays.items()
    }
    final_goal_errors = {
        name: float(
            np.linalg.norm(
                position_arrays[name][-1]
                - by_name[name].model.goal_position(by_name[name].goal)
            )
        )
        for name in names
    }
    final_orientation_errors = {
        name: float(
            np.linalg.norm(
                by_name[name].model.orientation_error(state_arrays[name][-1], by_name[name].goal)
            )
        )
        for name in names
    }
    min_distance, violation_count = _pairwise_metrics(position_arrays, by_name)
    return HarborResult(
        states=state_arrays,
        observed_states={
            name: np.asarray(values) for name, values in observed_histories.items()
        },
        positions=position_arrays,
        controls={name: np.asarray(values) for name, values in controls.items()},
        applied_controls={
            name: np.asarray(values) for name, values in applied_controls.items()
        },
        applied_effectiveness={
            name: np.asarray(values) for name, values in applied_effectiveness.items()
        },
        first_goal_steps=first_goal_steps,
        final_goal_errors=final_goal_errors,
        final_orientation_errors=final_orientation_errors,
        all_goals_reached=all(
            final_goal_errors[name] <= config.goal_tolerance
            and final_orientation_errors[name] <= config.orientation_tolerance
            for name in names
        ),
        min_pairwise_distance=min_distance,
        pairwise_violation_count=violation_count,
        messages_sent=network.sent_count,
        messages_delivered=network.delivered_count,
        messages_dropped=network.dropped_count,
        guidance_update_count=guidance_update_count,
    )


def _advect_position(
    model: PlatformModel, state: np.ndarray, current: np.ndarray, dt: float
) -> np.ndarray:
    """Apply an unmodeled world-frame current to marine position states."""
    value = np.asarray(state, dtype=float).copy()
    dimensions = 3 if model.kind == "rov" else 2
    value[:dimensions] += dt * np.asarray(current, dtype=float)[:dimensions]
    return value


def _normalize_effectiveness(value, name: str) -> float | tuple[float, ...]:
    array = np.asarray(value, dtype=float).reshape(-1)
    if len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} control effectiveness must be finite and nonempty")
    if np.any(array <= 0.0) or np.any(array > 1.5):
        raise ValueError(f"{name} control effectiveness must be in (0, 1.5]")
    if len(array) == 1:
        return float(array[0])
    return tuple(float(item) for item in array)


def _coordinate_velocity(
    name, position, goal, cruise_speed, desired, inbox, config, current_step
):
    adjusted = np.asarray(desired, dtype=float).copy()
    for other_name, message in inbox.items():
        message_position = message.position
        if config.predict_delayed_messages:
            message_age = max(0, current_step - message.sent_step)
            message_position = (
                message.position + message_age * config.dt * message.velocity
            )
        relative = position - message_position
        distance = float(np.linalg.norm(relative))
        if distance >= config.coordination_distance or distance < 1e-9:
            continue
        closing = float(np.dot(desired - message.velocity, -relative / distance))
        if closing <= 0.0:
            continue
        should_yield = name > other_name
        if config.coordination_policy == "eta_priority":
            own_eta = np.linalg.norm(goal - position) / cruise_speed
            other_eta = (
                np.linalg.norm(message.goal - message_position)
                / message.cruise_speed
            )
            should_yield = (own_eta, name) > (other_eta, other_name)
        if should_yield:
            adjusted *= config.yielding_speed_scale
            planar = relative[:2] / distance
            adjusted[:2] += config.avoidance_gain * np.array([-planar[1], planar[0]])
        else:
            response_scale = (
                1.0
                if config.coordination_policy == "reciprocal"
                else config.priority_response_scale
            )
            adjusted += (
                response_scale * config.avoidance_gain * relative / distance
            )
    return adjusted


def _platform_speed(model: PlatformModel) -> float:
    mission_speed = getattr(model, "mission_speed", None)
    if mission_speed is not None:
        return float(mission_speed)
    if model.kind == "rov":
        return float(model.max_horizontal_speed)
    return float(model.max_speed)


def _goal_reached(agent, state, goal, config) -> bool:
    position_error = np.linalg.norm(
        agent.model.position(state) - agent.model.goal_position(goal)
    )
    orientation_error = np.linalg.norm(agent.model.orientation_error(state, goal))
    return bool(
        position_error <= config.goal_tolerance
        and orientation_error <= config.orientation_tolerance
    )


def _intermediate_waypoint_complete(agent, state, route, index, config) -> bool:
    """Advance a waypoint when reached or crossed toward the next route pose."""
    position = agent.model.position(state)
    waypoint = agent.model.goal_position(route[index])
    if np.linalg.norm(position - waypoint) <= config.goal_tolerance:
        return True
    next_position = agent.model.goal_position(route[index + 1])
    outgoing = next_position - waypoint
    length_sq = float(np.dot(outgoing, outgoing))
    if length_sq <= np.finfo(float).eps:
        return False
    progress = float(np.dot(position - waypoint, outgoing) / length_sq)
    projection = waypoint + np.clip(progress, 0.0, 1.0) * outgoing
    cross_track = float(np.linalg.norm(position - projection))
    return progress >= 0.0 and cross_track <= 2.0 * config.goal_tolerance


def _pairwise_metrics(position_arrays, agents):
    names = sorted(position_arrays)
    minimum = float("inf")
    violations = 0
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            relative = position_arrays[first] - position_arrays[second]
            starts = relative[:-1]
            deltas = relative[1:] - starts
            speed_sq = np.sum(deltas * deltas, axis=1)
            alpha = np.zeros(len(starts))
            moving = speed_sq > np.finfo(float).eps
            alpha[moving] = -np.sum(starts[moving] * deltas[moving], axis=1) / speed_sq[moving]
            closest = starts + np.clip(alpha, 0.0, 1.0)[:, None] * deltas
            distances = np.linalg.norm(closest, axis=1)
            required = agents[first].radius + agents[second].radius
            minimum = min(minimum, float(np.min(distances)))
            violations += int(np.sum(distances < required))
    return minimum, violations
