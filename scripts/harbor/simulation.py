"""Heterogeneous, untethered UGV/USV/ROV harbor simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .communication import CommunicationNetwork, LinkConfig
from .models import PlatformModel


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

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("harbor agent name must not be empty")
        if np.asarray(self.start).shape != (self.model.state_dim,):
            raise ValueError("harbor agent start does not match model state dimension")
        if np.asarray(self.goal).shape != (3,):
            raise ValueError("harbor agent goal must have shape (3,)")
        if self.radius <= 0.0:
            raise ValueError("harbor agent radius must be positive")
        if not self.domain.contains(self.model.position(self.start)):
            raise ValueError("harbor agent start lies outside its operating domain")
        if not self.domain.contains(self.goal):
            raise ValueError("harbor agent goal lies outside its operating domain")


@dataclass(frozen=True)
class HarborSimulationConfig:
    """Platform-neutral coordination and rollout settings."""

    dt: float = 0.2
    horizon: int = 160
    goal_tolerance: float = 0.25
    coordination_distance: float = 2.0
    avoidance_gain: float = 1.5
    yielding_speed_scale: float = 0.15

    def __post_init__(self) -> None:
        if self.dt <= 0.0 or self.horizon <= 0:
            raise ValueError("harbor dt and horizon must be positive")
        if self.goal_tolerance <= 0.0 or self.coordination_distance <= 0.0:
            raise ValueError("harbor goal and coordination distances must be positive")
        if self.avoidance_gain < 0.0:
            raise ValueError("harbor avoidance_gain must be nonnegative")
        if not 0.0 <= self.yielding_speed_scale <= 1.0:
            raise ValueError("harbor yielding_speed_scale must be in [0, 1]")


@dataclass(frozen=True)
class HarborResult:
    """Trajectories and communication telemetry for one rollout."""

    states: dict[str, np.ndarray]
    positions: dict[str, np.ndarray]
    controls: dict[str, np.ndarray]
    first_goal_steps: dict[str, int | None]
    final_goal_errors: dict[str, float]
    all_goals_reached: bool
    min_pairwise_distance: float
    pairwise_violation_count: int
    messages_sent: int
    messages_delivered: int
    messages_dropped: int

def run_harbor_simulation(
    agents: list[HarborAgent],
    config: HarborSimulationConfig,
    communication: LinkConfig,
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
    controls = {name: [] for name in names}
    first_goal_steps: dict[str, int | None] = {name: None for name in names}
    network = CommunicationNetwork(names, communication)

    for step in range(config.horizon):
        observations = {
            name: (
                agent.model.position(current[name]),
                agent.model.velocity(current[name]),
                agent.goal,
            )
            for name, agent in by_name.items()
        }
        inboxes = network.exchange(step, observations)
        next_states = {}
        for name, agent in by_name.items():
            position = observations[name][0]
            goal_delta = agent.goal - position
            goal_distance = float(np.linalg.norm(goal_delta))
            if goal_distance <= config.goal_tolerance:
                desired_velocity = np.zeros(3)
                if first_goal_steps[name] is None:
                    first_goal_steps[name] = step
            else:
                desired_velocity = goal_delta / max(goal_distance, 1e-9)
                desired_velocity *= _platform_speed(agent.model)
                desired_velocity = _coordinate_velocity(
                    name,
                    position,
                    desired_velocity,
                    inboxes[name],
                    config,
                )
            control = agent.model.guidance_control(
                current[name], desired_velocity, config.dt
            )
            next_state = agent.model.step(current[name], control, config.dt)
            next_states[name] = agent.domain.project(agent.model, next_state)
            controls[name].append(control)

        current = next_states
        for name in names:
            histories[name].append(current[name].copy())

    for name, agent in by_name.items():
        if first_goal_steps[name] is None:
            final_distance = np.linalg.norm(
                agent.model.position(current[name]) - agent.goal
            )
            if final_distance <= config.goal_tolerance:
                first_goal_steps[name] = config.horizon

    state_arrays = {name: np.asarray(values) for name, values in histories.items()}
    position_arrays = {
        name: np.asarray([by_name[name].model.position(state) for state in values])
        for name, values in state_arrays.items()
    }
    final_goal_errors = {
        name: float(np.linalg.norm(position_arrays[name][-1] - by_name[name].goal))
        for name in names
    }
    min_distance, violation_count = _pairwise_metrics(position_arrays, by_name)
    return HarborResult(
        states=state_arrays,
        positions=position_arrays,
        controls={name: np.asarray(values) for name, values in controls.items()},
        first_goal_steps=first_goal_steps,
        final_goal_errors=final_goal_errors,
        all_goals_reached=all(
            error <= config.goal_tolerance for error in final_goal_errors.values()
        ),
        min_pairwise_distance=min_distance,
        pairwise_violation_count=violation_count,
        messages_sent=network.sent_count,
        messages_delivered=network.delivered_count,
        messages_dropped=network.dropped_count,
    )


def _coordinate_velocity(name, position, desired, inbox, config):
    adjusted = np.asarray(desired, dtype=float).copy()
    for other_name, message in inbox.items():
        relative = position - message.position
        distance = float(np.linalg.norm(relative))
        if distance >= config.coordination_distance or distance < 1e-9:
            continue
        closing = float(np.dot(desired - message.velocity, -relative / distance))
        if closing <= 0.0:
            continue
        if name > other_name:
            adjusted *= config.yielding_speed_scale
            planar = relative[:2] / distance
            adjusted[:2] += config.avoidance_gain * np.array([-planar[1], planar[0]])
        else:
            adjusted += config.avoidance_gain * relative / distance
    return adjusted


def _platform_speed(model: PlatformModel) -> float:
    if model.kind == "rov":
        return float(model.max_surge_speed)
    return float(model.max_speed)


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
