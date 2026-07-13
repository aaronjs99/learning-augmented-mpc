"""Composable runner for APF baseline and repeated manta LMPC iterations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time

import numpy as np

from scripts.dynamics import MantaDynamicsConfig, rk4_step_np
from scripts.mpc.manta_lmpc import MantaAgentOptimizer, MantaLMPCConfig
from scripts.simulation import Scenario, StaticObstacle

from .apf import APFConfig, compute_apf_control
from .hyperplanes import get_symmetric_hyperplanes_spatial
from .safe_sets import build_staggered_safe_sets, sample_terminal_safe_set


@dataclass
class TrajectoryValidation:
    """Safety and completion checks for one APF or LMPC trajectory."""

    all_goals_reached: bool
    min_pairwise_distance: float
    pairwise_violation_count: int
    min_obstacle_clearance: float
    obstacle_violation_count: int
    fallback_count: int

    @property
    def safe(self) -> bool:
        """Return true when pairwise and obstacle constraints are satisfied."""
        return (
            self.pairwise_violation_count == 0
            and self.obstacle_violation_count == 0
        )

    @property
    def valid(self) -> bool:
        """Return true when the trajectory is complete and safe."""
        return self.all_goals_reached and self.safe

    @property
    def usable_for_learning(self) -> bool:
        """Return true when a trajectory is a complete learned safe-set member."""
        return self.valid

    @property
    def solver_clean(self) -> bool:
        """Return true when no safe-set fallback was needed."""
        return self.fallback_count == 0

    def to_dict(self) -> dict[str, bool | float | int]:
        """Serialize validation metrics for run summaries."""
        return {
            "valid": self.valid,
            "safe": self.safe,
            "solver_clean": self.solver_clean,
            "usable_for_learning": self.usable_for_learning,
            "all_goals_reached": self.all_goals_reached,
            "min_pairwise_distance": self.min_pairwise_distance,
            "pairwise_violation_count": self.pairwise_violation_count,
            "min_obstacle_clearance": self.min_obstacle_clearance,
            "obstacle_violation_count": self.obstacle_violation_count,
            "fallback_count": self.fallback_count,
        }


@dataclass
class MantaLMPCRunResult:
    """In-memory result from one config-driven APF/LMPC run."""

    scenario_name: str
    histories: list[dict[int, np.ndarray]]
    controls_by_iteration: list[np.ndarray]
    statuses_by_iteration: list[list[dict[int, str]]]
    success_by_iteration: list[bool]
    goal_reached_by_iteration: list[bool]
    learned_by_iteration: list[bool]
    validation_by_iteration: list[TrajectoryValidation]
    selected_iteration: int | None

    @property
    def final_history(self) -> dict[int, np.ndarray]:
        """Return the selected valid history, or the latest history if none is valid."""
        if self.selected_iteration is None:
            return self.histories[-1]
        return self.histories[self.selected_iteration]

    @property
    def final_statuses(self) -> list[dict[int, str]] | None:
        """Return statuses for the selected final LMPC iteration."""
        if self.selected_iteration is None or self.selected_iteration == 0:
            return None
        return self.statuses_by_iteration[self.selected_iteration - 1]

    @property
    def final_controls(self) -> np.ndarray | None:
        """Return controls for the selected final LMPC iteration."""
        if self.selected_iteration is None or self.selected_iteration == 0:
            return None
        return self.controls_by_iteration[self.selected_iteration - 1]

    @property
    def report_histories(self) -> list[dict[int, np.ndarray]]:
        """Return histories through the selected final iteration for plots."""
        if self.selected_iteration is None:
            return self.histories
        return self.histories[: self.selected_iteration + 1]

    @property
    def report_statuses(self) -> list[list[dict[int, str]]] | None:
        """Return statuses through the selected final iteration for plots."""
        if self.selected_iteration is None or self.selected_iteration == 0:
            return None
        return self.statuses_by_iteration[: self.selected_iteration]


def run_manta_lmpc(
    scenario: Scenario,
    *,
    config: MantaLMPCConfig = MantaLMPCConfig(),
    apf_config: APFConfig = APFConfig(),
    dynamics_config: MantaDynamicsConfig = MantaDynamicsConfig(),
    should_stop: Callable[[], bool] | None = None,
    verbose: bool = True,
) -> MantaLMPCRunResult:
    """Run APF iteration 0 followed by decentralized manta LMPC iterations."""
    starts = np.asarray(scenario.starts, dtype=float)
    goals = np.asarray(scenario.goals, dtype=float)
    num_agents = len(starts)
    if goals.shape[0] != num_agents:
        raise ValueError("scenario starts and goals must have the same agent count")
    if num_agents < 2:
        raise ValueError("manta LMPC requires at least two agents")

    if verbose:
        print("Generating staggered APF trajectories (iteration 0)...")
    safe_sets, safe_controls = build_staggered_safe_sets(
        scenario,
        dt=config.dt,
        apf_config=apf_config,
        dynamics_config=dynamics_config,
    )

    histories = [_snapshot_safe_sets(safe_sets)]
    controls_by_iteration: list[np.ndarray] = []
    statuses_by_iteration: list[list[dict[int, str]]] = []
    initial_validation = validate_trajectory(
        safe_sets,
        goals,
        scenario.safety_distance,
        scenario.obstacle.center,
        scenario.obstacle.radius,
        config.goal_tolerance,
        statuses=None,
    )
    validation_by_iteration = [initial_validation]
    success_by_iteration: list[bool] = [initial_validation.valid]
    goal_reached_by_iteration: list[bool] = [initial_validation.all_goals_reached]
    learned_by_iteration: list[bool] = [initial_validation.usable_for_learning]
    selected_iteration: int | None = 0 if initial_validation.valid else None

    if config.iterations <= 0:
        return MantaLMPCRunResult(
            scenario_name=scenario.name,
            histories=histories,
            controls_by_iteration=controls_by_iteration,
            statuses_by_iteration=statuses_by_iteration,
            success_by_iteration=success_by_iteration,
            goal_reached_by_iteration=goal_reached_by_iteration,
            learned_by_iteration=learned_by_iteration,
            validation_by_iteration=validation_by_iteration,
            selected_iteration=selected_iteration,
        )

    agents = {
        agent: MantaAgentOptimizer(
            config=config,
            num_obstacles=num_agents - 1,
            obstacle=scenario.obstacle,
            dynamics_config=dynamics_config,
        )
        for agent in range(num_agents)
    }

    for iteration in range(1, config.iterations + 1):
        _raise_if_stop_requested(should_stop)
        if verbose:
            print(f"Starting LMPC iteration {iteration}...")

        current_states = {agent: starts[agent].copy() for agent in range(num_agents)}
        history = {
            agent: [current_states[agent].copy()] for agent in range(num_agents)
        }
        iteration_controls = np.zeros((config.max_steps, num_agents, 2), dtype=float)
        iteration_statuses: list[dict[int, str]] = []
        all_reached = False
        reached_agents: set[int] = set()

        for step in range(config.max_steps):
            _raise_if_stop_requested(should_stop)
            step_start = time.time()
            for agent in range(num_agents):
                if (
                    np.linalg.norm(current_states[agent][:2] - goals[agent][:2])
                    <= config.goal_tolerance
                ):
                    reached_agents.add(agent)

            if len(reached_agents) == num_agents:
                all_reached = True
                if verbose:
                    print(
                        "  success: all agents reached the goal tolerance "
                        f"at step {step}."
                    )
                break

            reference_sets = _live_reference_sets(
                safe_sets, current_states, reached_agents, step, config
            )
            hyperplanes = _build_pairwise_hyperplanes(reference_sets, step, config)
            next_states: dict[int, np.ndarray] = {}
            step_statuses: dict[int, str] = {}

            for agent in range(num_agents):
                _raise_if_stop_requested(should_stop)
                if agent in reached_agents:
                    next_states[agent] = current_states[agent].copy()
                    step_statuses[agent] = "hold"
                    continue

                target_idx = step + config.prediction_horizon
                safe_states, safe_costs = sample_terminal_safe_set(
                    safe_sets[agent], target_idx, config.k_hull
                )
                warm_states, warm_controls = _warm_start_from_safe_set(
                    safe_sets[agent], safe_controls[agent], step, config
                )
                agent_hyperplanes = [
                    hyperplanes[agent][other]
                    for other in range(num_agents)
                    if other != agent
                ]

                try:
                    control, next_state = agents[agent].solve_step(
                        current_state=current_states[agent],
                        goal_state=goals[agent],
                        hyperplanes=agent_hyperplanes,
                        safe_states=safe_states,
                        safe_costs=safe_costs,
                        warm_states=warm_states,
                        warm_controls=warm_controls,
                    )
                    step_statuses[agent] = "ok"
                except RuntimeError as exc:
                    if _is_interrupted_solve(exc):
                        raise KeyboardInterrupt from exc
                    control, next_state = _safe_fallback_apf_step(
                        current_state=current_states[agent],
                        goal_state=goals[agent],
                        obstacle=scenario.obstacle,
                        apf_config=apf_config,
                        dt=config.dt,
                        dynamics_config=dynamics_config,
                    )
                    step_statuses[agent] = "fallback_apf"

                iteration_controls[step, agent] = control
                next_states[agent] = next_state

            for agent in range(num_agents):
                current_states[agent] = next_states[agent]
                history[agent].append(current_states[agent].copy())

            iteration_statuses.append(step_statuses)
            for agent in range(num_agents):
                if (
                    np.linalg.norm(current_states[agent][:2] - goals[agent][:2])
                    <= config.goal_tolerance
                ):
                    reached_agents.add(agent)
            all_reached = len(reached_agents) == num_agents
            if verbose and step % config.log_interval == 0:
                elapsed = time.time() - step_start
                status_text = " ".join(
                    f"A{agent}:{step_statuses.get(agent, '?')}"
                    for agent in range(num_agents)
                )
                print(
                    f"  step {step:03d}/{config.max_steps}: "
                    f"{elapsed:.2f}s | {status_text}"
                )

            if all_reached:
                if verbose:
                    print(
                        "  success: all agents reached the goal tolerance "
                        f"after step {step}."
                    )
                break

        if verbose and not all_reached:
            print(
                f"  warning: iteration {iteration} ended at "
                f"{config.max_steps} steps without all goals reached."
            )

        histories.append(_snapshot_safe_sets(history))
        controls_by_iteration.append(iteration_controls[: len(iteration_statuses)])
        statuses_by_iteration.append(iteration_statuses)
        validation = validate_trajectory(
            history,
            goals,
            scenario.safety_distance,
            scenario.obstacle.center,
            scenario.obstacle.radius,
            config.goal_tolerance,
            statuses=iteration_statuses,
        )
        validation_by_iteration.append(validation)
        goal_reached_by_iteration.append(validation.all_goals_reached)
        success_by_iteration.append(validation.valid)
        learned_by_iteration.append(validation.usable_for_learning)
        if validation.usable_for_learning:
            safe_sets = {agent: history[agent] for agent in range(num_agents)}
            safe_controls = _controls_by_agent(
                iteration_controls[: len(iteration_statuses)], num_agents
            )
            histories[-1] = _snapshot_safe_sets(safe_sets)
            if verbose:
                print("  learned: complete safe trajectory kept for the next iteration.")
        elif validation.safe and verbose:
            print(
                "  skipped learning: trajectory was safe but did not reach all goals."
            )
        if validation.valid:
            selected_iteration = len(histories) - 1
        elif verbose:
            print(
                "  invalid: keeping previous safe set "
                f"(pair violations={validation.pairwise_violation_count}, "
                f"obstacle violations={validation.obstacle_violation_count}, "
                f"fallbacks={validation.fallback_count})."
            )

    return MantaLMPCRunResult(
        scenario_name=scenario.name,
        histories=histories,
        controls_by_iteration=controls_by_iteration,
        statuses_by_iteration=statuses_by_iteration,
        success_by_iteration=success_by_iteration,
        goal_reached_by_iteration=goal_reached_by_iteration,
        learned_by_iteration=learned_by_iteration,
        validation_by_iteration=validation_by_iteration,
        selected_iteration=selected_iteration,
    )


def validate_trajectory(
    history: dict[int, list[np.ndarray] | np.ndarray],
    goals: np.ndarray,
    safety_distance: float,
    obstacle_center: tuple[float, float],
    obstacle_radius: float,
    goal_tolerance: float,
    *,
    statuses: list[dict[int, str]] | None,
) -> TrajectoryValidation:
    """Validate goal completion, pairwise separation, and obstacle clearance."""
    agents, states = _history_to_tensor(history)
    goal_pos = np.asarray(goals, dtype=float)[agents, :2]
    pos = states[:, :, :2]

    goal_error = np.linalg.norm(pos - goal_pos[None, :, :], axis=2)
    all_goals_reached = bool(np.all(goal_error[-1] <= goal_tolerance))

    pairwise_distances = []
    for i in range(pos.shape[1]):
        for j in range(i + 1, pos.shape[1]):
            pairwise_distances.append(np.linalg.norm(pos[:, i] - pos[:, j], axis=1))
    pairwise = (
        np.vstack(pairwise_distances).T
        if pairwise_distances
        else np.full((pos.shape[0], 0), np.inf)
    )
    min_pairwise = float(np.min(pairwise)) if pairwise.size else float("inf")
    pairwise_violations = int(np.sum(pairwise < safety_distance))

    center = np.asarray(obstacle_center, dtype=float)
    clearance = np.linalg.norm(pos - center[None, None, :], axis=2) - obstacle_radius
    min_clearance = float(np.min(clearance))
    obstacle_violations = int(np.sum(clearance < 0.0))

    fallback_count = 0
    if statuses is not None:
        fallback_count = sum(
            1
            for step_statuses in statuses
            for status in step_statuses.values()
            if status.startswith("fallback")
        )

    return TrajectoryValidation(
        all_goals_reached=all_goals_reached,
        min_pairwise_distance=min_pairwise,
        pairwise_violation_count=pairwise_violations,
        min_obstacle_clearance=min_clearance,
        obstacle_violation_count=obstacle_violations,
        fallback_count=fallback_count,
    )


def _history_to_tensor(
    history: dict[int, list[np.ndarray] | np.ndarray],
) -> tuple[list[int], np.ndarray]:
    """Convert ragged per-agent histories to a padded state tensor."""
    agents = sorted(history)
    max_len = max(len(history[agent]) for agent in agents)
    state_dim = np.asarray(history[agents[0]], dtype=float).shape[1]
    states = np.zeros((max_len, len(agents), state_dim), dtype=float)
    for out_agent, agent in enumerate(agents):
        traj = np.asarray(history[agent], dtype=float)
        states[: len(traj), out_agent] = traj
        if len(traj) < max_len:
            states[len(traj) :, out_agent] = traj[-1]
    return agents, states


def cost_by_iteration(
    histories: list[dict[int, np.ndarray]],
    goals: np.ndarray,
    *,
    goal_tolerance: float = 0.5,
) -> dict[int, list[int]]:
    """Return each agent's first goal-tolerance hit step per iteration."""
    g = np.asarray(goals, dtype=float)
    num_agents = g.shape[0]
    costs: dict[int, list[int]] = {agent: [] for agent in range(num_agents)}
    for run in histories:
        for agent in range(num_agents):
            traj = np.asarray(run[agent], dtype=float)
            distances = np.linalg.norm(traj[:, :2] - g[agent, :2], axis=1)
            reached = np.where(distances <= goal_tolerance)[0]
            costs[agent].append(int(reached[0]) if len(reached) else len(traj))
    return costs


def _snapshot_safe_sets(
    safe_sets: dict[int, list[np.ndarray]],
) -> dict[int, np.ndarray]:
    return {
        agent: np.asarray(states, dtype=float).copy()
        for agent, states in safe_sets.items()
    }


def _controls_by_agent(
    controls: np.ndarray, num_agents: int
) -> dict[int, np.ndarray]:
    """Convert iteration controls from ``(T, A, 2)`` to per-agent histories."""
    control_array = np.asarray(controls, dtype=float)
    return {
        agent: control_array[:, agent, :].copy()
        for agent in range(num_agents)
    }


def _live_reference_sets(
    safe_sets: dict[int, list[np.ndarray]],
    current_states: dict[int, np.ndarray],
    reached_agents: set[int],
    step: int,
    config: MantaLMPCConfig,
) -> dict[int, list[np.ndarray] | np.ndarray]:
    """Use held current states as references after individual goal arrival."""
    if not reached_agents:
        return safe_sets

    reference_sets: dict[int, list[np.ndarray] | np.ndarray] = dict(safe_sets)
    reference_len = step + config.prediction_horizon + 1
    for agent in reached_agents:
        reference_sets[agent] = np.tile(current_states[agent], (reference_len, 1))
    return reference_sets


def _build_pairwise_hyperplanes(
    safe_sets: dict[int, list[np.ndarray] | np.ndarray],
    step: int,
    config: MantaLMPCConfig,
) -> dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]:
    agents = sorted(safe_sets)
    hyperplanes: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {
        agent: {} for agent in agents
    }
    for index, i in enumerate(agents):
        for j in agents[index + 1 :]:
            H_i, h_i, H_j, h_j = get_symmetric_hyperplanes_spatial(
                step,
                step,
                config.prediction_horizon,
                safe_sets[i],
                safe_sets[j],
                safety_margin=config.hyperplane_safety_margin,
                ignore_distance=config.hyperplane_ignore_distance,
            )
            hyperplanes[i][j] = (H_i, h_i)
            hyperplanes[j][i] = (H_j, h_j)
    return hyperplanes


def _warm_start_from_safe_set(
    safe_set: list[np.ndarray] | np.ndarray,
    safe_controls: list[np.ndarray] | np.ndarray,
    step: int,
    config: MantaLMPCConfig,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(safe_set, dtype=float)
    controls = np.asarray(safe_controls, dtype=float)
    control_blend = config.warm_start_control_blend
    if not 0.0 <= control_blend <= 1.0:
        raise ValueError("lmpc.warm_start_control_blend must be in [0, 1]")
    warm_states = np.zeros((7, config.prediction_horizon + 1), dtype=float)
    warm_controls = np.full(
        (2, config.prediction_horizon),
        config.warm_start_control,
        dtype=float,
    )
    for k in range(config.prediction_horizon + 1):
        idx = min(step + k, len(states) - 1)
        warm_states[:, k] = states[idx]
    if len(controls) > 0:
        for k in range(config.prediction_horizon):
            idx = min(step + k, len(controls) - 1)
            warm_controls[:, k] = (
                (1.0 - control_blend) * warm_controls[:, k]
                + control_blend * controls[idx]
            )
    return warm_states, warm_controls


def _safe_fallback_apf_step(
    *,
    current_state: np.ndarray,
    goal_state: np.ndarray,
    obstacle: StaticObstacle,
    apf_config: APFConfig,
    dt: float,
    dynamics_config: MantaDynamicsConfig,
) -> tuple[np.ndarray, np.ndarray]:
    apf_control = compute_apf_control(
        current_state=current_state,
        goal_state=goal_state,
        obstacle=obstacle,
        apf_config=apf_config,
        dynamics_config=dynamics_config,
    )
    candidates = _fallback_control_candidates(apf_control, dynamics_config)
    scored_feasible: list[tuple[float, np.ndarray, np.ndarray]] = []
    best_any: tuple[float, np.ndarray, np.ndarray] | None = None

    for control in candidates:
        next_state = rk4_step_np(current_state, control, dt, dynamics_config)
        clearance = _inflated_obstacle_clearance(next_state, obstacle)
        goal_distance = float(np.linalg.norm(next_state[:2] - goal_state[:2]))
        score = goal_distance - 0.25 * clearance + 0.01 * float(np.linalg.norm(control))
        if clearance >= 0.0:
            scored_feasible.append((score, control, next_state))
        if best_any is None or clearance > best_any[0]:
            best_any = (clearance, control, next_state)

    if scored_feasible:
        _, control, next_state = min(scored_feasible, key=lambda item: item[0])
        return control, next_state
    if best_any is None:
        raise RuntimeError("fallback APF candidate set was empty")
    _, control, next_state = best_any
    return control, next_state


def _fallback_control_candidates(
    apf_control: np.ndarray,
    dynamics_config: MantaDynamicsConfig,
) -> list[np.ndarray]:
    mu_min = dynamics_config.mu_min
    mu_max = dynamics_config.mu_max
    low = min(mu_max, max(mu_min, 0.25))
    medium = min(mu_max, max(mu_min, 0.75))
    high = min(mu_max, max(mu_min, 1.5))
    raw = [
        apf_control,
        np.zeros(2, dtype=float),
        np.array([low, 0.0], dtype=float),
        np.array([0.0, low], dtype=float),
        np.array([medium, 0.0], dtype=float),
        np.array([0.0, medium], dtype=float),
        np.array([high, 0.0], dtype=float),
        np.array([0.0, high], dtype=float),
        np.array([low, low], dtype=float),
        np.array([medium, medium], dtype=float),
    ]
    candidates: list[np.ndarray] = []
    seen: set[tuple[float, float]] = set()
    for control in raw:
        clipped = np.clip(np.asarray(control, dtype=float), mu_min, mu_max)
        key = (round(float(clipped[0]), 6), round(float(clipped[1]), 6))
        if key not in seen:
            seen.add(key)
            candidates.append(clipped)
    return candidates


def _inflated_obstacle_clearance(state: np.ndarray, obstacle: StaticObstacle) -> float:
    position = np.asarray(state, dtype=float)[:2]
    center = np.asarray(obstacle.center, dtype=float)
    return float(np.linalg.norm(position - center) - obstacle.radius)


def _is_interrupted_solve(exc: RuntimeError) -> bool:
    """Return true when CasADi/IPOPT converted Ctrl+C into RuntimeError."""
    message = str(exc)
    return "KeyboardInterrupt" in message or "KeyboardInterruptException" in message


def _raise_if_stop_requested(should_stop: Callable[[], bool] | None) -> None:
    if should_stop is not None and should_stop():
        raise KeyboardInterrupt
