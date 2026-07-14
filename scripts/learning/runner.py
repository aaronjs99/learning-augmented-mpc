"""Composable runner for APF baseline and repeated manta LMPC iterations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time

import numpy as np

from scripts.dynamics import MantaDynamicsConfig
from scripts.metrics import TrajectoryValidation, validate_trajectory
from scripts.mpc.manta_lmpc import MantaAgentOptimizer, MantaLMPCConfig
from scripts.simulation import Scenario

from .apf import APFConfig
from .hyperplanes import get_symmetric_hyperplanes_spatial
from .policies import priority_margins, warm_start_from_safe_set
from .recovery import repair_incomplete_with_apf, safe_fallback_apf_step
from .safe_sets import build_staggered_safe_sets, sample_terminal_safe_set


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
            hyperplanes = _build_pairwise_hyperplanes(
                reference_sets,
                step,
                config,
                current_states=current_states,
                goals=goals,
            )
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
                warm_states, warm_controls = warm_start_from_safe_set(
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
                    control, next_state = safe_fallback_apf_step(
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

        main_controls = iteration_controls[: len(iteration_statuses)]
        stored_controls = main_controls
        if not all_reached and config.repair_incomplete_with_apf:
            recovery = repair_incomplete_with_apf(
                history=history,
                current_states=current_states,
                safe_sets=safe_sets,
                reached_agents=reached_agents,
                goals=goals,
                scenario=scenario,
                config=config,
                apf_config=apf_config,
                dynamics_config=dynamics_config,
                should_stop=should_stop,
                verbose=verbose,
            )
            if recovery.statuses:
                iteration_statuses.extend(recovery.statuses)
                stored_controls = np.vstack((main_controls, recovery.controls))
                all_reached = len(reached_agents) == num_agents

        histories.append(_snapshot_safe_sets(history))
        controls_by_iteration.append(stored_controls)
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
            safe_controls = _controls_by_agent(stored_controls, num_agents)
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
    if control_array.ndim != 3 or control_array.shape[1:] != (num_agents, 2):
        raise ValueError(
            f"controls must have shape (T, {num_agents}, 2), "
            f"got {control_array.shape}"
        )
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
    *,
    current_states: dict[int, np.ndarray],
    goals: np.ndarray,
) -> dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]:
    agents = sorted(safe_sets)
    hyperplanes: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {
        agent: {} for agent in agents
    }
    for index, i in enumerate(agents):
        for j in agents[index + 1 :]:
            margin_i, margin_j = priority_margins(
                i, j, safe_sets, step, current_states, goals, config
            )
            H_i, h_i, H_j, h_j = get_symmetric_hyperplanes_spatial(
                step,
                step,
                config.prediction_horizon,
                safe_sets[i],
                safe_sets[j],
                safety_margin=config.hyperplane_safety_margin,
                safety_margin_i=margin_i,
                safety_margin_j=margin_j,
                ignore_distance=config.hyperplane_ignore_distance,
            )
            hyperplanes[i][j] = (H_i, h_i)
            hyperplanes[j][i] = (H_j, h_j)
    return hyperplanes


def _is_interrupted_solve(exc: RuntimeError) -> bool:
    """Return true when CasADi/IPOPT converted Ctrl+C into RuntimeError."""
    message = str(exc)
    return "KeyboardInterrupt" in message or "KeyboardInterruptException" in message


def _raise_if_stop_requested(should_stop: Callable[[], bool] | None) -> None:
    if should_stop is not None and should_stop():
        raise KeyboardInterrupt
