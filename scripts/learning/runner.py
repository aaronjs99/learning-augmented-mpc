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
class MantaLMPCRunResult:
    """In-memory result from one config-driven APF/LMPC run."""

    scenario_name: str
    histories: list[dict[int, np.ndarray]]
    controls_by_iteration: list[np.ndarray]
    statuses_by_iteration: list[list[dict[int, str]]]
    success_by_iteration: list[bool]

    @property
    def final_history(self) -> dict[int, np.ndarray]:
        """Return the most recent per-agent state history."""
        return self.histories[-1]


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
    if verbose:
        print("Generating staggered APF trajectories (iteration 0)...")
    safe_sets, _ = build_staggered_safe_sets(
        scenario,
        dt=config.dt,
        apf_config=apf_config,
        dynamics_config=dynamics_config,
    )

    histories = [_snapshot_safe_sets(safe_sets)]
    controls_by_iteration: list[np.ndarray] = []
    statuses_by_iteration: list[list[dict[int, str]]] = []
    success_by_iteration: list[bool] = [True]

    if config.iterations <= 0:
        return MantaLMPCRunResult(
            scenario_name=scenario.name,
            histories=histories,
            controls_by_iteration=controls_by_iteration,
            statuses_by_iteration=statuses_by_iteration,
            success_by_iteration=success_by_iteration,
        )

    agents = {
        agent: MantaAgentOptimizer(
            config=config,
            num_obstacles=2,
            obstacle=scenario.obstacle,
            dynamics_config=dynamics_config,
        )
        for agent in range(3)
    }

    starts = np.asarray(scenario.starts, dtype=float)
    goals = np.asarray(scenario.goals, dtype=float)

    for iteration in range(1, config.iterations + 1):
        _raise_if_stop_requested(should_stop)
        if verbose:
            print(f"Starting LMPC iteration {iteration}...")

        current_states = {agent: starts[agent].copy() for agent in range(3)}
        history = {agent: [current_states[agent].copy()] for agent in range(3)}
        iteration_controls = np.zeros((config.max_steps, 3, 2), dtype=float)
        iteration_statuses: list[dict[int, str]] = []
        all_reached = False

        for step in range(config.max_steps):
            _raise_if_stop_requested(should_stop)
            step_start = time.time()
            hyperplanes = _build_pairwise_hyperplanes(safe_sets, step, config)
            next_states: dict[int, np.ndarray] = {}
            step_statuses: dict[int, str] = {}
            all_reached = True

            for agent in range(3):
                _raise_if_stop_requested(should_stop)
                dist_to_goal = np.linalg.norm(
                    current_states[agent][:2] - goals[agent][:2]
                )
                if dist_to_goal <= config.goal_tolerance:
                    next_states[agent] = current_states[agent].copy()
                    step_statuses[agent] = "hold"
                    continue

                all_reached = False
                target_idx = step + config.prediction_horizon
                safe_states, safe_costs = sample_terminal_safe_set(
                    safe_sets[agent], target_idx, config.k_hull
                )
                warm_states, warm_controls = _warm_start_from_safe_set(
                    safe_sets[agent], step, config
                )
                agent_hyperplanes = [
                    hyperplanes[agent][other] for other in range(3) if other != agent
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

            for agent in range(3):
                current_states[agent] = next_states[agent]
                history[agent].append(current_states[agent].copy())

            iteration_statuses.append(step_statuses)
            if verbose and step % config.log_interval == 0:
                elapsed = time.time() - step_start
                status_text = " ".join(
                    f"A{agent}:{step_statuses.get(agent, '?')}" for agent in range(3)
                )
                print(
                    f"  step {step:03d}/{config.max_steps}: "
                    f"{elapsed:.2f}s | {status_text}"
                )

            if all_reached:
                if verbose:
                    print(
                        "  success: all agents reached the goal tolerance "
                        f"and are holding goals at step {step}."
                    )
                break

        if verbose and not all_reached:
            print(
                f"  warning: iteration {iteration} ended at "
                f"{config.max_steps} steps without all goals reached."
            )

        safe_sets = {agent: history[agent] for agent in range(3)}
        histories.append(_snapshot_safe_sets(safe_sets))
        controls_by_iteration.append(iteration_controls[: len(iteration_statuses)])
        statuses_by_iteration.append(iteration_statuses)
        success_by_iteration.append(all_reached)

    return MantaLMPCRunResult(
        scenario_name=scenario.name,
        histories=histories,
        controls_by_iteration=controls_by_iteration,
        statuses_by_iteration=statuses_by_iteration,
        success_by_iteration=success_by_iteration,
    )


def cost_by_iteration(
    histories: list[dict[int, np.ndarray]],
    goals: np.ndarray,
    *,
    goal_tolerance: float = 0.5,
) -> dict[int, list[int]]:
    """Return each agent's first goal-tolerance hit step per iteration."""
    g = np.asarray(goals, dtype=float)
    costs: dict[int, list[int]] = {agent: [] for agent in range(3)}
    for run in histories:
        for agent in range(3):
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


def _build_pairwise_hyperplanes(
    safe_sets: dict[int, list[np.ndarray]],
    step: int,
    config: MantaLMPCConfig,
) -> dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]:
    hyperplanes: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {
        agent: {} for agent in range(3)
    }
    for i in range(3):
        for j in range(i + 1, 3):
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
    step: int,
    config: MantaLMPCConfig,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(safe_set, dtype=float)
    warm_states = np.zeros((7, config.prediction_horizon + 1), dtype=float)
    warm_controls = np.full(
        (2, config.prediction_horizon),
        config.warm_start_control,
        dtype=float,
    )
    for k in range(config.prediction_horizon + 1):
        idx = min(step + k, len(states) - 1)
        warm_states[:, k] = states[idx]
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
