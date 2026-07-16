"""Iteration-0 guidance, MPC baseline, and admitted harbor LMPC rollouts."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .communication import LinkConfig
from .mpc import DistributedHarborMPC, HarborMPCConfig
from .simulation import HarborAgent, HarborResult, HarborSimulationConfig, run_harbor_simulation


@dataclass(frozen=True)
class HarborLearningIteration:
    """One controller rollout and its safe-set admission decision."""

    label: str
    result: HarborResult
    admitted: bool
    completion_step_sum: int
    solver_calls: int
    solver_fallbacks: int
    solve_time_seconds: float
    max_collision_slack: float
    max_terminal_slack: float
    solve_count_by_agent: dict[str, int]
    fallback_count_by_agent: dict[str, int]
    failure_steps_by_agent: dict[str, list[int]]
    failure_status_counts: dict[str, int]


def run_distributed_harbor_lmpc(
    agents: list[HarborAgent],
    simulation: HarborSimulationConfig,
    communication: LinkConfig,
    mpc_config: HarborMPCConfig,
) -> list[HarborLearningIteration]:
    """Run guidance, plain MPC, and repeated safe-set LMPC experiments."""
    optimized_simulation = replace(
        simulation,
        guidance_update_interval_steps=mpc_config.replan_interval_steps,
    )
    guidance = run_harbor_simulation(agents, simulation, communication)
    if not _admissible(guidance):
        raise RuntimeError("harbor iteration-0 guidance rollout is not safe and complete")
    iterations = [
        _record(
            "guidance_iter_0",
            guidance,
            admitted=True,
            simulation=simulation,
        )
    ]
    safe_states = guidance.states
    safe_controls = guidance.controls
    best_cost = _completion_cost(guidance, simulation.horizon)

    mpc_controller = DistributedHarborMPC(
        agents=agents,
        config=mpc_config,
        dt=simulation.dt,
        safe_states=safe_states,
        safe_controls=safe_controls,
        learning=False,
    )
    mpc_result = run_harbor_simulation(
        agents,
        optimized_simulation,
        communication,
        control_provider=mpc_controller,
    )
    mpc_admitted = _controller_admissible(mpc_result, mpc_controller)
    mpc_cost = _completion_cost(mpc_result, simulation.horizon)
    iterations.append(
        _controller_record(
            "distributed_mpc",
            mpc_result,
            mpc_controller,
            admitted=mpc_admitted,
            simulation=simulation,
        )
    )
    if mpc_config.seed_learning_from_mpc and mpc_admitted and mpc_cost <= best_cost:
        safe_states = mpc_result.states
        safe_controls = mpc_result.controls
        best_cost = mpc_cost

    for iteration in range(1, mpc_config.learning_iterations + 1):
        controller = DistributedHarborMPC(
            agents=agents,
            config=mpc_config,
            dt=simulation.dt,
            safe_states=safe_states,
            safe_controls=safe_controls,
            learning=True,
        )
        result = run_harbor_simulation(
            agents,
            optimized_simulation,
            communication,
            control_provider=controller,
        )
        cost = _completion_cost(result, simulation.horizon)
        admitted = _controller_admissible(result, controller) and cost <= best_cost
        iterations.append(
            _controller_record(
                f"distributed_lmpc_{iteration}",
                result,
                controller,
                admitted=admitted,
                simulation=simulation,
            )
        )
        if admitted:
            safe_states = result.states
            safe_controls = result.controls
            best_cost = cost
    return iterations


def _controller_record(
    label,
    result,
    controller,
    *,
    admitted,
    simulation,
) -> HarborLearningIteration:
    return _record(
        label,
        result,
        admitted=admitted,
        simulation=simulation,
        solver_calls=controller.solve_count,
        solver_fallbacks=controller.fallback_count,
        solve_time_seconds=controller.solve_time_seconds,
        max_collision_slack=controller.max_collision_slack,
        max_terminal_slack=controller.max_terminal_slack,
        solve_count_by_agent=controller.solve_count_by_agent,
        fallback_count_by_agent=controller.fallback_count_by_agent,
        failure_steps_by_agent=controller.failure_steps_by_agent,
        failure_status_counts=controller.failure_status_counts,
    )


def _record(
    label,
    result,
    *,
    admitted,
    simulation,
    solver_calls=0,
    solver_fallbacks=0,
    solve_time_seconds=0.0,
    max_collision_slack=0.0,
    max_terminal_slack=0.0,
    solve_count_by_agent=None,
    fallback_count_by_agent=None,
    failure_steps_by_agent=None,
    failure_status_counts=None,
) -> HarborLearningIteration:
    return HarborLearningIteration(
        label=label,
        result=result,
        admitted=admitted,
        completion_step_sum=_completion_cost(result, simulation.horizon),
        solver_calls=solver_calls,
        solver_fallbacks=solver_fallbacks,
        solve_time_seconds=solve_time_seconds,
        max_collision_slack=max_collision_slack,
        max_terminal_slack=max_terminal_slack,
        solve_count_by_agent=solve_count_by_agent or {},
        fallback_count_by_agent=fallback_count_by_agent or {},
        failure_steps_by_agent=failure_steps_by_agent or {},
        failure_status_counts=failure_status_counts or {},
    )


def _admissible(result: HarborResult) -> bool:
    return result.all_goals_reached and result.pairwise_violation_count == 0


def _controller_admissible(result: HarborResult, controller) -> bool:
    """Reject physically invalid or solver-contaminated learning data."""
    return (
        _admissible(result)
        and controller.fallback_count == 0
        and controller.max_collision_slack <= 1e-9
    )


def _completion_cost(result: HarborResult, horizon: int) -> int:
    return sum(
        step if step is not None else horizon + 1
        for step in result.first_goal_steps.values()
    )
