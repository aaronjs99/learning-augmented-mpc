"""Controlled communication ablations for heterogeneous harbor coordination."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .communication import LinkConfig
from .config import (
    HarborFaultEnsembleConfig,
    HarborFaultStudyConfig,
    HarborTemporaryFaultEnsembleConfig,
    HarborTimeVaryingFaultConfig,
)
from .learning import run_distributed_harbor_lmpc
from .mpc import DistributedHarborMPC, HarborMPCConfig
from .simulation import (
    HarborAgent,
    HarborDisturbanceConfig,
    HarborObservationNoiseConfig,
    HarborResult,
    HarborSimulationConfig,
    run_harbor_simulation,
)


@dataclass(frozen=True)
class HarborRobustnessTrial:
    """One matched plant-mismatch trial and its controller telemetry."""

    label: str
    identification_strategy: str
    result: HarborResult
    valid: bool
    source_controller: str | None
    completion_step_sum: int
    solver_fallbacks: int
    solver_fallbacks_by_agent: dict[str, int]
    solver_failure_steps_by_agent: dict[str, list[int]]
    solver_failure_status_counts: dict[str, int]
    max_collision_slack: float
    residual_history: dict[str, np.ndarray]
    final_residual_estimates: dict[str, np.ndarray]
    effectiveness_history: dict[str, np.ndarray]
    final_effectiveness_estimates: dict[str, np.ndarray]
    effectiveness_change_steps_by_agent: dict[str, list[int]]
    excitation_history: dict[str, np.ndarray]
    information_std_history: dict[str, np.ndarray]
    probe_count_by_agent: dict[str, int]
    probe_channel_counts: dict[str, np.ndarray]
    probe_sequence_by_agent: dict[str, list[int]]
    probe_rejection_counts: dict[str, np.ndarray]


@dataclass(frozen=True)
class HarborFaultEnsembleCase:
    """One hidden actuator-loss draw and its matched controller trials."""

    seed: int
    disturbance: HarborDisturbanceConfig
    observation_seed: int | None
    trials: tuple[HarborRobustnessTrial, ...]


def sweep_network_robustness(
    agents: list[HarborAgent],
    simulation: HarborSimulationConfig,
    communication: LinkConfig,
    *,
    delays: list[int],
    dropout_probabilities: list[float],
    seeds: list[int],
) -> list[dict[str, float | int]]:
    """Measure safety and completion across delay/dropout network conditions."""
    if not delays or not dropout_probabilities or not seeds:
        raise ValueError("network sweep axes and seeds must not be empty")
    records = []
    for delay in delays:
        for dropout in dropout_probabilities:
            results = [
                run_harbor_simulation(
                    agents,
                    simulation,
                    replace(
                        communication,
                        enabled=True,
                        delay_steps=delay,
                        dropout_probability=dropout,
                        seed=seed,
                    ),
                )
                for seed in seeds
            ]
            completion_costs = [
                sum(
                    (
                        result.first_goal_steps[name]
                        if result.first_goal_steps[name] is not None
                        and result.final_goal_errors[name]
                        <= simulation.goal_tolerance
                        else simulation.horizon + 1
                    )
                    for name in result.first_goal_steps
                )
                for result in results
            ]
            delivery_ratios = [
                result.messages_delivered / max(result.messages_sent, 1)
                for result in results
            ]
            records.append(
                {
                    "delay_steps": delay,
                    "dropout_probability": dropout,
                    "trials": len(results),
                    "safe_rate": float(
                        np.mean(
                            [result.pairwise_violation_count == 0 for result in results]
                        )
                    ),
                    "completion_rate": float(
                        np.mean([result.all_goals_reached for result in results])
                    ),
                    "mean_completion_step_sum": float(np.mean(completion_costs)),
                    "worst_min_pairwise_distance": float(
                        min(result.min_pairwise_distance for result in results)
                    ),
                    "mean_delivery_ratio": float(np.mean(delivery_ratios)),
                }
            )
    return records


def sweep_prediction_horizons(
    agents: list[HarborAgent],
    simulation: HarborSimulationConfig,
    communication: LinkConfig,
    mpc_config: HarborMPCConfig,
    *,
    horizons: list[int],
) -> list[dict[str, float | int | str | bool]]:
    """Compare matched distributed MPC and LMPC across horizon lengths."""
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("prediction horizons must be positive")
    records = []
    for horizon in horizons:
        iterations = run_distributed_harbor_lmpc(
            agents,
            simulation,
            communication,
            replace(
                mpc_config,
                prediction_horizon=horizon,
                learning_iterations=1,
            ),
        )
        for iteration in iterations[1:]:
            records.append(
                {
                    "prediction_horizon": horizon,
                    "controller": (
                        "MPC"
                        if iteration.label == "distributed_mpc"
                        else "LMPC"
                    ),
                    "complete": iteration.result.all_goals_reached,
                    "admitted": iteration.admitted,
                    "completion_step_sum": iteration.completion_step_sum,
                    "solve_time_seconds": iteration.solve_time_seconds,
                    "solver_calls": iteration.solver_calls,
                    "mean_solve_time_ms": (
                        1000.0
                        * iteration.solve_time_seconds
                        / max(iteration.solver_calls, 1)
                    ),
                    "solver_fallbacks": iteration.solver_fallbacks,
                    "pairwise_violation_count": (
                        iteration.result.pairwise_violation_count
                    ),
                    "min_pairwise_distance": (
                        iteration.result.min_pairwise_distance
                    ),
                    "max_terminal_slack": iteration.max_terminal_slack,
                }
            )
    return records


def run_model_mismatch_study(
    agents: list[HarborAgent],
    simulation: HarborSimulationConfig,
    communication: LinkConfig,
    mpc_config: HarborMPCConfig,
    disturbance: HarborDisturbanceConfig,
) -> list[HarborRobustnessTrial]:
    """Compare nominal and residual-adaptive controllers on one hidden plant."""
    seed_iterations = run_distributed_harbor_lmpc(
        agents,
        simulation,
        communication,
        replace(
            mpc_config,
            learning_iterations=1,
            residual_adaptation=False,
            control_effectiveness_adaptation=False,
        ),
    )
    seed = min(
        (record for record in seed_iterations if record.admitted),
        key=lambda record: record.completion_step_sum,
    )
    evaluation = replace(
        simulation,
        guidance_update_interval_steps=mpc_config.replan_interval_steps,
        goal_hold_steps=disturbance.evaluation_hold_steps,
    )
    definitions = (
        ("Nominal MPC", False, False, False),
        ("Residual-adaptive MPC", True, False, False),
        ("Joint-adaptive MPC", True, True, False),
        ("Joint-adaptive LMPC", True, True, True),
    )
    trials = []
    for label, residual_adaptive, effectiveness_adaptive, learning in definitions:
        controller = DistributedHarborMPC(
            agents=agents,
            config=replace(
                mpc_config,
                residual_adaptation=residual_adaptive,
                control_effectiveness_adaptation=effectiveness_adaptive,
            ),
            dt=simulation.dt,
            safe_states=seed.result.states,
            safe_controls=seed.result.controls,
            learning=learning,
        )
        result = run_harbor_simulation(
            agents,
            evaluation,
            communication,
            control_provider=controller,
            disturbance=disturbance,
        )
        completion_cost = sum(
            step if step is not None else simulation.horizon + 1
            for step in result.first_goal_steps.values()
        )
        valid = (
            result.all_goals_reached
            and result.pairwise_violation_count == 0
            and controller.fallback_count == 0
            and controller.max_collision_slack <= 1e-9
        )
        trials.append(
            HarborRobustnessTrial(
                label=label,
                identification_strategy=controller.config.identification_strategy,
                result=result,
                valid=valid,
                source_controller=None,
                completion_step_sum=completion_cost,
                solver_fallbacks=controller.fallback_count,
                solver_fallbacks_by_agent=controller.fallback_count_by_agent.copy(),
                solver_failure_steps_by_agent={
                    name: steps.copy()
                    for name, steps in controller.failure_steps_by_agent.items()
                },
                solver_failure_status_counts=controller.failure_status_counts.copy(),
                max_collision_slack=controller.max_collision_slack,
                residual_history={
                    name: np.asarray(values, dtype=float)
                    for name, values in controller.residual_history.items()
                },
                final_residual_estimates={
                    name: value.copy()
                    for name, value in controller.position_drift_estimates.items()
                },
                effectiveness_history={
                    name: np.asarray(values, dtype=float)
                    for name, values in controller.effectiveness_history.items()
                },
                final_effectiveness_estimates={
                    name: value.copy()
                    for name, value in controller.control_effectiveness_estimates.items()
                },
                effectiveness_change_steps_by_agent={
                    name: steps.copy()
                    for name, steps in controller.effectiveness_change_steps_by_agent.items()
                },
                excitation_history={
                    name: np.asarray(values, dtype=float)
                    for name, values in controller.excitation_history.items()
                },
                information_std_history={
                    name: np.asarray(values, dtype=float)
                    for name, values in controller.information_std_history.items()
                },
                probe_count_by_agent=dict(
                    controller.identification_probe_count_by_agent
                ),
                probe_channel_counts={
                    name: value.copy()
                    for name, value in controller.identification_probe_channel_counts.items()
                },
                probe_sequence_by_agent={
                    name: list(value)
                    for name, value in controller.identification_probe_sequence_by_agent.items()
                },
                probe_rejection_counts={
                    name: value.copy()
                    for name, value in controller.identification_probe_rejection_counts.items()
                },
            )
        )
        print(
            f"{label}: complete={result.all_goals_reached}, "
            f"safe={result.pairwise_violation_count == 0}, "
            f"cost={completion_cost}, fallbacks={controller.fallback_count}",
            flush=True,
        )
    return trials


def run_actuator_fault_study(
    agents: list[HarborAgent],
    simulation: HarborSimulationConfig,
    communication: LinkConfig,
    mpc_config: HarborMPCConfig,
    disturbance: HarborDisturbanceConfig,
    study_config: HarborFaultStudyConfig,
) -> list[HarborRobustnessTrial]:
    """Compare scalar and diagonal local estimators under asymmetric faults."""
    study_mpc = replace(
        mpc_config,
        prediction_horizon=study_config.prediction_horizon,
        terminal_goal_weight=study_config.terminal_goal_weight,
        terminal_slack_bound=study_config.terminal_slack_bound,
        terminal_slack_weight=study_config.terminal_slack_weight,
    )
    seed_iterations = run_distributed_harbor_lmpc(
        agents,
        simulation,
        communication,
        replace(
            study_mpc,
            learning_iterations=1,
            residual_adaptation=False,
            control_effectiveness_adaptation=False,
        ),
    )
    seed = min(
        (record for record in seed_iterations if record.admitted),
        key=lambda record: record.completion_step_sum,
    )
    evaluation = replace(
        simulation,
        guidance_update_interval_steps=study_mpc.replan_interval_steps,
        goal_hold_steps=disturbance.evaluation_hold_steps,
    )
    definitions = (
        ("Nominal MPC", False, "scalar", False, False, None, "energy", 2),
        ("Scalar-adaptive MPC", True, "scalar", False, False, None, "energy", 2),
        ("Diagonal-adaptive MPC", True, "diagonal", False, False, None, "energy", 2),
        ("Active diagonal MPC", True, "diagonal", False, True, None, "energy", 2),
        (
            "One-pass active MPC",
            True,
            "diagonal",
            False,
            True,
            None,
            "energy",
            1,
        ),
        (
            "Information-aware MPC",
            True,
            "diagonal",
            False,
            True,
            None,
            "information",
            1,
        ),
        (
            "Retained diagonal LMPC",
            False,
            "diagonal",
            True,
            False,
            "Diagonal-adaptive MPC",
            "energy",
            2,
        ),
        (
            "Retained active-ID LMPC",
            False,
            "diagonal",
            True,
            False,
            "Active diagonal MPC",
            "energy",
            2,
        ),
        (
            "Retained information-ID LMPC",
            False,
            "diagonal",
            True,
            False,
            "Information-aware MPC",
            "information",
            1,
        ),
    )
    return _run_fault_trials(
        agents,
        simulation,
        evaluation,
        communication,
        study_mpc,
        disturbance,
        seed.result,
        definitions,
    )


def generate_fault_ensemble(
    agents: list[HarborAgent],
    base_disturbance: HarborDisturbanceConfig,
    config: HarborFaultEnsembleConfig,
) -> list[tuple[int, HarborDisturbanceConfig]]:
    """Generate deterministic Latin-hypercube actuator losses by channel."""
    channel_count = sum(agent.model.control_dim for agent in agents)
    case_count = len(config.seeds)
    rng = np.random.default_rng(np.random.SeedSequence(config.seeds))
    unit_samples = np.empty((case_count, channel_count), dtype=float)
    for channel in range(channel_count):
        strata = rng.permutation(case_count)
        unit_samples[:, channel] = (strata + rng.random(case_count)) / case_count
    samples = config.effectiveness_min + unit_samples * (
        config.effectiveness_max - config.effectiveness_min
    )

    cases = []
    for row, seed_value in zip(samples, config.seeds, strict=True):
        offset = 0
        effectiveness = {}
        for agent in agents:
            dimension = agent.model.control_dim
            effectiveness[agent.name] = tuple(row[offset : offset + dimension])
            offset += dimension
        cases.append(
            (
                seed_value,
                replace(
                    base_disturbance,
                    agent_control_effectiveness=effectiveness,
                ),
            )
        )
    return cases


def generate_temporary_fault_ensemble(
    agents: list[HarborAgent],
    base_disturbance: HarborDisturbanceConfig,
    config: HarborTemporaryFaultEnsembleConfig,
) -> list[tuple[int, int, HarborDisturbanceConfig]]:
    """Generate matched temporary losses with stratified timing and severity."""
    case_count = len(config.seeds)
    channel_count = sum(agent.model.control_dim for agent in agents)
    timing_count = 2 * len(agents)
    rng = np.random.default_rng(np.random.SeedSequence(config.seeds))
    unit_samples = np.empty((case_count, channel_count + timing_count), dtype=float)
    for dimension in range(unit_samples.shape[1]):
        strata = rng.permutation(case_count)
        unit_samples[:, dimension] = (strata + rng.random(case_count)) / case_count

    effectiveness = config.effectiveness_min + unit_samples[:, :channel_count] * (
        config.effectiveness_max - config.effectiveness_min
    )
    cases = []
    for case_index, seed_value in enumerate(config.seeds):
        schedule = {}
        channel_offset = 0
        for agent_index, agent in enumerate(agents):
            dimension = agent.model.control_dim
            degraded = tuple(
                effectiveness[
                    case_index, channel_offset : channel_offset + dimension
                ]
            )
            onset_unit = unit_samples[case_index, channel_count + 2 * agent_index]
            duration_unit = unit_samples[
                case_index, channel_count + 2 * agent_index + 1
            ]
            onset = config.onset_step_min + int(
                onset_unit * (config.onset_step_max - config.onset_step_min + 1)
            )
            duration = config.duration_step_min + int(
                duration_unit
                * (config.duration_step_max - config.duration_step_min + 1)
            )
            schedule[agent.name] = (
                (onset, degraded),
                (onset + duration, tuple(np.ones(dimension))),
            )
            channel_offset += dimension
        cases.append(
            (
                seed_value,
                config.observation_seed_offset + seed_value,
                replace(
                    base_disturbance,
                    agent_control_effectiveness={},
                    agent_control_effectiveness_schedule=schedule,
                ),
            )
        )
    return cases


def run_actuator_fault_generalization(
    agents: list[HarborAgent],
    simulation: HarborSimulationConfig,
    communication: LinkConfig,
    mpc_config: HarborMPCConfig,
    base_disturbance: HarborDisturbanceConfig,
    study_config: HarborFaultStudyConfig,
    ensemble_config: HarborFaultEnsembleConfig,
    observation_noise: HarborObservationNoiseConfig | None = None,
) -> list[HarborFaultEnsembleCase]:
    """Compare equal-budget probe policies over stratified hidden faults."""
    study_mpc = replace(
        mpc_config,
        prediction_horizon=study_config.prediction_horizon,
        terminal_goal_weight=study_config.terminal_goal_weight,
        terminal_slack_bound=study_config.terminal_slack_bound,
        terminal_slack_weight=study_config.terminal_slack_weight,
    )
    seed_iterations = run_distributed_harbor_lmpc(
        agents,
        simulation,
        communication,
        replace(
            study_mpc,
            learning_iterations=1,
            residual_adaptation=False,
            control_effectiveness_adaptation=False,
        ),
    )
    seed = min(
        (record for record in seed_iterations if record.admitted),
        key=lambda record: record.completion_step_sum,
    )
    if observation_noise is not None and observation_noise.enabled:
        definitions = (
            ("Instantaneous diagonal MPC", True, "diagonal", False, False, None, "energy", 1),
            ("Recursive diagonal MPC", True, "recursive_diagonal", False, False, None, "energy", 1),
            ("Recursive one-pass MPC", True, "recursive_diagonal", False, True, None, "energy", 1),
            (
                "Recursive information-aware MPC",
                True,
                "recursive_diagonal",
                False,
                True,
                None,
                "information",
                1,
            ),
        )
    else:
        definitions = (
            ("Passive diagonal MPC", True, "diagonal", False, False, None, "energy", 1),
            ("One-pass active MPC", True, "diagonal", False, True, None, "energy", 1),
            (
                "Information-aware MPC",
                True,
                "diagonal",
                False,
                True,
                None,
                "information",
                1,
            ),
        )
    cases = []
    for seed_value, disturbance in generate_fault_ensemble(
        agents, base_disturbance, ensemble_config
    ):
        case_noise = (
            replace(observation_noise, seed=observation_noise.seed + seed_value)
            if observation_noise is not None and observation_noise.enabled
            else None
        )
        evaluation = replace(
            simulation,
            guidance_update_interval_steps=study_mpc.replan_interval_steps,
            goal_hold_steps=disturbance.evaluation_hold_steps,
        )
        print(f"Fault ensemble seed {seed_value}", flush=True)
        trials = _run_fault_trials(
            agents,
            simulation,
            evaluation,
            communication,
            study_mpc,
            disturbance,
            seed.result,
            definitions,
            observation_noise=case_noise,
        )
        cases.append(
            HarborFaultEnsembleCase(
                seed=seed_value,
                disturbance=disturbance,
                observation_seed=(case_noise.seed if case_noise is not None else None),
                trials=tuple(trials),
            )
        )
    return cases


def run_obstacle_prediction_generalization(
    agents: list[HarborAgent],
    simulation: HarborSimulationConfig,
    communication: LinkConfig,
    mpc_config: HarborMPCConfig,
    base_disturbance: HarborDisturbanceConfig,
    study_config: HarborFaultStudyConfig,
    ensemble_config: HarborFaultEnsembleConfig,
    observation_noise: HarborObservationNoiseConfig,
) -> list[HarborFaultEnsembleCase]:
    """Compare peer-motion predictors from one common clean trajectory seed."""
    study_mpc = replace(
        mpc_config,
        prediction_horizon=study_config.prediction_horizon,
        terminal_goal_weight=study_config.terminal_goal_weight,
        terminal_slack_bound=study_config.terminal_slack_bound,
        terminal_slack_weight=study_config.terminal_slack_weight,
    )
    seed_iterations = run_distributed_harbor_lmpc(
        agents,
        simulation,
        communication,
        replace(
            study_mpc,
            obstacle_prediction_mode="constant_velocity",
            learning_iterations=1,
            residual_adaptation=False,
            control_effectiveness_adaptation=False,
        ),
    )
    seed = min(
        (record for record in seed_iterations if record.admitted),
        key=lambda record: record.completion_step_sum,
    )
    modes = (
        ("Constant-velocity prediction", "constant_velocity"),
        ("Goal-bounded prediction", "goal_bounded_velocity"),
    )
    cases = []
    for seed_value, disturbance in generate_fault_ensemble(
        agents, base_disturbance, ensemble_config
    ):
        case_noise = replace(
            observation_noise,
            enabled=True,
            seed=observation_noise.seed + seed_value,
        )
        evaluation = replace(
            simulation,
            guidance_update_interval_steps=study_mpc.replan_interval_steps,
            goal_hold_steps=disturbance.evaluation_hold_steps,
        )
        trials = []
        print(f"Prediction ensemble seed {seed_value}", flush=True)
        for label, mode in modes:
            trials.extend(
                _run_fault_trials(
                    agents,
                    simulation,
                    evaluation,
                    communication,
                    replace(study_mpc, obstacle_prediction_mode=mode),
                    disturbance,
                    seed.result,
                    (
                        (
                            label,
                            True,
                            "recursive_diagonal",
                            False,
                            False,
                            None,
                            "energy",
                            1,
                        ),
                    ),
                    observation_noise=case_noise,
                )
            )
        cases.append(
            HarborFaultEnsembleCase(
                seed=seed_value,
                disturbance=disturbance,
                observation_seed=case_noise.seed,
                trials=tuple(trials),
            )
        )
    return cases


def run_time_varying_fault_study(
    agents: list[HarborAgent],
    simulation: HarborSimulationConfig,
    communication: LinkConfig,
    mpc_config: HarborMPCConfig,
    disturbance: HarborDisturbanceConfig,
    study_config: HarborFaultStudyConfig,
    experiment_config: HarborTimeVaryingFaultConfig,
    observation_noise: HarborObservationNoiseConfig,
) -> list[HarborFaultEnsembleCase]:
    """Compare fixed, threshold-adaptive, and CUSUM-adaptive scheduled-fault RLS."""
    study_mpc = replace(
        mpc_config,
        prediction_horizon=study_config.prediction_horizon,
        terminal_goal_weight=study_config.terminal_goal_weight,
        terminal_slack_bound=study_config.terminal_slack_bound,
        terminal_slack_weight=study_config.terminal_slack_weight,
    )
    seed_iterations = run_distributed_harbor_lmpc(
        agents,
        simulation,
        communication,
        replace(
            study_mpc,
            learning_iterations=1,
            residual_adaptation=False,
            control_effectiveness_adaptation=False,
        ),
    )
    seed = min(
        (record for record in seed_iterations if record.admitted),
        key=lambda record: record.completion_step_sum,
    )
    evaluation = replace(
        simulation,
        guidance_update_interval_steps=study_mpc.replan_interval_steps,
        goal_hold_steps=disturbance.evaluation_hold_steps,
    )
    cases = []
    for observation_seed in experiment_config.observation_seeds:
        noise = replace(observation_noise, enabled=True, seed=observation_seed)
        print(f"Scheduled-fault observation seed {observation_seed}", flush=True)
        trials = _run_temporary_fault_trials(
            agents,
            simulation,
            evaluation,
            communication,
            study_mpc,
            disturbance,
            seed.result,
            experiment_config,
            noise,
        )
        cases.append(
            HarborFaultEnsembleCase(
                seed=observation_seed,
                disturbance=disturbance,
                observation_seed=observation_seed,
                trials=tuple(trials),
            )
        )
    return cases


def run_temporary_fault_generalization(
    agents: list[HarborAgent],
    simulation: HarborSimulationConfig,
    communication: LinkConfig,
    mpc_config: HarborMPCConfig,
    base_disturbance: HarborDisturbanceConfig,
    study_config: HarborFaultStudyConfig,
    experiment_config: HarborTimeVaryingFaultConfig,
    ensemble_config: HarborTemporaryFaultEnsembleConfig,
    observation_noise: HarborObservationNoiseConfig,
    controller_labels: tuple[str, ...] | None = None,
) -> list[HarborFaultEnsembleCase]:
    """Evaluate temporary-fault adaptation across hidden severities and timings."""
    study_mpc = replace(
        mpc_config,
        prediction_horizon=study_config.prediction_horizon,
        terminal_goal_weight=study_config.terminal_goal_weight,
        terminal_slack_bound=study_config.terminal_slack_bound,
        terminal_slack_weight=study_config.terminal_slack_weight,
    )
    seed_iterations = run_distributed_harbor_lmpc(
        agents,
        simulation,
        communication,
        replace(
            study_mpc,
            learning_iterations=1,
            residual_adaptation=False,
            control_effectiveness_adaptation=False,
        ),
    )
    seed = min(
        (record for record in seed_iterations if record.admitted),
        key=lambda record: record.completion_step_sum,
    )
    cases = []
    generated = generate_temporary_fault_ensemble(
        agents, base_disturbance, ensemble_config
    )
    for case_seed, observation_seed, disturbance in generated:
        noise = replace(observation_noise, enabled=True, seed=observation_seed)
        evaluation = replace(
            simulation,
            guidance_update_interval_steps=study_mpc.replan_interval_steps,
            goal_hold_steps=disturbance.evaluation_hold_steps,
        )
        print(
            f"Temporary-fault case {case_seed}; observation seed {observation_seed}",
            flush=True,
        )
        trials = _run_temporary_fault_trials(
            agents,
            simulation,
            evaluation,
            communication,
            study_mpc,
            disturbance,
            seed.result,
            experiment_config,
            noise,
            controller_labels=controller_labels,
        )
        cases.append(
            HarborFaultEnsembleCase(
                seed=case_seed,
                disturbance=disturbance,
                observation_seed=observation_seed,
                trials=tuple(trials),
            )
        )
    return cases


def _run_temporary_fault_trials(
    agents,
    simulation,
    evaluation,
    communication,
    study_mpc,
    disturbance,
    seed_result,
    experiment_config,
    observation_noise,
    controller_labels: tuple[str, ...] | None = None,
) -> list[HarborRobustnessTrial]:
    """Run the matched passive and event-triggered temporary-fault policies."""
    available = {
        "Fixed-covariance RLS",
        "Innovation-threshold RLS",
        "Chi-square CUSUM RLS",
        "CUSUM-triggered probing RLS",
    }
    selected = available if controller_labels is None else set(controller_labels)
    if not selected or not selected <= available:
        raise ValueError("temporary-fault controller labels must be known and nonempty")
    study_mpc = replace(
        study_mpc,
        effectiveness_rls_change_warmup_steps=(
            experiment_config.change_warmup_steps
        ),
        effectiveness_rls_change_cooldown_steps=(
            experiment_config.change_cooldown_steps
        ),
    )
    trials = []
    for label, adaptive, detector in (
        ("Fixed-covariance RLS", False, "threshold"),
        ("Innovation-threshold RLS", True, "threshold"),
        ("Chi-square CUSUM RLS", True, "cusum"),
    ):
        if label not in selected:
            continue
        trials.extend(
            _run_fault_trials(
                agents,
                simulation,
                evaluation,
                communication,
                replace(
                    study_mpc,
                    effectiveness_rls_adaptive_covariance=adaptive,
                    effectiveness_rls_change_detector=detector,
                    effectiveness_rls_change_threshold=(
                        experiment_config.change_threshold
                    ),
                    effectiveness_rls_covariance_inflation=(
                        experiment_config.covariance_inflation
                    ),
                ),
                disturbance,
                seed_result,
                (
                    (
                        label,
                        True,
                        "recursive_diagonal",
                        False,
                        False,
                        None,
                        "energy",
                        1,
                    ),
                ),
                observation_noise=observation_noise,
            )
        )
    if "CUSUM-triggered probing RLS" in selected:
        trials.extend(
            _run_fault_trials(
                agents,
                simulation,
                evaluation,
                communication,
                replace(
                    study_mpc,
                    effectiveness_rls_adaptive_covariance=True,
                    effectiveness_rls_change_detector="cusum",
                    effectiveness_rls_change_threshold=(
                        experiment_config.change_threshold
                    ),
                    effectiveness_rls_covariance_inflation=(
                        experiment_config.covariance_inflation
                    ),
                    identification_reset_on_change=True,
                    identification_arm_on_change=True,
                    identification_arm_on_loss_only=True,
                ),
                disturbance,
                seed_result,
                (
                    (
                        "CUSUM-triggered probing RLS",
                        True,
                        "recursive_diagonal",
                        False,
                        True,
                        None,
                        "information",
                        1,
                    ),
                ),
                observation_noise=observation_noise,
            )
        )
    return trials


def _run_fault_trials(
    agents,
    simulation,
    evaluation,
    communication,
    study_mpc,
    disturbance,
    seed_result,
    definitions,
    observation_noise: HarborObservationNoiseConfig | None = None,
) -> list[HarborRobustnessTrial]:
    """Run matched fault definitions from one common safe trajectory."""
    trials = []
    for (
        label,
        adaptive,
        estimator_mode,
        learning,
        active,
        source_label,
        identification_strategy,
        minimum_probes,
    ) in definitions:
        source = (
            next(trial for trial in trials if trial.label == source_label)
            if source_label is not None
            else None
        )
        safe_result = source.result if source is not None else seed_result
        controller = DistributedHarborMPC(
            agents=agents,
            config=replace(
                study_mpc,
                residual_adaptation=False,
                control_effectiveness_adaptation=adaptive,
                effectiveness_estimator_mode=estimator_mode,
                active_identification=active,
                identification_strategy=identification_strategy,
                identification_min_probes_per_channel=minimum_probes,
            ),
            dt=simulation.dt,
            safe_states=safe_result.states,
            safe_controls=safe_result.controls,
            learning=learning,
            initial_effectiveness_estimates=(
                source.final_effectiveness_estimates
                if source is not None
                else None
            ),
        )
        result = run_harbor_simulation(
            agents,
            evaluation,
            communication,
            control_provider=controller,
            disturbance=disturbance,
            observation_noise=observation_noise,
        )
        completion_cost = sum(
            step if step is not None else simulation.horizon + 1
            for step in result.first_goal_steps.values()
        )
        valid = (
            result.all_goals_reached
            and result.pairwise_violation_count == 0
            and controller.fallback_count == 0
            and controller.max_collision_slack <= 1e-9
        )
        trials.append(
            HarborRobustnessTrial(
                label=label,
                identification_strategy=controller.config.identification_strategy,
                result=result,
                valid=valid,
                source_controller=source_label,
                completion_step_sum=completion_cost,
                solver_fallbacks=controller.fallback_count,
                solver_fallbacks_by_agent=controller.fallback_count_by_agent.copy(),
                solver_failure_steps_by_agent={
                    name: steps.copy()
                    for name, steps in controller.failure_steps_by_agent.items()
                },
                solver_failure_status_counts=controller.failure_status_counts.copy(),
                max_collision_slack=controller.max_collision_slack,
                residual_history={
                    name: np.asarray(values, dtype=float)
                    for name, values in controller.residual_history.items()
                },
                final_residual_estimates={
                    name: value.copy()
                    for name, value in controller.position_drift_estimates.items()
                },
                effectiveness_history={
                    name: np.asarray(values, dtype=float)
                    for name, values in controller.effectiveness_history.items()
                },
                final_effectiveness_estimates={
                    name: value.copy()
                    for name, value in controller.control_effectiveness_estimates.items()
                },
                effectiveness_change_steps_by_agent={
                    name: steps.copy()
                    for name, steps in controller.effectiveness_change_steps_by_agent.items()
                },
                excitation_history={
                    name: np.asarray(values, dtype=float)
                    for name, values in controller.excitation_history.items()
                },
                information_std_history={
                    name: np.asarray(values, dtype=float)
                    for name, values in controller.information_std_history.items()
                },
                probe_count_by_agent=dict(
                    controller.identification_probe_count_by_agent
                ),
                probe_channel_counts={
                    name: value.copy()
                    for name, value in controller.identification_probe_channel_counts.items()
                },
                probe_sequence_by_agent={
                    name: list(value)
                    for name, value in controller.identification_probe_sequence_by_agent.items()
                },
                probe_rejection_counts={
                    name: value.copy()
                    for name, value in controller.identification_probe_rejection_counts.items()
                },
            )
        )
        print(
            f"{label}: complete={result.all_goals_reached}, "
            f"safe={result.pairwise_violation_count == 0}, "
            f"cost={completion_cost}, fallbacks={controller.fallback_count}",
            flush=True,
        )
    return trials
