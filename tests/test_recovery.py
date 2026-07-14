"""Tests for bounded fallback and terminal recovery behavior."""

from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from scripts.dynamics import MantaDynamicsConfig
from scripts.learning import APFConfig
from scripts.learning.recovery import (
    _fallback_control_candidates,
    repair_incomplete_with_apf,
    safe_fallback_apf_step,
)
from scripts.learning.runner import _controls_by_agent
from scripts.mpc import MantaLMPCConfig
from scripts.simulation import Scenario, StaticObstacle


class RecoveryTests(unittest.TestCase):
    def test_control_histories_preserve_recovery_steps(self) -> None:
        controls = np.arange(12, dtype=float).reshape(3, 2, 2)

        by_agent = _controls_by_agent(controls, num_agents=2)

        np.testing.assert_array_equal(by_agent[0], controls[:, 0, :])
        np.testing.assert_array_equal(by_agent[1], controls[:, 1, :])
        with self.assertRaisesRegex(ValueError, "controls must have shape"):
            _controls_by_agent(np.zeros((3, 2)), num_agents=2)

    def test_fallback_candidates_preserve_original_unique_grid(self) -> None:
        dynamics = MantaDynamicsConfig()
        candidates = _fallback_control_candidates(np.array([1.2, 1.3]), dynamics)

        self.assertEqual(len(candidates), 10)
        self.assertEqual(len({tuple(control) for control in candidates}), 10)
        self.assertTrue(
            all(
                np.all(control >= dynamics.mu_min)
                and np.all(control <= dynamics.mu_max)
                for control in candidates
            )
        )

    def test_safe_fallback_returns_bounded_dynamic_step(self) -> None:
        dynamics = MantaDynamicsConfig()
        state = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.1, 0.0])
        control, next_state = safe_fallback_apf_step(
            current_state=state,
            goal_state=np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            obstacle=StaticObstacle(center=(10.0, 10.0), radius=0.5),
            apf_config=APFConfig(),
            dt=0.2,
            dynamics_config=dynamics,
        )

        self.assertEqual(control.shape, (2,))
        self.assertEqual(next_state.shape, (7,))
        self.assertTrue(np.all(np.isfinite(next_state)))
        self.assertTrue(np.all(control >= dynamics.mu_min))
        self.assertTrue(np.all(control <= dynamics.mu_max))

    def test_repair_is_noop_when_all_agents_already_reached(self) -> None:
        starts = np.array(
            [
                [0.0, 0.0, 0.0, 0.1, 0.0, 0.1, 0.0],
                [2.0, 0.0, 0.0, 0.1, 0.0, 0.1, 0.0],
            ]
        )
        scenario = Scenario(
            name="complete",
            starts=starts,
            goals=starts.copy(),
            safety_distance=0.5,
            obstacle=StaticObstacle(center=(10.0, 10.0), radius=0.5),
        )
        history = {agent: [state.copy()] for agent, state in enumerate(starts)}
        current = {agent: state.copy() for agent, state in enumerate(starts)}
        safe_sets = {agent: np.array([state]) for agent, state in enumerate(starts)}
        result = repair_incomplete_with_apf(
            history=history,
            current_states=current,
            safe_sets=safe_sets,
            reached_agents={0, 1},
            goals=starts.copy(),
            scenario=scenario,
            config=replace(MantaLMPCConfig(), repair_max_steps=2),
            apf_config=APFConfig(),
            dynamics_config=MantaDynamicsConfig(),
            should_stop=None,
            verbose=False,
        )

        self.assertEqual(result.controls.shape, (0, 2, 2))
        self.assertEqual(result.statuses, [])
        self.assertTrue(all(len(agent_history) == 1 for agent_history in history.values()))

    def test_repair_appends_one_staged_step_at_configured_cap(self) -> None:
        starts = np.array(
            [
                [0.0, 0.0, 0.0, 0.1, 0.0, 0.1, 0.0],
                [0.0, 2.0, 0.0, 0.1, 0.0, 0.1, 0.0],
            ]
        )
        goals = starts.copy()
        goals[0, 0] = 4.0
        scenario = Scenario(
            name="repair",
            starts=starts,
            goals=goals,
            safety_distance=0.5,
            obstacle=StaticObstacle(center=(10.0, 10.0), radius=0.5),
        )
        history = {agent: [state.copy()] for agent, state in enumerate(starts)}
        current = {agent: state.copy() for agent, state in enumerate(starts)}
        safe_sets = {
            0: np.vstack((starts[0], goals[0])),
            1: np.array([starts[1]]),
        }
        result = repair_incomplete_with_apf(
            history=history,
            current_states=current,
            safe_sets=safe_sets,
            reached_agents={1},
            goals=goals,
            scenario=scenario,
            config=replace(
                MantaLMPCConfig(), repair_max_steps=1, repair_waypoint_lookahead=1
            ),
            apf_config=APFConfig(),
            dynamics_config=MantaDynamicsConfig(),
            should_stop=None,
            verbose=False,
        )

        self.assertEqual(result.controls.shape, (1, 2, 2))
        self.assertEqual(result.statuses, [{0: "repair_apf", 1: "repair_hold"}])
        self.assertTrue(all(len(agent_history) == 2 for agent_history in history.values()))


if __name__ == "__main__":
    unittest.main()
