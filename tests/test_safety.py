"""Tests for the execution-time swept transition safety filter."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.config import load_project_config
from scripts.learning.safety import (
    filter_unsafe_transitions,
    unsafe_transition_agents,
)


class SafetyFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.project = load_project_config(scenario_name="manta_crossover")

    def test_detects_synchronous_crossing_between_safe_endpoints(self) -> None:
        current = {
            0: np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            1: np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        }
        proposed = {
            0: np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            1: np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        }

        unsafe = unsafe_transition_agents(
            current, proposed, scenario=self.project.scenario
        )

        self.assertEqual(unsafe, {0, 1})

    def test_filter_replaces_obstacle_crossing_with_safe_action(self) -> None:
        scenario = self.project.scenario
        current = {
            0: self.project.scenario.starts[0].copy(),
            1: self.project.scenario.starts[1].copy(),
        }
        proposed = {agent: state.copy() for agent, state in current.items()}
        proposed[0][:2] = scenario.obstacle.center
        controls = {agent: np.ones(2) for agent in current}

        result = filter_unsafe_transitions(
            current_states=current,
            proposed_controls=controls,
            proposed_next_states=proposed,
            goals=self.project.scenario.goals,
            scenario=scenario,
            config=self.project.lmpc,
            apf_config=self.project.apf,
            dynamics_config=self.project.dynamics,
        )

        self.assertIn(0, result.statuses)
        self.assertFalse(
            unsafe_transition_agents(
                current, result.next_states, scenario=scenario
            )
        )

    def test_filter_holds_protected_goal_agent(self) -> None:
        current = {
            0: self.project.scenario.goals[0].copy(),
            1: self.project.scenario.goals[0].copy(),
        }
        current[1][0] += 1.0
        proposed = {agent: state.copy() for agent, state in current.items()}
        proposed[1][:2] = current[0][:2]

        result = filter_unsafe_transitions(
            current_states=current,
            proposed_controls={0: np.zeros(2), 1: np.ones(2)},
            proposed_next_states=proposed,
            goals=self.project.scenario.goals,
            scenario=self.project.scenario,
            config=self.project.lmpc,
            apf_config=self.project.apf,
            dynamics_config=self.project.dynamics,
            protected_agents={0},
        )

        self.assertEqual(result.statuses[0], "safety_filter_hold")
        np.testing.assert_allclose(result.next_states[0][:2], current[0][:2])


if __name__ == "__main__":
    unittest.main()
