"""Tests for YAML loading, overrides, and scenario boundary validation."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.config import (
    list_config_scenarios,
    load_project_config,
    override_project_config,
)
from scripts.simulation import Scenario, StaticObstacle


class ConfigTests(unittest.TestCase):
    def test_every_yaml_scenario_has_valid_manta_shapes(self) -> None:
        names = list_config_scenarios()

        self.assertEqual(
            names,
            [
                "lane_swap",
                "manta_crossover",
                "manta_triangle",
                "narrow_gate",
                "offset_crossing",
            ],
        )
        for name in names:
            project = load_project_config(scenario_name=name)
            self.assertEqual(project.scenario.starts.shape[1], 7)
            self.assertEqual(project.scenario.goals.shape, project.scenario.starts.shape)

    def test_override_helper_filters_none_and_preserves_false_values(self) -> None:
        project = load_project_config(scenario_name="manta_crossover")
        overridden = override_project_config(
            project,
            lmpc={"iterations": 0, "max_steps": None},
            apf={"max_steps": 12},
            make_video=False,
            quiet=True,
        )

        self.assertEqual(overridden.lmpc.iterations, 0)
        self.assertEqual(overridden.lmpc.max_steps, project.lmpc.max_steps)
        self.assertEqual(overridden.apf.max_steps, 12)
        self.assertFalse(overridden.make_video)
        self.assertTrue(overridden.quiet)

    def test_obstacle_rejects_physical_radius_larger_than_constraint(self) -> None:
        with self.assertRaisesRegex(ValueError, "physical_radius"):
            StaticObstacle(center=(0.0, 0.0), radius=0.5, physical_radius=0.6)

    def test_scenario_rejects_mismatched_state_arrays(self) -> None:
        with self.assertRaisesRegex(ValueError, "goals must match"):
            Scenario(
                name="invalid",
                starts=np.zeros((2, 7)),
                goals=np.zeros((3, 7)),
                safety_distance=0.5,
                obstacle=StaticObstacle(center=(0.0, 0.0), radius=0.5),
            )


if __name__ == "__main__":
    unittest.main()
