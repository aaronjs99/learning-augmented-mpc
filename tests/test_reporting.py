"""Tests for the stable manta report and artifact contract."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.config import load_project_config
from scripts.learning import MantaLMPCRunResult
from scripts.metrics import validate_trajectory
from scripts.reporting import prepare_manta_report, save_manta_run_report


class ReportingTests(unittest.TestCase):
    def setUp(self) -> None:
        loaded = load_project_config(scenario_name="manta_crossover")
        self.project = replace(
            loaded,
            lmpc=replace(loaded.lmpc, iterations=1),
            make_video=False,
            quiet=True,
        )
        scenario = self.project.scenario
        history = {
            agent: np.vstack((scenario.starts[agent], scenario.goals[agent]))
            for agent in range(len(scenario.starts))
        }
        validation = validate_trajectory(
            history,
            scenario.goals,
            scenario.safety_distance,
            scenario.obstacle.center,
            scenario.obstacle.radius,
            self.project.lmpc.goal_tolerance,
            statuses=[{0: "ok", 1: "fallback_apf"}],
        )
        self.result = MantaLMPCRunResult(
            scenario_name=scenario.name,
            histories=[history, history],
            controls_by_iteration=[np.zeros((1, 2, 2))],
            slack_by_iteration=[
                np.array([[[0.01, 0.0, 0.3], [0.0, 0.02, 0.4]]], dtype=float)
            ],
            statuses_by_iteration=[[{0: "ok", 1: "fallback_apf"}]],
            success_by_iteration=[True, validation.valid],
            goal_reached_by_iteration=[True, validation.all_goals_reached],
            learned_by_iteration=[True, validation.usable_for_learning],
            validation_by_iteration=[validation, validation],
            selected_iteration=1,
        )

    def test_prepare_report_preserves_summary_contract(self) -> None:
        report = prepare_manta_report(
            self.result,
            self.project,
            config_path="config/manta.yaml",
        )

        self.assertEqual(report.summary["scenario"], "manta_crossover")
        self.assertEqual(report.summary["selected_iteration"], 1)
        self.assertEqual(
            report.summary["optimizer_relaxations"]["terminal_slack_weight"],
            10000.0,
        )
        self.assertEqual(
            report.summary["status_counts_by_iteration"],
            [{}, {"fallback_apf": 1, "ok": 1}],
        )
        self.assertEqual(
            report.summary["optimizer_slack_by_iteration"][1],
            {
                "solved_agent_steps": 2,
                "max_static_slack": 0.01,
                "nonzero_static_slack_steps": 1,
                "max_hyperplane_slack": 0.02,
                "nonzero_hyperplane_slack_steps": 1,
                "max_terminal_slack": 0.4,
                "nonzero_terminal_slack_steps": 2,
            },
        )
        per_agent = report.summary["optimizer_slack_by_agent_by_iteration"][1]
        self.assertEqual(per_agent["0"]["max_terminal_slack"], 0.3)
        self.assertEqual(per_agent["1"]["max_terminal_slack"], 0.4)
        self.assertEqual(per_agent["0"]["solved_agent_steps"], 1)
        self.assertEqual(report.summary["cost_by_iteration"], {"0": [1, 1], "1": [1, 1]})
        self.assertEqual(report.final_states.shape, (2, 2, 7))
        self.assertEqual(report.pairwise_distances.shape, (2, 1))

    def test_save_report_writes_expected_non_video_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            save_manta_run_report(
                directory,
                self.result,
                self.project,
                config_path="config/manta.yaml",
            )
            output = Path(directory)
            expected = {
                "summary.json",
                "states_by_iteration.csv",
                "learning_progression.png",
                "cost_decrease.png",
                "final_trajectories.png",
                "pairwise_distances.png",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            for filename in expected:
                if not filename.endswith(".png"):
                    continue
                with Image.open(output / filename) as image:
                    self.assertGreater(image.width, 100)
                    self.assertGreater(image.height, 100)
                    extrema = image.convert("RGB").getextrema()
                    self.assertTrue(any(low < high for low, high in extrema))
            with (output / "summary.json").open(encoding="utf-8") as file:
                self.assertEqual(json.load(file)["selected_iteration"], 1)


if __name__ == "__main__":
    unittest.main()
