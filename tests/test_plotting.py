"""Visual contract tests for shared primitives and the active manta GIF."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.config import load_project_config
from scripts.plotting import save_manta_animation
from scripts.plotting._backend import configure_matplotlib

configure_matplotlib()
from matplotlib import pyplot as plt

from scripts.plotting.primitives import add_obstacle_layers, agent_color
from scripts.plotting.diagnostics import compute_diagnostics


class PlottingTests(unittest.TestCase):
    def test_obstacle_layers_preserve_physical_constraint_and_padding_radii(self) -> None:
        obstacle = load_project_config(
            scenario_name="manta_triangle"
        ).scenario.obstacle
        fig, ax = plt.subplots()
        try:
            add_obstacle_layers(ax, obstacle, obstacle_padding=0.2)
            self.assertEqual(len(ax.patches), 3)
            self.assertEqual(
                [patch.radius for patch in ax.patches],
                [obstacle.radius + 0.2, obstacle.radius, obstacle.physical_radius],
            )
        finally:
            plt.close(fig)

    def test_agent_colors_cycle_and_reject_negative_indices(self) -> None:
        self.assertEqual(agent_color(0), agent_color(6))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            agent_color(-1)

    def test_diagnostics_detect_between_sample_crossing(self) -> None:
        diagnostics = compute_diagnostics(
            {
                0: np.array([[-1.0, 0.0], [1.0, 0.0]]),
                1: np.array([[1.0, 0.0], [-1.0, 0.0]]),
            },
            obstacle=None,
            safety_distance=0.5,
        )

        self.assertAlmostEqual(diagnostics.min_pairwise_distance, 0.0)
        self.assertEqual(diagnostics.pairwise_violation_count, 1)

    def test_manta_animation_writes_nonblank_multiframe_gif(self) -> None:
        project = load_project_config(scenario_name="manta_crossover")
        histories = {
            agent: np.vstack(
                (
                    project.scenario.starts[agent],
                    0.5
                    * (
                        project.scenario.starts[agent]
                        + project.scenario.goals[agent]
                    ),
                    project.scenario.goals[agent],
                )
            )
            for agent in range(len(project.scenario.starts))
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manta.gif"
            save_manta_animation(
                histories,
                project.scenario.goals,
                project.scenario.obstacle,
                dt=project.lmpc.dt,
                out_path=str(path),
                fps=2,
                goal_tolerance=project.lmpc.goal_tolerance,
                obstacle_padding=project.apf.obstacle_padding,
                safety_distance=project.scenario.safety_distance,
            )

            with Image.open(path) as image:
                self.assertGreaterEqual(image.n_frames, 2)
                self.assertGreater(image.width, 100)
                self.assertGreater(image.height, 100)
                image.seek(0)
                extrema = image.convert("RGB").getextrema()
                self.assertTrue(any(low < high for low, high in extrema))


if __name__ == "__main__":
    unittest.main()
