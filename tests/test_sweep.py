"""Tests for compact benchmark serialization and display."""

from __future__ import annotations

from contextlib import redirect_stdout
import csv
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.run_sweep import _print_table, _write_csv


class SweepTests(unittest.TestCase):
    def test_benchmark_outputs_runtime_and_latest_solver_status(self) -> None:
        record = {
            "scenario": "probe",
            "iterations": 1,
            "max_steps": 10,
            "elapsed_seconds": 12.3456,
            "selected_iteration": 0,
            "selected_valid": True,
            "selected_safe": True,
            "selected_solver_clean": True,
            "selected_min_pairwise": 1.2,
            "selected_min_obstacle_clearance": 0.3,
            "selected_fallback_count": 0,
            "selected_safety_interventions": 0,
            "selected_max_static_slack": None,
            "selected_max_hyperplane_slack": None,
            "selected_nonzero_static_slack_steps": 0,
            "selected_nonzero_hyperplane_slack_steps": 0,
            "latest_max_static_slack": 0.0,
            "latest_max_hyperplane_slack": 0.0,
            "latest_nonzero_static_slack_steps": 0,
            "latest_nonzero_hyperplane_slack_steps": 0,
            "latest_valid": False,
            "latest_safe": True,
            "latest_solver_clean": False,
            "latest_fallback_count": 1,
            "latest_safety_interventions": 0,
            "selected_costs": {"0": 10},
        }

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "summary.csv"
            _write_csv(path, [record])
            with path.open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))

        self.assertEqual(row["elapsed_seconds"], "12.3456")
        self.assertEqual(row["latest_solver_clean"], "False")

        output = StringIO()
        with redirect_stdout(output):
            _print_table([record])
        rendered = output.getvalue()
        self.assertIn("seconds", rendered)
        self.assertIn("latest_clean", rendered)
        self.assertIn("probe, 12.346, 0, True, True, True, False", rendered)


if __name__ == "__main__":
    unittest.main()
