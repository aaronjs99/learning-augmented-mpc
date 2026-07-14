"""Run the project's dependency-light regression test suite."""

from __future__ import annotations

from pathlib import Path
import unittest


def main() -> None:
    """Discover tests from the project root and return a shell-friendly status."""
    project_root = Path(__file__).resolve().parents[1]
    suite = unittest.defaultTestLoader.discover(str(project_root / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
