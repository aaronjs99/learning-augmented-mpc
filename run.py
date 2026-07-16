"""Root command dispatcher for manta LMPC and harbor experiments."""

from __future__ import annotations

import sys


def main() -> None:
    """Dispatch manta runs, benchmarks, sanity checks, and tests."""
    command = sys.argv[1] if len(sys.argv) > 1 else "manta"
    if command in {"manta", "lmpc"}:
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        from scripts.run_manta_lmpc import main as run_manta

        run_manta()
    elif command == "baseline":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        from scripts.run_baseline_mpc import main as run_baseline

        run_baseline()
    elif command == "sanity":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        from scripts.run_sanity_checks import main as run_sanity

        run_sanity()
    elif command == "sweep":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        from scripts.run_sweep import main as run_sweep

        run_sweep()
    elif command == "test":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        from scripts.run_tests import main as run_tests

        run_tests()
    elif command == "harbor":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        from scripts.run_harbor import main as run_harbor

        run_harbor()
    elif command == "harbor-sweep":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        from scripts.run_harbor_sweep import main as run_harbor_sweep

        run_harbor_sweep()
    else:
        from scripts.run_manta_lmpc import main as run_manta

        run_manta()


if __name__ == "__main__":
    main()
