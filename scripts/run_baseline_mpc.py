"""CLI shortcut for the iteration-0 APF manta baseline."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.run_manta_lmpc import main as run_manta_lmpc_main


def main() -> None:
    """Run the manta workflow with LMPC iterations forced to zero."""
    sys.argv = [sys.argv[0], *sys.argv[1:], "--iterations", "0"]
    run_manta_lmpc_main()


if __name__ == "__main__":
    main()
