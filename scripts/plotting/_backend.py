"""Matplotlib backend setup shared by plotting utilities."""

from __future__ import annotations

import os
from pathlib import Path


def configure_matplotlib() -> None:
    """Use a noninteractive backend and local cache for reproducible scripts."""
    if "MPLCONFIGDIR" not in os.environ:
        cache_dir = Path("results") / "tmp" / "matplotlib_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache_dir)

    import matplotlib

    matplotlib.use("Agg")
