"""CLI entry point for config-driven manta APF/LMPC runs."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.config import DEFAULT_CONFIG_PATH, ProjectConfig, load_project_config
from scripts.learning import run_manta_lmpc
from scripts.reporting import save_manta_run_report


def parse_args() -> argparse.Namespace:
    """Parse CLI overrides for the YAML-backed manta LMPC run."""
    parser = argparse.ArgumentParser(description="Run config-driven manta APF/LMPC.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--apf-max-steps", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--mpc-horizon", type=int, default=None)
    parser.add_argument("--k-hull", type=int, default=None)
    parser.add_argument("--goal-tolerance", type=float, default=None)
    parser.add_argument("--make-video", action="store_true", default=None)
    parser.add_argument("--no-video", action="store_false", dest="make_video")
    parser.add_argument("--quiet", action="store_true", default=None)
    parser.add_argument("--verbose", action="store_false", dest="quiet")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="override output directory for this run",
    )
    parser.add_argument(
        "--stop-file",
        default="STOP_RUN",
        help="file path that stops the run when created",
    )
    return parser.parse_args()


def main() -> None:
    """Run APF/LMPC from YAML config and save all diagnostics."""
    args = parse_args()
    project_config = _load_effective_config(args)
    scenario = project_config.scenario
    lmpc_config = project_config.lmpc
    apf_config = project_config.apf

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = (
        Path(args.output_dir)
        if args.output_dir
        else project_config.output.root_dir
        / f"{project_config.output.run_prefix}_{timestamp}"
    )
    root.mkdir(parents=True, exist_ok=True)
    stop_path = _resolve_stop_file(args.stop_file)
    if stop_path.exists():
        stop_path.unlink()

    try:
        result = run_manta_lmpc(
            scenario,
            config=lmpc_config,
            apf_config=apf_config,
            dynamics_config=project_config.dynamics,
            should_stop=stop_path.exists,
            verbose=not project_config.quiet,
        )
    except KeyboardInterrupt:
        if stop_path.exists():
            stop_path.unlink()
        print("Stopped manta LMPC run.")
        raise SystemExit(130) from None

    save_manta_run_report(
        root,
        result,
        project_config,
        config_path=args.config,
    )
    print(f"Saved manta LMPC outputs to: {root}")


def _load_effective_config(args: argparse.Namespace) -> ProjectConfig:
    """Load YAML config, then apply explicit CLI overrides."""
    config = load_project_config(args.config, scenario_name=args.scenario)
    lmpc_updates = {
        "iterations": args.iterations,
        "max_steps": args.max_steps,
        "dt": args.dt,
        "prediction_horizon": args.mpc_horizon,
        "k_hull": args.k_hull,
        "goal_tolerance": args.goal_tolerance,
    }
    lmpc_updates = {
        key: value for key, value in lmpc_updates.items() if value is not None
    }
    apf_updates = {"max_steps": args.apf_max_steps}
    apf_updates = {
        key: value for key, value in apf_updates.items() if value is not None
    }

    make_video = config.make_video if args.make_video is None else args.make_video
    quiet = config.quiet if args.quiet is None else args.quiet

    return replace(
        config,
        lmpc=replace(config.lmpc, **lmpc_updates),
        apf=replace(config.apf, **apf_updates),
        make_video=make_video,
        quiet=quiet,
    )


def _resolve_stop_file(path: str | Path) -> Path:
    stop_path = Path(path)
    if not stop_path.is_absolute():
        stop_path = Path.cwd() / stop_path
    return stop_path


if __name__ == "__main__":
    main()
