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
    elif command == "harbor-lmpc":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        from scripts.run_harbor_lmpc import main as run_harbor_lmpc

        run_harbor_lmpc()
    elif command == "harbor-horizon-study":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        from scripts.run_harbor_horizon_study import main as run_horizon_study

        run_horizon_study()
    elif command == "harbor-robustness":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        from scripts.run_harbor_robustness import main as run_harbor_robustness

        run_harbor_robustness()
    elif command == "harbor-fault-study":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        from scripts.run_harbor_fault_study import main as run_fault_study

        run_fault_study()
    elif command == "harbor-fault-generalization":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        from scripts.run_harbor_fault_generalization import main as run_generalization

        run_generalization()
    elif command == "harbor-fault-noise-study":
        sys.argv = [sys.argv[0], *sys.argv[2:], "--with-observation-noise"]
        from scripts.run_harbor_fault_generalization import main as run_noise_study

        run_noise_study()
    elif command == "harbor-prediction-study":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        from scripts.run_harbor_prediction_study import main as run_prediction_study

        run_prediction_study()
    elif command == "harbor-time-varying-fault-study":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        from scripts.run_harbor_time_varying_fault_study import main as run_fault_tracking

        run_fault_tracking()
    elif command == "harbor-temporary-fault-generalization":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        from scripts.run_harbor_temporary_fault_generalization import (
            main as run_temporary_fault_generalization,
        )

        run_temporary_fault_generalization()
    elif command == "harbor-joint-uncertainty-study":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        from scripts.run_harbor_joint_uncertainty_study import (
            main as run_joint_uncertainty,
        )

        run_joint_uncertainty()
    elif command == "harbor-projected-residual-study":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        from scripts.run_harbor_projected_residual_study import (
            main as run_projected_residual,
        )

        run_projected_residual()
    elif command == "harbor-dynamic-envelope-study":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        from scripts.run_harbor_dynamic_envelope_study import (
            main as run_dynamic_envelope,
        )

        run_dynamic_envelope()
    elif command == "harbor-station-keeping-study":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        from scripts.run_harbor_station_keeping_study import (
            main as run_station_keeping,
        )

        run_station_keeping()
    elif command == "harbor-localization-study":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        from scripts.run_harbor_localization_study import main as run_localization

        run_localization()
    else:
        from scripts.run_manta_lmpc import main as run_manta

        run_manta()


if __name__ == "__main__":
    main()
