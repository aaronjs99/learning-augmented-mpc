# Changelog

All notable repository-level changes are tracked here.

## [Unreleased]
### Added
- Swept-safe compact APF staging with dynamically replayed delayed controls.
- Modular untethered UGV/USV/ROV harbor simulation with configurable delayed
  communication and a reproducible independent-versus-coordinated ablation.
- MIT project license.
- Baseline decentralized MPC controller using CVXPY/OSQP.
- Closed-loop baseline MPC runner with metrics, CSV trajectory/control logs, solver statuses, plots, and optional GIF animations.
- Baseline MPC experiment record and dependency list.
- Baseline solver diagnostics with per-timestep, per-agent statuses.

### Changed
- Triangle APF initialization now compares compact concurrent schedules against
  the original sequential fallback using the shared admission validator.
- The root CLI now supports `python run.py harbor` without writing artifacts by
  default.
- Plotting now uses a noninteractive Matplotlib backend and local ignored cache for reproducible script runs.
- Generated baseline and sanity result folders are ignored by default.
- `crossing_paths` baseline uses soft linearized collision penalties because the hard formulation is infeasible from its straight-line reference.
- Project docs now describe the baseline MPC command, outputs, and resolved hard linearized collision-constraint assumption.

## [2026-04-02]
### Added
- Research documentation hub and planning scaffolds
- Contribution guide and GitHub issue/PR templates
- Repository hygiene files (`.editorconfig`, improved `.gitignore`, license/security placeholders)
