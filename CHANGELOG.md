# Changelog

All notable repository-level changes are tracked here.

## [Unreleased]
### Added
- Swept-safe compact APF staging with dynamically replayed delayed controls.
- Modular untethered UGV/USV/ROV harbor simulation with configurable delayed
  communication and a reproducible independent-versus-coordinated ablation.
- ETA-negotiated asymmetric harbor coordination with a reciprocal-policy
  baseline and completion-cost regression test.
- Seeded network robustness sweeps with separate safety/final-completion
  heatmaps and optional delayed-message prediction.
- Explicit shoreline, quay, surface-water, and underwater operating regions for
  physically meaningful UGV/USV/ROV harbor scenarios.
- Two independent quay UGVs and a depth-changing, 12-state 6-DOF ROV route.
- Separate 3-DOF planar and 6-DOF underwater pose-goal validation.
- Harbor safety envelopes, successful-rollout samples, goal headings, depth,
  and roll/pitch/yaw diagnostics.
- Per-agent distributed harbor MPC and LMPC with hard communicated collision
  constraints, safe-trajectory terminal hulls, time-to-go costs, strict clean
  rollout admission, and curated progress PNG/GIF telemetry.
- Configurable dynamic skid-steer UGV, 3-DOF marine USV, and 6-DOF marine ROV
  dynamics with matching NumPy/CasADi transitions and a frozen reduced baseline.
- Named platform profiles for SRI Lab's Jackal-based RobEn and Husky-based
  Inspector-Gadget, a full-payload Clearpath Heron, and BlueROV2 Heavy, with
  separate hardware bounds and inspection mission speeds.
- Matched-horizon MPC/LMPC study and horizon-efficiency plot demonstrating
  learned-terminal-set liveness and completion-cost gains.
- Hidden execution-plant current/actuator mismatch, separated local causal
  effectiveness and residual adaptation, sustained-goal evaluation, and
  nominal-versus-adaptive robustness diagnostics.
- MIT project license.
- Baseline decentralized MPC controller using CVXPY/OSQP.
- Closed-loop baseline MPC runner with metrics, CSV trajectory/control logs, solver statuses, plots, and optional GIF animations.
- Baseline MPC experiment record and dependency list.
- Baseline solver diagnostics with per-timestep, per-agent statuses.

### Changed
- External MPC providers now continue station-keeping updates after first goal
  entry, preventing stale nonzero UGV controls while slower agents finish.
- Physical harbor models are now the default; all masses, damping, hydrostatics,
  platform profiles, and axis-specific actuator limits live in
  `config/harbor.yaml`.
- Harbor learning seeds from the best clean MPC rollout and rejects later safe
  iterations that regress completion cost.
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
