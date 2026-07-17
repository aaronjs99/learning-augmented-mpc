# MAE 271D: Decentralized Learning MPC

Config-driven multi-agent manta LMPC implementation plus a modular
heterogeneous harbor research testbed. The manta state is:

`[x, y, theta, p_L, q_L, p_R, q_R]`

The default run is controlled by `config/manta.yaml`.

The harbor testbed supplies guidance, distributed MPC, and distributed LMPC
controllers over dynamic skid-steer, 3-DOF surface-marine, and 6-DOF underwater-
marine dynamics. Named profiles distinguish RobEn/Jackal, Inspector-Gadget/
Husky, full-payload Heron, and BlueROV2 Heavy. UGV/USV goals are 3-DOF planar
poses and ROV goals are 6-DOF poses. See
`docs/harbor_dynamics.md` for the exact equations and fidelity boundary.
The complete estimator, optimization, safe-set, observability, fault-adaptation,
and belief-retry equations are collected in `docs/harbor_control_math.md`.

## Workflow
1. Generate iteration-0 safe trajectories with a staged APF autopilot.
2. Run decentralized CasADi/IPOPT LMPC with learned safe-set terminal hulls and time-to-go costs.
3. Warm-start IPOPT from a blend of nominal controls and stored safe-set controls.
4. Use priority-aware SVM spatial hyperplanes plus soft slacks for inter-agent avoidance and a softened circular static-obstacle constraint.
5. Add only complete, collision-free rollouts back into the learned safe set.
6. Select the latest APF or LMPC iteration that reaches all goals without pairwise or obstacle violations.

## Layout
- `run.py`: root command dispatcher.
- `config/`: YAML runtime configuration.
- `docs/`: research contribution notes and experiment roadmap.
- `scripts/dynamics/`: manta dynamics and RK4 integration.
- `scripts/simulation/`: manta environment and scenario dataclasses.
- `scripts/harbor/`: independent UGV/USV/ROV dynamics, domains, communication,
  and heterogeneous simulation.
- `scripts/learning/`: APF initialization, safe-set sampling, SVM hyperplanes, runner.
- `scripts/mpc/`: CasADi LMPC agent builder.
- `scripts/metrics/`, `scripts/plotting/`: diagnostics, plots, and GIFs.
- `scripts/reporting/`: stable summary, CSV, plot, and animation artifact generation.
- `tests/`: fast regression tests for evaluation, safe-set admission, priority, and warm starts.
- `results/`: generated outputs.

## Setup
`python3 -m pip install -r requirements.txt`

## Run
Full manta LMPC using `config/manta.yaml`:

`python3 run.py`

APF baseline only, equivalent to iteration 0:

`python3 run.py baseline`

Zero-control manta sanity check:

`python3 run.py sanity`

Compact benchmark sweep, APF-only by default:

`python3 run.py sweep`

Regression suite, including one clean physical harbor MPC/LMPC solve:

`python3 run.py test`

Heterogeneous harbor communication ablation (writes no file by default):

`python3 run.py harbor`

Harbor comparison PNG and coordinated GIF under ignored temporary results:

`python3 run.py harbor --plot-dir results/tmp/harbor_eta`

Seeded communication delay/dropout sweep and robustness heatmap:

`python3 run.py harbor-sweep`

Distributed harbor MPC/LMPC with a rolling research dashboard and GIF:

`python3 run.py harbor-lmpc`

Matched-horizon MPC/LMPC efficiency study:

`python3 run.py harbor-horizon-study`

Combined current/actuator-mismatch study with nominal, residual-only, and
joint-adaptive distributed MPC/LMPC:

`python3 run.py harbor-robustness`

Asymmetric per-platform, per-control-channel fault study comparing scalar and
diagonal local adaptation:

`python3 run.py harbor-fault-study`

Five-case stratified actuator-fault generalization with paired equal-budget
probe scheduling:

`python3 run.py harbor-fault-generalization`

The same hidden-fault ensemble with seeded platform-specific onboard sensor
noise and robust recursive actuator identification:

`python3 run.py harbor-fault-noise-study`

Matched constant-velocity versus goal-bounded peer-motion prediction under the
same noisy hidden-fault ensemble:

`python3 run.py harbor-prediction-study`

Matched fixed-covariance, innovation-threshold, chi-square CUSUM, and
CUSUM-triggered probing under temporary, initially hidden actuator losses:

`python3 run.py harbor-time-varying-fault-study`

Stratified temporary-fault development and separately seeded holdout studies:

`python3 run.py harbor-temporary-fault-generalization`

`python3 run.py harbor-temporary-fault-generalization --holdout`

Frozen fixed-versus-threshold confirmation after candidate selection:

`python3 run.py harbor-temporary-fault-generalization --confirmation`

Recovery-prior development and separately frozen confirmation:

`python3 run.py harbor-temporary-fault-generalization --recovery-development`

`python3 run.py harbor-temporary-fault-generalization --recovery-confirmation`

Rank-gated transient-recovery development and frozen confirmation:

`python3 run.py harbor-temporary-fault-generalization --transient-recovery-development`

`python3 run.py harbor-temporary-fault-generalization --transient-recovery-confirmation`

Joint hidden-current/temporary-fault development and frozen confirmation:

`python3 run.py harbor-joint-uncertainty-study`

`python3 run.py harbor-joint-uncertainty-study --confirmation`

Controllability-projected residual and dynamic-envelope ablations:

`python3 run.py harbor-projected-residual-study`

`python3 run.py harbor-dynamic-envelope-study`

Actuator-independent marine-current station-keeping development:

`python3 run.py harbor-station-keeping-study`

GPS-denied dead reckoning, range localization, joint range SLAM, and dropout:

`python3 run.py harbor-localization-study`

Estimated-state distributed MPC under simultaneous localization error, hidden
current, observation noise, and temporary actuator loss:

`python3 run.py harbor-joint-localization-study`

These commands overwrite curated artifacts in `results/latest/harbor/`. The
robustness command writes one metrics JSON, one combined diagnostic, and one
GIF. Plant parameters are hidden from the controllers. Joint adaptation first
fits scalar control effectiveness from local velocity/rate response, then fits
the remaining position drift. Add `--no-gif` for a faster metrics-and-PNG
iteration.

The fault-study command keeps RobEn and Inspector-Gadget as distinct UGVs,
estimates their physical left/right drive losses separately, and compares
passive, round-robin, equal-budget one-pass, and information-aware active
identification. Repeated LMPC trials retain only their own preceding local
model estimate and clean rollout. The generalization command samples every
physical actuator channel across the configured effectiveness range and writes
aggregate paired statistics without generating redundant animations.
The noisy study adds an instantaneous-versus-recursive estimator ablation and
records solver recovery by platform and IPOPT status.
The prediction study isolates the communicated obstacle model from estimation,
safe memory, plant faults, and observation seeds.
The time-varying-fault study applies abrupt per-channel loss and recovery during
execution, records plant truth only for offline scoring, and compares matched
local estimators without giving any controller the change times or magnitudes.
The temporary-fault ensemble commands additionally vary every actuator loss,
per-agent onset, duration, and observation seed. The holdout seeds are separate
from development and must not be used for controller tuning. The confirmation
command uses a third seed set, only the selected threshold-RLS and fixed
comparator, and YAML-predeclared closed-loop pass/fail gates.

The recovery-prior commands evaluate an aggregate-direction-gated,
channel-selective nominal prior. Its independent confirmation did not pass every
gate, so ordinary innovation-threshold RLS remains the confirmed default.
The transient comparator keeps raw RLS untouched and applies a decaying,
rank-gated controller offset for at most one armed loss/recovery episode per
agent. Its confirmation result and every predeclared gate are stored in the
matching JSON artifact. The dwell-gated transient comparator passed its
independent simulation confirmation and is the confirmed recovery
accommodation layered over the ordinary threshold-RLS estimator.

The later joint-current confirmation did not pass its frozen task gates:
recovery tracking improved, but three USV station-keeping cases were
incomplete. Always-on actuation-subspace residual projection and always-on
elastic dynamic-state envelopes were then rejected on separate development
ensembles. A measured-yaw-gated elastic retry rescues the exposed failure with
`0.0030` maximum relaxation, but fresh development produced no completion
rescues and slightly worse yaw/recovery metrics. Its confirmation set remains
untouched. These are retained negative ablations, not promoted defaults.

The kinematic constant-bias current observer is a positive development result:
five fresh matched cases reduced mean Heron current-estimation RMSE from
`0.0425` to `0.0096 m/s`, final position error from `0.131` to `0.062 m`, and
waterjet command variation by about `74%`. It won final yaw on only three of
five cases and one case required solver fallbacks, so its independent
confirmation set remains untouched.

The optional range-aided localization boundary adds accumulated odometry drift,
known and unknown harbor beacons, range noise/bias/dropout, and numerical
observability reports. In the first three-case joint development ensemble,
dead reckoning completes `0/3`; joint SLAM plus a bounded belief-feasibility
retry completes and remains collision-free in `3/3` with zero fallbacks. It
matches the direct-position comparator's completion with a one-step larger mean
completion cost (`158` versus `157`). The candidate uses one retry and `0.00148`
maximum dynamic slack while retaining hard collision, actuator, medium, and
world-domain constraints. This result is not yet independently confirmed. See
`docs/harbor_control_math.md` and `docs/range_aided_slam.md`.

Harbor MPC keeps the physical operating-domain boundary hard. A configurable
interior warning band uses bounded, heavily penalized slack so underactuated
marine agents can recover inside their valid medium without converting land or
surface penetration into an accepted solution. Maximum warning-band use is
reported separately from collision slack.
Velocity and angular-rate envelopes have separately bounded quadratic slack
for explicit feasibility diagnostics. The default controller keeps them hard;
the experimental retry policy can expose the elastic optimizer only near a USV
goal after a measured yaw miss. Collision, actuator, medium, and true domain
constraints remain hard in both solves.

Useful overrides:

`python3 run.py --iterations 1`

`python3 run.py --scenario lane_swap --iterations 2 --max-steps 240 --no-video --output-dir results/lmpc/test_lane_swap`

Validated 2-agent LMPC benchmark:

`python3 run.py --scenario manta_crossover --iterations 2 --max-steps 230 --no-video --output-dir results/lmpc/test_manta_crossover`

Equivalent sweep summary:

`python3 run.py sweep --scenario manta_crossover --iterations 2 --max-steps 230`

`python3 run.py --make-video`

`python3 run.py --config config/manta.yaml --output-dir results/debug_run`

Outputs are written under the output root set in `config/manta.yaml`.
`summary.json` records every candidate iteration in `validation_by_iteration`.
`selected_iteration` is the APF or LMPC iteration used for the final plots.
`solver_clean: false` means the run used a solver fallback or an execution-time
safety intervention. The trajectory can still be valid if the recorded states
are complete and swept-collision-free; inspect both counts in `summary.json`.
`safe: true` with `valid: false` means the rollout avoided collisions but did
not reach every goal, so it is not reused as learned safe-set data.

## Tracked Results and Data
Timestamped run folders are ignored by git. Keep only the current report-ready
snapshot in `results/latest/`.

The current workflow does not require external data files. `data/` is available
for optional local inputs or scratch exports, but its contents are ignored
except for `data/README.md`.

## License
MIT. See `LICENSE`.
