# Harbor Package

The harbor package is independent from the manta-specific seven-state LMPC
stack. It contains guidance, per-agent distributed MPC, and safe-set LMPC
controllers behind the same simulation and communication boundary.

- `models.py`: dynamic skid-steer UGV, 3-DOF marine USV, and 6-DOF marine ROV,
  each owning matching numerical and symbolic transitions.
- `communication.py`: range-, rate-, delay-, TTL-, and dropout-aware message
  delivery with deterministic random seeds.
- `simulation.py`: operating-domain projection, configurable hidden plant
  disturbances, sustained-goal evaluation, rollout orchestration, and swept
  3D separation metrics.
- `config.py`: strict `config/harbor.yaml` loading and model construction.
- `plotting.py`: safety envelopes, successful-rollout samples, pose-goal
  headings, ROV depth/attitude diagnostics, and a coordinated GIF.
- `experiments.py`: communication robustness, matched-horizon MPC/LMPC sweeps,
  and nominal-versus-joint-adaptive model-mismatch trials.
- `mpc.py`: platform-specific per-agent CasADi optimizers using local state and
  received messages, with hard collision constraints.
- `learning.py`: verified guidance seed, plain MPC baseline, clean-rollout
  admission, terminal safe-set hull, and learned time-to-go iterations.

`reciprocal` coordination makes both agents maneuver around a closing conflict.
`eta_priority` estimates each agent's remaining travel time from communicated
state/goal data; the lower-priority agent yields strongly while the priority
agent retains a configurable fraction (`priority_response_scale`) of reciprocal
avoidance. Communication never modifies another platform's state and does not
create a tether, formation lock, or relative-pose constraint.

When `predict_delayed_messages` is enabled, received positions are propagated
from their send timestamp with the communicated world velocity before conflict
evaluation. It is off by default because that approximation regresses for
turning USV/ROV motion. `python run.py harbor-sweep` measures safety and final
completion separately over seeded delay/dropout trials and writes only ignored
temporary artifacts by default.

`guidance_update_interval_steps` applies the block-replanning concept retained
from the audited legacy `distmpc` prototype. Controls are held between guidance
updates, and `guidance_update_count` makes the computational reduction explicit.

`python run.py harbor-lmpc` turns the successful guidance rollout into the
initial safe trajectory, compares plain distributed MPC, and then runs LMPC.
Only complete, swept-safe, zero-fallback, zero-collision-slack rollouts can
replace the learned trajectory. The terminal hull uses position coordinates;
full 3-DOF/6-DOF orientation remains in tracking and final acceptance.

Named YAML profiles distinguish SRI Lab's Jackal-based RobEn from its Husky-
based Inspector-Gadget and configure a full-payload Heron and BlueROV2 Heavy.
Controllers operate on left/right UGV drive sides, port/starboard Heron
waterjets, and all eight BlueROV2 Heavy T200 channels.
Manufacturer limits are separated from conservative mission cruise speeds;
unidentified payload and hydrodynamic coefficients are documented assumptions.
`config/harbor_reduced.yaml` preserves the previous reduced models.

The ROV guidance loop uses finite velocity/attitude response gains and
configurable command smoothing instead of one-step saturated pose correction.
The animation shows yaw in the top-down panel and ROV pitch in side elevation;
the static diagnostic includes roll, pitch, and yaw histories.

`python run.py harbor-robustness` executes every controller against the same
hidden current and actuator-effectiveness settings from `disturbance_study`.
The joint estimator is local and causal. It first fits scalar control
effectiveness from the measured velocity/rate response to the agent's previous
command. It then compares measured position with the effectiveness-adjusted
prediction and inserts the filtered remaining world-velocity residual into its
own model. It receives neither configured disturbance nor another agent's
state. The hold window prevents transient tolerance crossings from being
reported as successful station keeping.

`python run.py harbor-fault-study` isolates asymmetric actuator faults without
current. It compares nominal, scalar-adaptive, and diagonal-adaptive distributed
MPC plus diagonal-adaptive LMPC. Each named platform owns its hidden vector and
local estimate. In particular, RobEn and Inspector-Gadget are separate UGVs
with separate model parameters and left/right effectiveness vectors. Heron and
BlueROV2 estimates are likewise waterjet- and thruster-specific. Active trials
request bounded alternating first-step pulses for under-observed channels
inside the normal NLP. The information-aware trial ranks admissible pulses by
local expected Fisher-information gain and is compared against an equal-count
one-pass schedule. Unsafe or
infeasible requests are skipped or retried without the pulse, and repeated-task
LMPC retains its own preceding local estimate and clean rollout.
