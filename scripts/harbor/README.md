# Harbor Package

The harbor package is independent from the manta-specific seven-state LMPC
stack. It contains guidance, per-agent distributed MPC, and safe-set LMPC
controllers behind the same simulation and communication boundary.

- `models.py`: kinematic-bicycle UGV, 3-DOF marine USV, and 6-DOF marine ROV,
  each owning matching numerical and symbolic transitions.
- `communication.py`: range-, rate-, delay-, TTL-, and dropout-aware message
  delivery with deterministic random seeds.
- `simulation.py`: operating-domain projection, decentralized coordination,
  rollout orchestration, and swept 3D separation metrics.
- `config.py`: strict `config/harbor.yaml` loading and model construction.
- `plotting.py`: safety envelopes, successful-rollout samples, pose-goal
  headings, ROV depth/attitude diagnostics, and a coordinated GIF.
- `experiments.py`: communication robustness and matched-horizon MPC/LMPC sweeps.
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

The default equations are physically structured but use illustrative YAML
coefficients. `config/harbor_reduced.yaml` preserves the previous reduced
models. Exact equations and fidelity limits are in `docs/harbor_dynamics.md`.

The ROV guidance loop uses finite velocity/attitude response gains and
configurable command smoothing instead of one-step saturated pose correction.
The animation shows yaw in the top-down panel and ROV pitch in side elevation;
the static diagnostic includes roll, pitch, and yaw histories.
