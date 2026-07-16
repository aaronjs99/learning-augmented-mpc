# Harbor Package

The harbor package is independent from the manta-specific seven-state LMPC
stack. It is a distributed guidance baseline, not yet an LMPC controller.

- `models.py`: 3-DOF-pose UGV/USV models and a 6-DOF-pose, 12-state untethered
  ROV behind one guidance contract.
- `communication.py`: range-, rate-, delay-, TTL-, and dropout-aware message
  delivery with deterministic random seeds.
- `simulation.py`: operating-domain projection, decentralized coordination,
  rollout orchestration, and swept 3D separation metrics.
- `config.py`: strict `config/harbor.yaml` loading and model construction.
- `plotting.py`: safety envelopes, successful-rollout samples, pose-goal
  headings, ROV depth/attitude diagnostics, and a coordinated GIF.
- `experiments.py`: seeded communication delay/dropout robustness sweeps.

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

Successful rollout samples in the plots are diagnostic candidates only. They
do not become a learned safe set until a future per-agent MPC uses them in a
terminal constraint and cost-to-go approximation.

The ROV guidance loop uses finite velocity/attitude response gains and
configurable command smoothing instead of one-step saturated pose correction.
The animation shows yaw in the top-down panel and ROV pitch in side elevation;
the static diagnostic includes roll, pitch, and yaw histories.
