# Harbor Package

The harbor package is independent from the manta-specific seven-state LMPC
stack.

- `models.py`: UGV, USV, and untethered ROV dynamics behind one guidance
  contract.
- `communication.py`: range-, rate-, delay-, TTL-, and dropout-aware message
  delivery with deterministic random seeds.
- `simulation.py`: operating-domain projection, decentralized coordination,
  rollout orchestration, and swept 3D separation metrics.
- `config.py`: strict `config/harbor.yaml` loading and model construction.
- `plotting.py`: optional comparison PNG and coordinated depth-aware GIF.
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
