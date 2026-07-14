# Learning

Learning and initialization modules for manta LMPC.

- `apf.py`: iteration-0 APF autopilot with optional extra static obstacles, recorded controls, and one-step APF backup control.
- `safe_sets.py`: dynamically valid staged safe-set and control-history construction plus terminal sampling.
- `hyperplanes.py`: SVM spatial hyperplanes for pairwise avoidance, including asymmetric margins for priority-aware coordination.
- `policies.py`: reusable priority allocation and safe-memory warm-start policies.
- `recovery.py`: bounded APF fallback selection and optional staged terminal repair.
- `runner.py`: APF baseline plus repeated decentralized LMPC orchestration and safe-set updates.

Trajectory validation and learning-cost evaluation live in `scripts/metrics/` so
the run CLI, sweep CLI, plotting, and learning loop share one implementation.

Tuning values for these modules are loaded from `config/manta.yaml`.

Only rollouts that are both collision-free and complete are added back into the
learned safe set. Safe-but-incomplete LMPC attempts are still reported in
`summary.json`, but they are not used as terminal safe-set data.

LMPC warm starts use stored safe-set controls blended with a nominal constant
control, which keeps IPOPT close to dynamically plausible prior motion without
fully inheriting APF or previous-iteration control artifacts.

Priority-aware hyperplanes keep the total pairwise margin fixed while assigning
more of the margin to the lower-priority agent. This is a decentralized
right-of-way heuristic for reducing multi-agent traffic jams. The default
priority metric favors agents closer to their goals, while an experimental
remaining-safe-time metric can favor agents with longer stored routes.

Safe but incomplete LMPC attempts can optionally be finished with a capped APF
terminal repair. The repair follows waypoints from the stored safe-set route so
it is less prone to direct-to-goal APF local minima, but it is off by default
until the repair strategy is stronger.
