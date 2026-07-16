# Learning

Learning and initialization modules for manta LMPC.

- `apf.py`: iteration-0 APF autopilot with optional extra static obstacles, recorded controls, and one-step APF backup control.
- `safe_sets.py`: dynamically valid sequential and swept-safe compact staging,
  control-history construction, and terminal sampling.
- `hyperplanes.py`: SVM spatial hyperplanes for pairwise avoidance, including asymmetric margins for priority-aware coordination.
- `policies.py`: reusable priority allocation and safe-memory warm-start policies.
- `recovery.py`: bounded APF fallback selection and optional staged terminal repair.
- `safety.py`: synchronous swept-transition filtering before controls execute.
- `runner.py`: APF baseline plus repeated decentralized LMPC orchestration and safe-set updates.

Trajectory validation and learning-cost evaluation live in `scripts/metrics/` so
the run CLI, sweep CLI, plotting, and learning loop share one implementation.

Tuning values for these modules are loaded from `config/manta.yaml`.

Only rollouts that are both collision-free and complete are added back into the
learned safe set. Safe-but-incomplete LMPC attempts are still reported in
`summary.json`, but they are not used as terminal safe-set data.

Compact APF staging delay-schedules the best order-conditioned routes for
concurrent execution. It replays every delayed control through the full
seven-state dynamics, validates swept separation independently, and retains the
original sequential route as an automatic fallback.

Safety is checked independently of the optimizer. Unsafe proposed transitions
are replaced by bounded APF actions or zero-translation holds, and the report
separates these interventions from IPOPT fallback events.

Successful optimizer steps retain maximum static, hyperplane, and absolute
terminal slack telemetry. Run summaries distinguish unavailable APF/fallback
telemetry from solved zero-slack steps and report both maxima and nonzero-use
counts, both globally and per agent.

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
