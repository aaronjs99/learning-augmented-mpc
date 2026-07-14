# Config

`manta.yaml` is the default runtime configuration for the project.

It owns the scenario states/goals, obstacle, dynamics constants, APF initializer
tuning, LMPC horizons/weights/slacks, and output defaults. CLI flags in
`run.py` are intended for quick overrides only.

The LMPC defaults are conservative: SVM hyperplanes use a margin larger than
half the configured pairwise safety distance, and terminal safe-set matching is
position-only by default so oscillator phase states do not over-constrain the
short-horizon NLP.

`warm_start_control_blend` controls how much stored safe-set control history is
used in IPOPT initialization. `0.0` means constant nominal controls only, while
`1.0` means raw stored controls only.

`priority_hyperplanes` enables asymmetric pairwise margins. The total
separation budget for each pair is preserved, but the lower-priority agent gets
the larger half-space margin. `priority_metric: goal_distance` gives
right-of-way to agents closer to their goals; `remaining_safe_time` is available
for experiments that favor agents with longer stored routes.

`repair_incomplete_with_apf` optionally enables a capped terminal repair phase
for safe but incomplete LMPC attempts. The repair follows stored safe-set
waypoints instead of aiming directly at the final goal, but it is off by default
until it is more reliable.

Configuration dataclasses validate dimensions, bounds, supported policy names,
and positive step/horizon values as soon as YAML is loaded. Invalid experiments
therefore fail before CasADi builds or starts solving an optimization problem.
