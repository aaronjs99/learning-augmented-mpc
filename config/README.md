# Config

`manta.yaml` is the default runtime configuration for manta LMPC.
`harbor.yaml` independently configures heterogeneous UGV/USV/ROV dynamics,
operating domains, coordination, and delayed communication experiments.
UGV and USV goals are planar `[x, y, yaw]` poses. ROV goals and waypoints are
`[x, y, z, roll, pitch, yaw]` poses for its 12-state, 6-DOF model. Position and
orientation tolerances are configured separately.

Harbor `coordination_policy` selects reciprocal or ETA-priority response.
`priority_response_scale` controls how much reciprocal avoidance the priority
agent retains; `yielding_speed_scale` controls the yielding agent. Network
range, update interval, delay, message TTL, dropout probability, and seed are
all explicit YAML values and never act as physical pose constraints.
`predict_delayed_messages` optionally enables timestamp-aware constant-velocity
state propagation before decentralized conflict evaluation. It is disabled by
default because turning harbor trajectories invalidate that approximation at
high delay.
`guidance_update_interval_steps` controls zero-order-hold block execution and
supports update-load versus safety ablations.
ROV response gains and `control_smoothing` tune finite-rate 6-DOF pose tracking
without changing the UGV/USV controllers.

`harbor_mpc` configures the per-agent distributed optimizer and LMPC terminal
safe set. `terminal_position_only` avoids invalid Euclidean convex combinations
of wrapped angles while full pose tracking and final orientation tolerances
remain active. A rollout is learned only when it is complete, swept-safe,
solver-clean, and uses no collision slack.

It owns the scenario states/goals, obstacle, dynamics constants, APF initializer
tuning, LMPC horizons/weights/slacks, and output defaults. CLI flags in
`run.py` are intended for quick overrides only.

The LMPC defaults are conservative: SVM hyperplanes use a margin larger than
half the configured pairwise safety distance, and terminal safe-set matching is
position-only by default so oscillator phase states do not over-constrain the
short-horizon NLP.

`safety_constraint_substeps` applies obstacle and pairwise half-space
constraints at evenly spaced points inside every MPC interval. A value of `2`
checks the midpoint and endpoint, preventing inter-sample crossings that an
endpoint-only nonlinear program can miss.

`safety_filter_buffer` adds a small execution-time clearance beyond the nominal
obstacle and pairwise limits. This keeps certified trajectories away from
floating-point tangency without changing the obstacle drawn in reports.

`warm_start_control_blend` controls how much stored safe-set control history is
used in IPOPT initialization. `0.0` means constant nominal controls only, while
`1.0` means raw stored controls only.

`terminal_slack_weight` is intentionally high enough to keep agents close to
the staged learned route while retaining bounded slack for nonlinear manta
reachability. The default `10000` reduced unintended waiting-agent drift and
terminal relaxation by about one order of magnitude in short A/B probes without
adding solver failures or runtime.

`static_agent_radius_scales` controls the APF staging search. When
`compact_staging` is enabled, the best `compact_staging_candidates` valid route
sets are also delay-scheduled for concurrent execution. The scheduler supports
up to `compact_staging_max_agents` agents and keeps the original sequential
candidate as a fallback. Recovery behavior is configured with
`fallback_control_levels`, `fallback_diagonal_levels`, and the three
`fallback_*_weight` values; these preserve the original candidate grid and
scoring while making experiments reproducible from YAML.

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
CLI entry points apply overrides through one shared helper, so omitted values
retain YAML defaults consistently across full runs and benchmark sweeps.
