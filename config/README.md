# Config

`manta.yaml` is the default runtime configuration for manta LMPC.
`harbor.yaml` independently configures the physical UGV/USV/ROV dynamics,
operating domains, coordination, and delayed communication experiments.
`harbor_reduced.yaml` preserves the former reduced models and benchmark.
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
`seed_learning_from_mpc` chooses whether a clean, lower-cost plain MPC rollout
replaces guidance as the first learned safe trajectory. It is enabled for the
physical experiment and disabled only in the frozen reduced benchmark.

`actuator_fault_study` defines hidden effectiveness vectors by agent name, so
RobEn and Inspector-Gadget can have different left/right drive faults even
though both use skid-steer dynamics. Their profiles retain distinct drivetrain
metadata, effective tracks, masses, damping, speed, and force limits.
`actuator_fault_mpc` contains only the
horizon and terminal settings used by that controlled comparison. Scalar
values remain supported for kind-wide robustness experiments; a vector must
match the selected platform's control dimension.

`actuator_fault_ensemble` configures reproducible per-channel Latin-hypercube
coverage for `python run.py harbor-fault-generalization`: unique case seeds,
effectiveness bounds, and paired-bootstrap sample count. It does not replace
the fixed `actuator_fault_study`; the two sections support single-case tracing
and multi-case generalization respectively.

`observation_noise` configures seeded local state-observation noise separately
for UGV, USV, and ROV state layouts, with optional per-agent overrides. Angles
are wrapped and measured dynamic states are projected to each platform's
physical bounds before they reach guidance or MPC. The execution plant and all
safety/completion metrics continue to use the unobserved true state.

`effectiveness_estimator_mode: recursive_diagonal` enables the covariance-form
local actuator estimator. Its forgetting factor, normalized measurement-noise
scale, process-noise floor, and Mahalanobis innovation gate are configured by
the four `effectiveness_rls_*` values. The legacy `diagonal` mode remains the
instantaneous finite-difference comparator.

`time_varying_fault_study` defines execution-only per-agent effectiveness
timelines. Each event switches the hidden plant vector at its configured step;
controllers receive neither the schedule nor its values. The matched study uses
`effectiveness_rls_adaptive_covariance`, `effectiveness_rls_change_threshold`,
and `effectiveness_rls_covariance_inflation` to reopen a locally overconfident
recursive estimate after an unusually large normalized innovation. Persistence
and cooldown limit repeated inflation. These events indicate model surprise,
not a certified fault classification.

`effectiveness_rls_change_detector: cusum` accumulates normalized innovation-
squared excess above `effectiveness_rls_cusum_drift` and inflates covariance at
`effectiveness_rls_cusum_threshold`. `identification_arm_on_change` keeps active
probing dormant until such an event; `identification_reset_on_change` then
clears only that agent's local excitation, information, quota, and rejection
state. The execution-plant schedule remains unavailable to both mechanisms.

`temporary_fault_ensemble` and `temporary_fault_holdout` define separate,
reproducible Latin-hypercube studies over physical-channel loss severity plus
per-agent fault onset and duration. The holdout section is reserved for one-way
evaluation after development settings are fixed. `change_warmup_steps` rejects
startup transients and `change_cooldown_steps` sets the detector refractory
period. `identification_arm_on_loss_only` prevents active probing from being
reopened by a likely restoration event; the direction uses only the local RLS
estimate update, never plant truth.

`obstacle_prediction_mode` selects legacy unbounded `constant_velocity` or
`goal_bounded_velocity` peer extrapolation. The latter retains constant
velocity when peer motion is not aligned with communicated intent, but caps
aligned along-track travel at the communicated goal. Alignment is controlled
by `obstacle_prediction_alignment_threshold`. This changes only obstacle
prediction; hard pairwise constraints and communication delay/TTL remain active.

`active_identification` enables local constraint-aware calibration. Probe
fraction, normalized energy target, interval, minimum successful probes per
channel, maximum rejected probes, and extra communication-clearance guard are
all YAML settings. A requested pulse is solved inside the platform NLP. An
infeasible request is retried without the pulse and recorded separately, so it
cannot silently become an executed fallback.

`identification_strategy` selects `energy` round-robin probing or
`information` scheduling. The latter uses `identification_prior_std`,
`identification_measurement_noise`, and `identification_target_std` to rank
candidate actuator pulses by expected local log-determinant information gain.
`identification_fault_focus_weight` additionally prioritizes channels whose
local transition estimate has departed from nominal effectiveness.
These values tune a linearized scheduling proxy, not a calibrated confidence
interval.

`platform_profiles` lets each agent select an independent named model. The
default UGV profiles are SRI Lab's Jackal-based RobEn and Husky-based Inspector-
Gadget; they do not share mass, inertia, damping, limits, or footprint. Heron
and BlueROV2 Heavy profiles likewise own their parameters. Manufacturer maximum
speed and `mission_speed` are separate. Physical model mass/inertia, damping,
buoyancy, body centers, axis-specific wrench limits, and speed limits are all
YAML values. The physical ROV profile also configures eight thruster-force
limits and the complete 6x8 allocation matrix in `[X,Y,Z,K,M,N]` row and
`[T1,...,T8]` column order.
Unidentified payload and hydrodynamic values remain documented
engineering estimates. Control costs are
normalized by each platform's actuator limits so steering angles, newtons, and
newton-meters remain comparable in the shared objective.

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
