# Range-Aided Harbor Estimation

## Scope

The simulator can put guidance, distributed MPC, or distributed LMPC behind an
estimated-state boundary. Plant truth is used only to propagate vehicles,
synthesize ranges, and score safety. Controllers receive noisy onboard states
whose position is replaced by a local range-aided estimate.

The implementation supports 2D UGV/USV position estimation and 3D ROV position
estimation. Attitude and velocity still come from the configured onboard
observation model. Full 3-DOF/6-DOF pose estimation remains future work.

## Models

Dead-reckoning propagation and scalar range sensing are

```text
p[k+1] = p[k] + delta_p_odom[k] + b_odom + w[k],  w ~ N(0, Q)
z_j = ||p - l_j||_2 + b_range + v_j,              v ~ N(0, sigma_range^2)
```

Known-anchor localization estimates `x = p`. Joint landmark SLAM estimates
`x = [p, l_1, ..., l_m]` for every beacon not marked `fixed`. With
`u_j = (p - l_j) / ||p - l_j||`, its Jacobian is

```text
H_j = [u_j, 0, ..., -u_j, ..., 0].
```

The EKF uses a Joseph-form covariance update. Bias, Gaussian noise, maximum
range, update rate, dropout, odometry drift, priors, and random seed are all
configured under `range_aided_slam` in `config/harbor.yaml`.

For the robust fixed-lag baseline, set `mode: fixed_lag_slam`. It supports 2D
UGV/USV positions and full six-component ROV pose states, with explicit
odometry, range, attitude, and beacon-prior factors. Huber reweighting records
downweighted range factors instead of silently accepting them as truth.

The range sensor also exposes deterministic development faults: ordinary and
burst dropout, positive NLOS bias, heavy-tailed outliers, and per-run telemetry.
Their probabilities, magnitudes, burst duration, and seed are configured in
`range_aided_slam`; fault telemetry is evidence and is never supplied to the
controller as truth.

Delayed or stale measurements carry their capture step and are delivered through
a deterministic queue. `active_observability` can insert a bounded local
information detour only while the estimator is weakly observable. The detour is
limited by `information_max_excursion`; normal task guidance resumes as soon as
the geometry becomes observable. Physical domain and collision constraints stay
in the MPC layer.

## Observability

The estimator stores a sliding window of measurement Jacobians and reports

```text
O = [H_(k-W+1)^T ... H_k^T]^T
rank(O), sigma_min(O), cond(O).
```

It declares numerical observability only when `rank(O) = dim(x)` at the YAML
tolerance. Unknown-beacon-only SLAM exposes gauge freedom as a rank deficiency.
Fixed non-collinear harbor anchors establish the position frame. Range loss or
poor geometry lowers rank or the smallest singular value.

Joint SLAM uses two navigation-preserving gates. Unknown-beacon updates are
deferred until recent fixed-anchor Jacobians span platform position. While a
landmark is uncertain, its range updates the map but not navigation pose or
pose covariance; coupling is allowed only below the configured landmark
standard-deviation threshold. These are development mechanisms, not yet
independently confirmed research claims.

## Configured Problems

- GPS-denied dead reckoning with accumulated odometry noise and bias.
- Known-anchor ranges with bias, noise, range limits, and update-rate limits.
- Joint unknown-beacon SLAM with explicit gauge and geometry diagnostics.
- Seeded communication or acoustic range dropout.
- Hidden current and temporary per-channel actuator loss from the existing
  disturbance and fault schedules.
- Joint localization, current, actuation, and communication uncertainty. The
  interfaces compose, but a distributed-LMPC confirmation is still required.

Run `python run.py harbor-localization-study`. The matched estimator diagnostic
reduces BlueROV2 position RMSE from about `0.579 m` under dead reckoning to
`0.076 m` with a known map and `0.105 m` with joint unknown-landmark SLAM.

Run `python run.py harbor-joint-localization-study` for the closed-loop test
with distributed MPC, hidden current, observation noise, and temporary
per-channel actuator losses. On the three-case development ensemble, dead
reckoning completes `0/3`, known-map localization completes `2/3`, plain joint
SLAM completes `3/3` with three solver fallbacks, and joint SLAM with the
belief-feasibility retry completes safely in `3/3` with zero fallbacks. The
candidate uses one retry over all three cases and `0.00148` maximum dynamic-
envelope slack. This is positive development evidence; an untouched
confirmation ensemble is still required.
