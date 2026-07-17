# HARBOR Control and Estimation Mathematics

This document is the compact mathematical contract for the heterogeneous
harbor stack. Detailed platform derivations and parameter provenance remain in
[`harbor_dynamics.md`](harbor_dynamics.md). Range-estimator implementation
details remain in [`range_aided_slam.md`](range_aided_slam.md).

## 1. Platform contracts

Each platform has discrete dynamics

```text
x_(k+1) = f_i(x_k, u_k, dt),  i in {UGV, USV, ROV}.
```

The default models are:

| Platform | State | Control | Goal |
|---|---|---|---|
| RobEn / Inspector-Gadget UGV | `[x,y,psi,v,r]` | left/right side force | `[x,y,psi]` |
| Heron USV | `[x,y,psi,u,v,r]` | port/starboard jet thrust | `[x,y,psi]` |
| BlueROV2 Heavy | `[x,y,z,phi,theta,psi,u,v,w,p,q,r]` | eight thruster forces | six-DOF pose |

The UGV, 3-DOF marine, and 6-DOF marine equations are given in Sections 1-3
of `harbor_dynamics.md`. The same NumPy and CasADi transitions are regression
tested against each other.

## 2. Hidden execution plant

The controller predicts with `f_i`; the simulated plant applies unknown
channel effectiveness and marine current:

```text
u_applied,k = diag(alpha_k) u_commanded,k
x_nominal,k+1 = f_i(x_k, u_applied,k, dt)
p_k+1 = p_nominal,k+1 + dt v_current,i.
```

`alpha_k` may change independently for every physical drive, waterjet, or
thruster channel. Current has horizontal components for Heron, three components
for BlueROV2, and is zero for UGVs. Neither hidden quantity is exposed to the
controller.

## 3. Distributed prediction

Agent `i` solves its own nonlinear program from local estimate `x_hat_i` and
received peer messages `(p_j, v_j, g_j, timestamp)`. No optimizer owns another
agent's state or control. A goal-bounded peer prediction is

```text
d_j = v_j / ||v_j||
s_j = max(d_j^T (g_j-p_j), 0)
p_j(t) = p_j + d_j min(||v_j||t, s_j)
```

when velocity points sufficiently toward the communicated goal; otherwise it
uses constant velocity. Delay, update interval, TTL, range, and packet dropout
are configured independently.

## 4. Per-agent nonlinear MPC

For horizon `N`, agent `i` chooses `U={u_0,...,u_(N-1)}` and predicted states
`X={x_0,...,x_N}`. The initial estimate is fixed:

```text
x_0 = x_hat_i,k.
```

The adaptive predictor is

```text
x_(t+1) = drift(f_i(x_t, diag(alpha_hat_i,k) u_t, dt), r_hat_i,k, dt).
```

For marine agents, `drift` adds `dt r_hat` to predicted position. The finite
horizon objective implemented in `scripts/harbor/mpc.py` is

```text
J_i = sum_(t=0)^(N-1) [
        w_p ||p_(t+1)-p_goal||_2^2
      + w_R e_R(x_(t+1),g)^2
      + w_v ||nu_(t+1)||_2^2
      + w_u ||u_t / u_scale||_2^2
      + w_du ||(u_t-u_(t-1))/u_scale||_2^2
    ]
    + w_f V_pose(x_N,g)
    + J_collision + J_domain + J_dynamic + J_LMPC.
```

Planar orientation error and each ROV Euler-angle error use the smooth periodic
cost

```text
e_angle^2 = 2(1-cos(theta-theta_goal)).
```

All weights map directly to `harbor_mpc.*_weight` in `config/harbor.yaml`.

### Physical constraints

Actuator constraints are componentwise hardware bounds:

```text
u_min <= u_t <= u_max.
```

For a communicated peer `j`, swept separation is approximated at every MPC
sample by

```text
||p_i,t - p_hat_j,t||_2^2 + s_collision,i,j,t
    >= (R_i + R_j + b_collision)^2.
```

The default sets `s_collision=0`; any experimental collision slack is bounded,
heavily penalized, and reported. Curated safe-set admission requires maximum
collision slack at numerical zero and independently checks swept continuous
segments.

Physical operating domains remain hard:

```text
l_axis <= p_axis,t <= u_axis.
```

An interior warning margin `m` uses bounded diagnostic slack without relaxing
the physical boundary:

```text
p_axis,t >= l_axis + m - s^-_axis,t
p_axis,t <= u_axis - m + s^+_axis,t
0 <= s^-,s^+ <= m.
```

This keeps UGVs on land, Heron at the water surface, and BlueROV2 below the
surface and above the seabed.

## 5. Learning MPC terminal set

A complete, collision-free, zero-fallback rollout supplies sampled terminal
states `x_safe,j` and stored cost-to-go `Q_j`. LMPC introduces convex weights

```text
lambda_j >= 0,  sum_j lambda_j = 1
x_N,terminal = sum_(j=1)^K lambda_j x_safe,j + s_terminal.
```

Its terminal contribution is

```text
J_LMPC = w_Q sum_j lambda_j Q_j
       + w_s ||s_terminal||_2^2
       + w_f V_pose(x_N,g).
```

The default terminal hull uses 2D positions for UGV/USV and 3D position for
ROV. Full 3-DOF/6-DOF pose remains in tracking and final rollout acceptance.
New memory is admitted only if the true rollout is complete, swept-safe,
fallback-free, collision-slack-free, and no more expensive than retained
memory.

## 6. Range-aided belief state

Odometry propagation and range observations are

```text
p_hat_k^- = p_hat_(k-1)^+ + delta_p_odom,k + b_odom
P_k^- = P_(k-1)^+ + Q_odom
z_j,k = ||p_k-l_j||_2 + b_range + v_j,k.
```

Known-map localization estimates `x_b=p`. Joint SLAM estimates

```text
x_b = [p, l_1, ..., l_m].
```

For `d_j=p-l_j`, `rho_j=||d_j||`, and `q_j=d_j/rho_j`,

```text
h_j(x_b)=rho_j
H_j=[q_j, 0,...,-q_j,...,0].
```

The EKF update is

```text
S_j = H_j P^- H_j^T + sigma_range^2
K_j = P^- H_j^T S_j^-1
x_b^+ = x_b^- + K_j(z_j-b_range-h_j(x_b^-))
P^+ = (I-K_jH_j)P^-(I-K_jH_j)^T + K_j R K_j^T.
```

The last line is Joseph form. Noise, bias, maximum range, update period,
dropout, odometry drift, priors, and beacon positions are YAML-owned.

### Observability-aware map coupling

Over a window of `W` accepted ranges,

```text
O_k = [H_(k-W+1)^T ... H_k^T]^T.
```

The estimator records `rank(O_k)`, `sigma_min(O_k)`, and `cond(O_k)`. Full
numerical observability requires

```text
rank(O_k) = dim(x_b).
```

Unknown-landmark measurements are deferred until fixed-anchor Jacobians span
the platform position subspace. While an unknown landmark's posterior standard
deviation exceeds `landmark_pose_coupling_std`, its update changes the map but
not the navigation pose or pose-covariance block. This prevents an uncertain
map from steering the vehicle.

## 7. Actuator-effectiveness RLS

Let `z` collect measured velocity/rate states and let `J_alpha` be the finite-
difference sensitivity of the next dynamic state to channel effectiveness.
The diagonal recursive estimator is

```text
P_alpha^- = P_alpha/lambda + Q_alpha
e_alpha = z_k - z_model(alpha_hat_(k-1))
S_alpha = J_alpha P_alpha^- J_alpha^T + R_alpha
K_alpha = P_alpha^- J_alpha^T S_alpha^-1
alpha_hat_k = clip(alpha_hat_(k-1) + K_alpha e_alpha)
P_alpha,k = (I-K_alpha J_alpha)P_alpha^-(I-K_alpha J_alpha)^T
            + K_alpha R_alpha K_alpha^T.
```

Large normalized innovation can reopen covariance after an abrupt loss. The
confirmed transient-recovery layer leaves raw RLS untouched and applies a
bounded, decaying controller-only offset after a locally detected, rank-
sufficient restoration event.

## 8. Actuator-independent current observer

Model-residual current estimation can confuse actuator mismatch with current.
The kinematic observer instead uses measured ground velocity and measured
through-water body velocity:

```text
v_ground,k = (p_k-p_(k-1))/Delta t
v_water,world,k = R(eta_(k-1)) nu_water,k
y_current,k = v_ground,k - v_water,world,k.
```

For a constant-bias current state `c`, each axis uses

```text
P_c^- = P_c + Q_c
K_c = P_c^- / (P_c^- + R_c)
c_hat_k = c_hat_(k-1) + K_c(y_current,k-c_hat_(k-1))
P_c,k = (1-K_c)P_c^-.
```

Innovation gating rejects outliers. Estimation runs at plant rate, even when
the nonlinear controller replans less frequently.

## 9. Belief-feasibility retry

Noisy estimated velocity can place the fixed MPC initial condition near a hard
dynamic envelope where exact prediction is numerically infeasible. The primary
solve keeps

```text
l_dynamic <= x_dynamic,t <= u_dynamic.
```

Only after that solve fails, the belief retry permits

```text
l_dynamic-s^-_t <= x_dynamic,t <= u_dynamic+s^+_t
0 <= s^-_t,s^+_t <= s_belief
J_dynamic = w_dynamic sum_t ||s_t||_2^2.
```

The retry does **not** relax collision separation, actuator bounds, land/water
membership, depth limits, or true world-domain constraints. Retry count and
maximum accepted dynamic slack are recorded. In the current three-case
development study, joint SLAM plus this retry is complete and collision-free in
`3/3` cases with zero fallbacks; the hardest case uses one retry and `0.00148`
maximum state-unit slack. This is development evidence, not confirmation.

## 10. Verification metrics

Plant truth, never controller belief, determines:

```text
position error       = ||p_true,T-p_goal||_2
orientation error    = periodic pose error at T
completion           = all pose errors within tolerance for the hold window
safety               = zero swept pairwise violations
localization RMSE     = sqrt(mean_k ||p_hat_k-p_true,k||_2^2)
current RMSE          = sqrt(mean_k ||c_hat_k-c_true||_2^2).
```

Solver fallbacks, dynamic retries, all slack maxima, communication delivery,
observability, actuator RMSE, and completion cost are stored separately so one
metric cannot conceal failure in another.

## 11. Robust fixed-lag factor graph

The EKF is retained as a directly comparable baseline. The experimental
smoother in `scripts/harbor/factor_graph.py` stores a window of platform poses
and unknown beacon positions. It minimizes prior, odometry, robust range,
attitude, and beacon-prior residuals. A range residual is

```text
r_range = (||p_k-l_b|| - z_kb) / sigma_range
```

Huber IRLS uses `w(r)=1` for `|r| <= delta` and `w(r)=delta/|r|` otherwise.
Once `fixed_lag` is exceeded, the oldest pose is removed and represented by a
retained prior. `factor_iterations`, damping, and the Huber threshold are
configurable. Each update reports residual RMS, rejected factors, window size,
Jacobian rank, smallest singular value, and condition number. The dependency-
free solver is a transparent research reference; solve time must be reported
before claiming real-time performance.
