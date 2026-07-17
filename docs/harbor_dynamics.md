# Harbor Dynamics

The default harbor experiment uses physically structured platform models. Each
model owns both its NumPy transition and its CasADi symbolic transition; a
regression test evaluates both from the same state and control and requires
machine-precision agreement. Angles remain continuous internally and are
wrapped only when pose errors are evaluated.

The default YAML profiles are tied to the lab's named platforms and public
vehicle specifications. Hydrodynamic added mass and damping remain literature-
based or engineering estimates rather than coefficients identified on the
specific hardware. The correct claim is therefore *platform-grounded,
mechanistically structured simulation*, not hardware-validated high fidelity.

## UGVs: dynamic skid steer

RobEn and Inspector-Gadget are different UGVs. The default profiles model the
[SRI Lab RobEn as a Jackal and Inspector-Gadget as a Husky](https://sri-lab.seas.ucla.edu/robotic-infrastructure-inspection/).
Each uses state `q = [x, y, psi, v, r]`. The optimizer commands physical
left/right drive-side forces `u = [F_L, F_R]`, mapped through the platform's
effective track `b`:

```text
F = F_L + F_R
N = (b / 2) (F_R - F_L)
```

The resulting body dynamics are:

```text
x_dot   = v cos(psi)
y_dot   = v sin(psi)
psi_dot = r
m v_dot = F - d_v v - d_v2 |v|v
I_z r_dot = N - d_r r - d_r2 |r|r
```

RobEn uses a 24 kg working-mass estimate around the 17 kg Jackal base; the exact
Ladybug/lidar mast mass is not available. Inspector-Gadget uses a 65 kg working-
mass estimate around the 50 kg Husky A200 base. Their inertia, damping, force,
moment, mission-speed, and footprint values are independent. Public base values
come from the [Jackal specification](https://clearpathrobotics.com/jackal-small-unmanned-ground-vehicle/)
and [Husky A200 manual](https://www.clearpathrobotics.com/wp-content/uploads/2013/02/Husky-A200-UGV-UserManual-0.12.pdf).

RobEn uses Jackal's compact four-wheel-skid drivetrain and a 0.43 m track;
Inspector-Gadget uses Husky A200's two-motor drivetrain and documented 0.555 m
effective track. The lumped side-force model omits individual wheel speeds,
longitudinal/lateral slip, terrain-dependent traction, suspension, and motor
current dynamics. Those are the next identification layer for hardware transfer.

## USV: underactuated 3-DOF marine craft

Earth-fixed pose `eta = [x, y, psi]` and body velocity `nu = [u, v, r]` satisfy:

```text
eta_dot = R(psi) nu
M nu_dot + C(nu) nu + D(nu) nu = tau
D(nu)nu = D_linear nu + D_quadratic |nu| nu
```

The optimizer commands port/starboard waterjet thrust `u_c = [T_P, T_S]`.
With jet separation `b_j`, `T = T_P + T_S` and
`N = (b_j/2)(T_S - T_P)`, producing the underactuated body input
`tau = [T, 0, N]`.

`M = diag(m_u, m_v, I_z)` includes effective surge/sway mass and yaw inertia.
The Coriolis product used by both simulator and optimizer is:

```text
C(nu)nu = [-m_v v r,
             m_u u r,
            (m_v - m_u) u v]
```

Because the USV and skid-steer UGVs cannot command lateral velocity directly,
their MPC approach pose uses the line-of-sight heading while translational
guidance is active,

```text
psi_approach = atan2(v_des,y, v_des,x),  ||v_des|| > 0
psi_approach = psi_goal,                  ||v_des|| = 0
```

then restores the requested final yaw for station keeping. This prevents the
underactuated optimizer from settling at zero control just outside the position
tolerance while trying to preserve final yaw throughout the approach.

Unlike the former scalar-surge model, this state includes sway and yaw-rate
dynamics. Its nominal transition remains a calm-water model. The robustness
study applies current outside that transition as execution-plant mismatch;
the Heron profile retains the 1.7 m/s hardware limit, 40 N total thrust, and a
39 kg full-payload rigid-mass basis. A separate 0.95 m/s mission speed keeps
inspection control away from the hardware ceiling. Effective mass and damping
are documented engineering estimates. Wind, waves, off-diagonal added mass,
and waterjet thrust curves remain future identification targets. See
the [Heron manual](https://www.generationrobots.com/media/clearpath_heron_usermanual.pdf).

## ROV: 6-DOF marine craft

Earth-fixed pose `eta = [x, y, z, phi, theta, psi]`, body velocity
`nu = [u, v, w, p, q, r]`, and body wrench
`tau = [X, Y, Z, K, M, N]` satisfy:

```text
eta_dot = J(eta) nu
M nu_dot + C(nu) nu + D(nu) nu + g(eta) = tau
```

`J` contains the full ZYX body-to-world rotation and Euler-rate transform.
`M` is a configurable positive diagonal effective inertia. `C(nu)nu` is built
from linear and angular momentum cross products, `D` contains linear and
quadratic damping, and `g` contains gravity/buoyancy forces plus moments from
the configured centers of gravity and buoyancy. The BlueROV2 Heavy profile uses
a near-neutral 11.5 kg rigid-mass basis, published diagonal added-mass/damping
values, and axis-specific wrench limits. Heavy is selected because the task
requires independently controlled 6-DOF goals; the standard six-thruster
BlueROV2 is only 5-DOF. Its center of buoyancy is above its center of gravity,
producing roll/pitch restoring moments.

The ROV has a real 12-state trajectory, eight bounded T200 force inputs, 6-DOF
pose goal, body-frame velocity, depth limits, and attitude dynamics. Its
allocation is

```text
tau = B f_thruster,    B in R^(6 x 8),    rank(B) = 6
```

The horizontal vectored thrusters provide surge, sway, and yaw; the four Heavy
vertical thrusters provide heave, roll, and pitch. The coefficient signs follow
[ArduSub's `VECTORED_6DOF` motor matrix](https://github.com/ArduPilot/ardupilot/blob/master/libraries/AP_Motors/AP_Motors6DOF.cpp),
while row magnitudes are calibrated to the configured BlueROV2 Heavy bollard-
force and moment limits. The complete matrix is YAML-owned and rank-validated
when the platform profile loads. Guidance uses a
pseudoinverse allocation followed by uniform desaturation, and MPC optimizes
the eight bounded forces directly. The model does not yet include asymmetric
forward/reverse T200 curves, off-diagonal added mass, Euler-angle singularity
avoidance, or wave forces. Public dimensions, mass, speed, and thrust are from the
[BlueROV2 page](https://bluerobotics.com/store/rov/bluerov2/) and
[Heavy retrofit](https://bluerobotics.com/store/rov/bluerov2-accessories/brov2-heavy-kit/).

## Execution-plant mismatch and joint local model

For marine platforms, the robustness plant advances nominal dynamics and then
adds a hidden world-frame current `v_c` to position:

```text
p_(k+1) = p_nominal_(k+1) + dt v_c
u_applied = alpha_platform u_commanded
```

The USV uses only horizontal current; the ROV uses all three components. The
UGV is not advected. Optional actuator effectiveness is also applied only in
execution. The controller does not read either value. The scalar robustness
experiment uses one shared gain per platform. The asymmetric fault experiment
instead uses a diagonal gain matrix over each platform's physical actuator channels:

```text
u_applied = diag(alpha_1, ..., alpha_nu) u_commanded
```

For RobEn and Inspector-Gadget, these are separate `[left drive, right drive]`
vectors; the two UGVs neither share an estimate nor share hidden fault values.
For Heron they are `[port jet, starboard jet]`, and for BlueROV2 Heavy they are
the eight physical channels `[T1, ..., T8]`.

Let `z` denote velocity/rate states, which are unaffected by current advection.
The scalar estimator uses a finite-difference sensitivity around the prior
effectiveness estimate:

```text
s_alpha = (z_model(alpha+epsilon) - z_model(alpha-epsilon)) / (2 epsilon)
alpha_measured = alpha_hat + s_alpha^T (z_k - z_model(alpha_hat))
                              / (s_alpha^T s_alpha)
alpha_hat_k = clip((1-beta) alpha_hat_(k-1) + beta alpha_measured)
```

The channel-wise estimator forms one sensitivity column for every sufficiently
excited command channel:

```text
S_j = (z_model(alpha + epsilon e_j) - z_model(alpha - epsilon e_j))
      / (2 epsilon)
delta_alpha = argmin_delta ||S delta - (z_measured - z_model(alpha_hat))||_2
alpha_hat_next = (1-gamma) alpha_hat
                 + gamma clip(alpha_hat + delta_alpha)
```

Only locally measured state and the platform's prior command enter this update.
An unexcited or dynamically indistinguishable channel remains at its prior;
this is an observability result, not evidence that the actuator is healthy.
The model identifies physical left/right UGV drives, Heron port/starboard
waterjets, and all eight BlueROV2 Heavy thrusters. RobEn's four wheels are
grouped into the two drive-side commands exposed by its base controller rather
than treated as independently commandable motors. An unidentifiable actuator
combination therefore reflects insufficient excitation, not a missing
allocation model.

The noisy benchmark gives each agent only a seeded local observation

```text
y_k = project_D(x_k + epsilon_k),  epsilon_k ~ N(0, diag(sigma_platform^2))
```

where `project_D` wraps angles and enforces platform pose-domain and dynamic-
state bounds. The plant transition, collision checks, and completion metrics
still use `x_k`. The robust estimator normalizes dynamic-state innovation by
the platform's velocity/rate limits and applies a covariance-form recursive
update:

```text
P_k^- = P_(k-1) / lambda + Q
e_k = z_k - z_model(alpha_hat_(k-1))
S_k = J_k P_k^- J_k^T + R
K_k = P_k^- J_k^T S_k^-1
alpha_hat_k = clip(alpha_hat_(k-1) + K_k e_k)
P_k = (I-K_k J_k) P_k^- (I-K_k J_k)^T + K_k R K_k^T
```

For each bounded position coordinate, MPC applies the physical domain to
predicted states `x_1,...,x_N`, while the measured initial state remains the
fixed equality `x_0=y_k`. With domain interval `[l,u]`, warning margin `m`, and
bounded warning-band slacks `s^-`, `s^+`, the constraints are

```text
l <= x_t <= u
x_t >= l + m - s_t^-
x_t <= u - m + s_t^+
0 <= s_t^-, s_t^+ <= m
J_domain = w_domain sum_t (||s_t^-||_2^2 + ||s_t^+||_2^2)
```

Warning-band use therefore cannot relax the true land, water, or depth
boundary. It is reported separately from collision slack. Applying dynamic-
state bounds only to future states also avoids requiring a noisy fixed
observation to satisfy two inconsistent constraints.

If `sqrt(e_k^T S_k^-1 e_k)` exceeds the configured gate, the innovation is
radially clipped before the update. This prevents one noisy transition from
being interpreted as a large actuator loss while retaining a local causal
estimator with no access to hidden fault values.

The scheduled-fault study tests a second estimator with innovation-adaptive
covariance. Before innovation clipping, it computes

```text
d_k = sqrt(e_k^T S_k^-1 e_k)
if d_k > tau_change and the prior command is sufficiently excited:
    P_k^- <- rho P_k^-,  rho >= 1
```

The innovation statistics and gain are recomputed after inflation. A startup
warmup rejects initial model transients and a cooldown prevents immediate
repeated inflation, while ordinary process-noise and forgetting updates
continue between events. This restores adaptation after an abrupt loss without
reading the configured change time or effectiveness. An
inflation event is evidence of local model surprise, not an isolated or
classified physical fault; offline plant truth is used only to calculate RMSE.

The CUSUM comparator uses normalized innovation squared per measured dynamic
state:

```text
q_k = d_k^2 / dim(e_k)
g_k = max(0, g_(k-1) + q_k - nu)
inflate P_k^- and reset g_k when g_k >= h
```

Here `nu` is the configured reference drift and `h` is the decision threshold.
The statistic accumulates moderate persistent mismatch that a one-step test can
miss. This follows the established chi-square CUSUM pattern used for innovation-
based fault monitoring, while the project-specific action is covariance
reopening rather than a standalone alarm
([de Oliveira et al., 2012](https://doi.org/10.1155/2012/740752)).

When change-triggered probing is enabled, the controller initially suppresses
active identification. After a CUSUM event for agent `i`, it resets that local
agent's excitation and information proxies,

```text
E_i <- 0,  I_i <- 0,  probe_quota_i <- 0
```

and reuses the existing constraint-aware information scheduler. Other agents'
estimators and budgets are untouched. The scheduled plant event remains hidden;
the trigger comes only from the local normalized innovation. CUSUM combined with
RLS is an established monitoring architecture, so the claim here is its causal
integration with safe physical-channel re-probing inside heterogeneous
distributed MPC, not invention of CUSUM itself
([Tran and Fowler, 2020](https://doi.org/10.3390/batteries6010001)).

The optional loss-only arm compares the mean local RLS update before requesting
new excitation. A negative update can reopen the probe budget; a positive
restoration update still receives covariance adaptation but does not reset
information or request probes. This classifier is intentionally modest: mixed-
sign simultaneous channel changes remain outside the current benchmark.

### Experimental nominal-recovery prior

Temporary restoration can leave RLS biased when an agent reaches its goal and
commands no longer excite every physical channel. The opt-in recovery comparator
first classifies the aggregate local update without using plant truth:

```text
r_k = mean(eta_hat_k^+ - eta_hat_k^-)
```

It acts only when the innovation detector fires and `r_k > epsilon`. For each
positively moving channel `j`, it applies

```text
eta_hat_(j,k) <- clip(
    eta_hat_(j,k)^+ + gamma_recovery (1 - eta_hat_(j,k)^+),
    eta_min,
    eta_max
)
```

Channels with nonpositive updates are unchanged. Nominal effectiveness `1`
means a healthy actuator, not knowledge of the hidden failure schedule.
Covariance inflation remains active so later measurements can override the
prior. The experiment gain is `0.20`; the base MPC gain remains zero, and only
the explicit recovery comparator enables it.

This is a project-specific regularization of event-adaptive diagonal RLS, not a
claim that covariance resetting or event-triggered fault accommodation is new.
Independent evidence improves average recovery tracking but fails stricter
final-bias and task-cost gates, so it remains an ablation rather than the
deployed estimator.

### Rank-gated transient recovery offset

The second recovery architecture leaves the recursive estimator state
`z_k=eta_hat_k` untouched. A loss event arms one later recovery action. On the
first aggregate-positive update, the controller requires the normalized local
sensitivity matrix to have full column rank,

```text
rank(S_k) = n_u.
```

This model-derived gate naturally rejects locally underdetermined input maps,
including the eight-thruster ROV update when only six dynamic-state residuals
are available. For positively recovering channels, it creates

```text
b_(j,k) = gamma_recovery (1 - z_(j,k)),
eta_controller,k = clip(z_k + b_k),
b_(k+1) = rho b_k,
```

while nonpositive channels receive zero offset. Raw RLS always recurses on
`z_k`, never `eta_controller,k`, so the nominal prior cannot accumulate as
estimator bias. Hysteresis consumes the arm after one recovery event, and the
configured per-agent episode budget suppresses repeated post-goal false events.
Positive events before the minimum causal dwell are rejected without consuming
the arm, allowing a later locally detected restoration to act.

For distributed collision prediction, a received peer message contains
position `p`, velocity `v`, and goal `g`. Legacy prediction uses `p(t)=p+tv`
for the full horizon. The goal-bounded model first computes

```text
d = v / ||v||
a = d^T (g-p) / ||g-p||
s_goal = max(d^T (g-p), 0)
p(t) = p + d min(||v||t, s_goal),  when a >= a_min
```

and retains constant velocity when alignment `a` is below `a_min`. Thus a peer
moving toward its communicated intent is not projected beyond it indefinitely.
The model does not reveal a future trajectory or centralize the optimization;
each agent still forms its own hard collision constraints from received data.

Optional active identification makes missing excitation explicit. For channel
`j`, the controller accumulates normalized command energy

```text
E_j(k) = sum_(i<k) (u_j(i) / u_j,max)^2
```

and requests a bounded alternating pulse while `E_j` is below its target or a
minimum direct-probe quota has not been met. The selected first-step command
is imposed inside the same nonlinear program:

```text
u_j(0) = sigma_k rho u_j,max,  sigma_k in {-1, +1}
```

The pulse is disabled near communicated agents and after repeated infeasibility.
Only the selected channel is fixed; all other channels remain optimization
variables. If the pulse-constrained NLP is infeasible, the agent immediately
re-solves the ordinary NLP. Thus a rejected information request is telemetry,
not an executed guidance fallback. A later LMPC iteration can initialize from
that same agent's prior local gain estimate and admitted control/state rollout.

The information-aware scheduler instead maintains a local linearized Fisher
information proxy over the actuator-effectiveness vector. For each measured
transition, with normalized sensitivity matrix `J_k`, it updates

```text
I_k = I_(k-1) + J_k^T J_k
Sigma_k = (Sigma_0^-1 + I_k / sigma_y^2)^-1
```

For every admissible channel pulse, it predicts `Delta I_j` and chooses the
channel with the largest fault-focused, uncertainty-weighted log-determinant
gain

```text
Delta h_j = log det(Sigma_k^-1 + Delta I_j / sigma_y^2)
            - log det(Sigma_k^-1)
j* = argmax_j Delta h_j sigma_j
                 (1 + w_f |1 - alpha_hat_j| / sigma_0)
```

subject to the same direct-probe quotas, rejection limits, domain checks, and
communicated clearance guard. The final factor prioritizes channels whose local
transition estimate already suggests a loss, without reading hidden plant
parameters. The reported diagonal of `Sigma_k` is a
linearized scheduling proxy: model mismatch and correlated finite-difference
errors mean it is not a calibrated statistical confidence interval.

Updates below a normalized command-excitation threshold are skipped. Using the
effectiveness-adjusted transition, the same agent then computes

```text
r_measured = (p_k - p_model(x_(k-1), alpha_hat_k u_(k-1))) / dt
r_hat_k = clip((1-gamma) r_hat_(k-1) + gamma r_measured)
p_pred_(j+1) = p_model_(j+1) + dt r_hat_k
```

These are bounded local gain/residual models, not learned hydrodynamic models.
They are appropriate for steady current, generalized actuator loss, and local
bias. They do not identify rapidly varying waves or coupled unmodeled velocity
dynamics.

### Dynamic-state envelope feasibility

Nominal velocity and angular-rate limits are componentwise

```text
l_i <= x_(i,k) <= u_i.
```

The diagnostic elastic formulation introduces nonnegative lower/upper slacks,

```text
l_i - s^-_(i,k) <= x_(i,k) <= u_i + s^+_(i,k),
0 <= s^+_(i,k), s^-_(i,k) <= s_max,
J_slack = w_s sum_k ||s_k||_2^2.
```

Actuator bounds, collision separation, medium membership, and true map-domain
bounds remain hard. The nominal controller sets `s_max=0`. The experimental
USV retry first solves that hard problem and exposes a separately built elastic
optimizer only inside a configured goal radius after measured yaw error exceeds
its event threshold. Maximum accepted slack and retry counts are recorded.

This mechanism restored one exposed joint-uncertainty failure (`0.268` to
`0.043` rad final USV yaw with `0.0030` maximum slack), but it did not outperform
the hard controller on the fresh five-case development ensemble. It therefore
remains a feasibility ablation rather than the default control law.

## Reduced reproducibility models

`config/harbor_reduced.yaml` preserves the earlier unicycle, scalar surge/yaw,
and world-frame damped ROV equations. It also preserves the original LMPC seed
policy and the published `205 -> 171 -> 169` completion-cost sequence. This
baseline exists for controlled model-fidelity comparisons; it is not the
default.

The marine equations follow the standard compact model summarized by
[Fossen's Marine Craft Model](https://www.fossen.biz/html/marineCraftModel.html).
A practical 3-DOF USV derivation and experimental model appears in
[Wang et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC6539673/).
