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
Each uses state `q = [x, y, psi, v, r]` and input `tau = [F, N]`:

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
and [Husky A200 manual](https://www.clearpathrobotics.com/wp-content/uploads/2013/08/Husky-A200-UGV-UserManual-0.20.pdf).

This lumped model omits individual wheel speeds, longitudinal/lateral slip,
terrain-dependent traction, suspension, and motor current dynamics. Those are
the next identification layer for hardware transfer.

## USV: underactuated 3-DOF marine craft

Earth-fixed pose `eta = [x, y, psi]`, body velocity `nu = [u, v, r]`, and
underactuated input `tau = [T, 0, N]` satisfy:

```text
eta_dot = R(psi) nu
M nu_dot + C(nu) nu + D(nu) nu = tau
D(nu)nu = D_linear nu + D_quadratic |nu| nu
```

`M = diag(m_u, m_v, I_z)` includes effective surge/sway mass and yaw inertia.
The Coriolis product used by both simulator and optimizer is:

```text
C(nu)nu = [-m_v v r,
             m_u u r,
            (m_v - m_u) u v]
```

Unlike the former scalar-surge model, this state includes sway and yaw-rate
dynamics. Its nominal transition remains a calm-water model. The robustness
study applies current outside that transition as execution-plant mismatch;
the Heron profile retains the 1.7 m/s hardware limit, 40 N total thrust, and a
39 kg full-payload rigid-mass basis. A separate 0.9 m/s mission speed keeps
inspection control away from the hardware ceiling. Effective mass and damping
are documented engineering estimates. Wind, waves, off-diagonal added mass,
and individual waterjet allocation remain future identification targets. See
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

The ROV has a real 12-state trajectory, six bounded wrench inputs, 6-DOF pose
goal, body-frame velocity, depth limits, and attitude dynamics. It does not yet
model off-diagonal added mass, Euler-angle singularity avoidance, wave forces,
or an explicit thruster allocation `tau = B f_thruster`. The next fidelity step
is to identify the actual eight-thruster allocation and coefficients on the lab
vehicle. Public dimensions, mass, speed, and thrust are from the
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
instead uses a diagonal gain matrix over each platform's generalized controls:

```text
u_applied = diag(alpha_1, ..., alpha_nu) u_commanded
```

For RobEn and Inspector-Gadget, these are separate `[force, yaw moment]`
vectors; the two UGVs neither share an estimate nor share hidden fault values.
For Heron they are `[surge force, yaw moment]`, and for BlueROV2 Heavy they are
the six generalized wrench channels `[X, Y, Z, K, M, N]`.

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
The model identifies generalized force/moment effectiveness. It does not yet
identify individual RobEn/Inspector-Gadget wheel motors, Heron waterjets, or
BlueROV2 thrusters; that requires an explicit allocation matrix and sufficiently
exciting maneuvers.

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
