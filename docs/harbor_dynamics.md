# Harbor Dynamics

The default harbor experiment uses physically structured platform models. Each
model owns both its NumPy transition and its CasADi symbolic transition; a
regression test evaluates both from the same state and control and requires
machine-precision agreement. Angles remain continuous internally and are
wrapped only when pose errors are evaluated.

The YAML coefficients are plausible research parameters, not values identified
for a particular vehicle. The correct claim is therefore *mechanistically
structured simulation*, not hardware-validated high fidelity.

## UGV: kinematic bicycle

State `q = [x, y, psi, v]` and input `u = [a, delta]`, where `delta` is steering
angle and `L` is wheelbase:

```text
x_dot   = v cos(psi)
y_dot   = v sin(psi)
psi_dot = (v / L) tan(delta)
v_dot   = a
```

The discrete transition updates bounded forward/reverse speed first and then
advances yaw and position. Acceleration, steering angle, wheelbase, forward
speed, reverse speed, and the quay domain are independently bounded. Guidance
uses a signed-speed nonlinear pose controller, so the UGV can converge to
`[x, y, yaw]` without an unphysical ability to rotate in place.

This model omits steering actuator state, wheel slip, tire forces, suspension,
and motor dynamics. Those effects are unnecessary at the configured low speed
but would matter in high-speed or low-friction studies.

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
wind, waves, explicit added-mass matrices, and individual propeller allocation
remain future disturbances and identification targets.

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
the configured centers of gravity and buoyancy. The default vehicle is neutrally
buoyant with its center of buoyancy above its center of gravity, producing
roll/pitch restoring moments.

The ROV has a real 12-state trajectory, six bounded wrench inputs, 6-DOF pose
goal, body-frame velocity, depth limits, and attitude dynamics. It does not yet
model off-diagonal added mass, Euler-angle singularity avoidance, wave forces,
or a thruster allocation `tau = B f_thruster`. Those require a specific vehicle
geometry and identified coefficients.

## Execution-plant mismatch and residual model

For marine platforms, the robustness plant advances nominal dynamics and then
adds a hidden world-frame current `v_c` to position:

```text
p_(k+1) = p_nominal_(k+1) + dt v_c
u_applied = alpha_platform u_commanded
```

The USV uses only horizontal current; the ROV uses all three components. The
UGV is not advected. Optional actuator effectiveness `alpha_platform` is also
applied only in execution. The controller does not read either value. With its
own measured state and prior command, agent `i` computes

```text
r_measured = (p_k - p_model(x_(k-1), u_(k-1))) / dt
r_hat_k = clip((1-gamma) r_hat_(k-1) + gamma r_measured)
p_pred_(j+1) = p_model_(j+1) + dt r_hat_k
```

This is a bounded constant-residual model, not a learned hydrodynamic model.
It is appropriate for steady current and local bias, but it does not identify
control effectiveness, rapidly varying waves, or coupled velocity dynamics.

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
