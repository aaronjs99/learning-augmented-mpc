# Harbor Dynamics

The harbor optimizer and simulator currently use the same discrete equations.
This avoids model mismatch inside the experiment, but the models are intentionally
reduced order. They should not be described as hydrodynamically or mechanically
high fidelity.

## Equations used now

All models use a semi-implicit Euler step of duration `dt`: velocity is updated
first, then pose is advanced with the new velocity.

### UGV

State `q = [x, y, psi, v]` and control `u = [a, omega]`:

```text
v+   = clip(v + dt a, 0, v_max)
psi+ = wrap(psi + dt omega)
x+   = x + dt v+ cos(psi+)
y+   = y + dt v+ sin(psi+)
```

This is an acceleration-controlled unicycle. It enforces planar motion, speed,
acceleration, yaw-rate, and quay-domain bounds. It has no wheelbase, steering
angle, tire force, slip, or motor dynamics.

### USV

State `q = [x, y, psi, v]` and control `u = [T, omega]`:

```text
v+   = clip(v + dt (T - d v), 0, v_max)
psi+ = wrap(psi + dt omega)
x+   = x + dt v+ cos(psi+)
y+   = y + dt v+ sin(psi+)
```

This is a surge-and-yaw surface model with scalar linear drag. It has no sway
velocity, yaw inertia, Coriolis force, current, wind, wave, or thruster allocation.

### ROV

Pose `eta = [x, y, z, phi, theta, psi]`, world-frame velocity
`nu = [vx, vy, vz, p, q, r]`, and world-frame wrench
`tau = [Fx, Fy, Fz, K, M, N]`:

```text
nu_linear+  = nu_linear  + dt (F - d_linear nu_linear)
nu_angular+ = nu_angular + dt (T - d_angular nu_angular)
eta_pos+     = eta_pos     + dt nu_linear+
eta_angle+   = wrap(eta_angle + dt nu_angular+)
```

This gives the ROV a real 6-DOF pose goal, 12-state trajectory, six controls,
depth limits, and attitude tracking. It assumes unit mass and inertia and omits
body-frame kinematics, added mass, Coriolis/centripetal terms, buoyancy, gravity,
hydrodynamic cross-coupling, currents, and actuator allocation.

## Target platform models

### UGV target: steering bicycle

Use `q = [x, y, psi, v, delta]` and `u = [a, delta_rate]`, with wheelbase `L`:

```text
x_dot     = v cos(psi)
y_dot     = v sin(psi)
psi_dot   = (v / L) tan(delta)
v_dot     = a
delta_dot = delta_rate
```

This preserves a compact nonlinear MPC while replacing direct yaw-rate command
with physically meaningful steering position and rate constraints. Tire-force
dynamics are only warranted for higher-speed or low-friction experiments.

### USV target: underactuated 3-DOF marine craft

Use earth-fixed pose `eta = [x, y, psi]`, body velocity `nu = [u, v, r]`, and
surge/yaw input `tau = [T, 0, N]`:

```text
eta_dot = R(psi) nu
M nu_dot + C(nu) nu + D(nu) nu = tau + tau_environment
```

This adds sway, mass and yaw inertia, added mass, nonlinear coupling, calibrated
damping, and optional wind/current disturbance while retaining a 3-DOF goal.

### ROV target: 6-DOF marine rigid-body dynamics

Use earth-fixed pose `eta = [x, y, z, phi, theta, psi]`, body velocity
`nu = [u, v, w, p, q, r]`, and generalized wrench `tau`:

```text
eta_dot = J(eta) nu
M nu_dot + C(nu) nu + D(nu) nu + g(eta) = tau + tau_environment
tau = B f_thruster
```

`M` contains rigid-body inertia and added mass, `C` contains Coriolis and
centripetal terms, `D` is hydrodynamic damping, `g` contains weight/buoyancy
restoring forces, and `B` maps individual thrusters to body wrench. This is the
appropriate next ROV model, but its coefficients must be YAML-configured and
identified or sourced for a particular vehicle; inventing them would only make
the simulation look more complicated, not more accurate.

The marine equations follow the standard compact model summarized by
[Fossen's Marine Craft Model](https://www.fossen.biz/html/marineCraftModel.html).
A 3-DOF surge-sway-yaw reduction is standard for surface vessels; one practical
derivation and experimental model appears in
[Wang et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC6539673/).
