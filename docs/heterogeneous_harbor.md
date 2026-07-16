# Heterogeneous Harbor Coordination

## Research Question

Can delayed, range-limited communication improve safety for independently
actuated UGV, USV, and ROV platforms in a physically partitioned harbor without
imposing a tether or any other relative-pose constraint?

The controlled experiments compare three policies with identical starts, goals,
dynamics, domains, and network settings:

- `independent`: each platform tracks only its own goal.
- `communication`: platforms exchange position, velocity, and goal messages and
  apply decentralized coordination. `reciprocal` makes both platforms respond;
  `eta_priority` negotiates priority from estimated time to goal and gives the
  yielding platform the stronger maneuver.

## Modular Contract

Each platform owns its state dimension, pose-goal dimension, control dimension,
dynamics, control bounds, position/velocity extraction, and operating domain.
Shared coordination sees only world-frame position, velocity, and desired
velocity.

- UGV: dynamic skid-steer force/yaw-moment model with a 3-DOF `[x, y, yaw]`
  pose goal. RobEn (Jackal) and Inspector-Gadget (Husky) use separate named
  mass, inertia, damping, speed, actuation, and footprint profiles.
- USV: underactuated body-frame surge/sway/yaw marine dynamics, thrust/yaw-
  moment controls, and a 3-DOF pose goal at the water surface.
- ROV: body-frame 6-DOF marine dynamics with inertia, Coriolis, linear and
  quadratic damping, hydrostatic restoring forces, six wrench controls, and
  `[x, y, z, roll, pitch, yaw]` goals inside the underwater volume.

The configured shoreline separates the operating media. The UGV remains on the
quay/ground above the shoreline, the USV remains in harbor water at `z=0`, and
the ROV remains in the same water region below the surface. The default safety
case includes two UGVs passing in overlapping quay lanes, a surface vessel, and
an ROV dive/ascent route. Ground vehicles never enter water, the USV never
leaves the surface, and the ROV never leaves its submerged volume.

Communication changes information only. It is configured by range, update
interval, delay, message time-to-live, dropout probability, and random seed. No
model reads another platform's state during integration, and no constraint
binds one platform's pose to another.

## Distributed MPC and LMPC

Each platform now solves its own CasADi receding-horizon problem from local
state and received messages. Other agents enter as timestamped communicated
constant-velocity predictions, never as direct global-state reads. Plain MPC
uses terminal goal tracking. LMPC adds a convex hull of sampled positions from
the prior admitted rollout and its time-to-go values. Full orientation remains
in the objective and final 3-DOF/6-DOF goal validator.

The default deterministic benchmark is complete, swept-safe, and solver-clean:

| Controller | Step sum | Min distance | Fallbacks | Collision slack |
| --- | ---: | ---: | ---: | ---: |
| Guidance seed | 201 | 1.709 m | 0 | 0 |
| Distributed MPC (`N=12`) | **116** | 1.375 m | 0 | numerical zero |
| Distributed LMPC 1 (`N=12`, rejected) | 127 | 1.375 m | 0 | 0 |
| Distributed LMPC 2 (`N=12`, rejected) | 127 | 1.375 m | 0 | 0 |

At `N=12`, plain MPC improves completion cost by `42.3%` over guidance. Both
LMPC candidates are complete, swept-safe, and solver-clean but are rejected
because their `127` cost regresses the admitted MPC rollout. The admission gate
therefore prevents a safe but slower learned trajectory from replacing the
incumbent.

The matched horizon study exposes a horizon-dependent learning benefit. At
`N=8`, both controllers are incomplete. At `N=12`, MPC is admitted at `116`
and LMPC is rejected at `127`. At `N=15`, both complete and LMPC is admitted,
improving cost from `128` to `124` (`3.1%`). Every study rollout has zero swept
violations and zero solver fallback.

## Initial Evidence

Run the deterministic ablation without creating an output file:

```text
python run.py harbor
```

The default scenario produces:

| Policy | Goals | Violations | Min distance | Step sum |
| --- | ---: | ---: | ---: | ---: |
| Independent | 4/4 | 0 | 1.375 m | 196 |
| Reciprocal communication | 3/4 | 0 | 1.906 m | 365* |
| ETA-priority communication | 4/4 | 0 | 1.709 m | 201 |

`*` assigns `horizon + 1` to each incomplete platform. Real platform footprints
and separated quay lanes make the independent rollout safe. ETA-priority trades
five summed steps for `0.334 m` more minimum separation and a slightly lower
makespan. Reciprocal response remains safe but loses USV liveness.

Use `--mode communication` or `--mode independent` for one policy, and
`--policy reciprocal` for the prior communication baseline. Add
`--output results/tmp/harbor.json` only when a persistent JSON artifact is
useful. All experiment quantities live in `config/harbor.yaml`.

Generate an ignored comparison plot and coordinated GIF while experimenting:

```text
python run.py harbor --plot-dir results/tmp/harbor_eta
```

Generate the curated MPC/LMPC telemetry, dashboard, and GIF:

```text
python run.py harbor-lmpc
```

Generate the matched-horizon evidence:

```text
python run.py harbor-horizon-study
```

Evaluate combined model mismatch and joint local adaptation:

```text
python run.py harbor-robustness
```

The default study applies an unmodeled `[-0.10, 0.00, 0.03]` m/s current plus
hidden control effectiveness `0.92/0.88/0.88` for UGV/USV/ROV execution. USV
vertical current is discarded at the surface. Joint adaptation estimates the
scalar actuator gain from local velocity/rate response before fitting remaining
position drift. A 12-step in-tolerance hold tests station keeping. All four
controllers complete safely, but nominal MPC incurs three solver fallbacks and
residual-only MPC incurs two, so neither is admitted. Joint MPC and joint LMPC
are fallback-free and recover the hidden current and actuator gains. Joint MPC
reduces combined USV+ROV terminal error from `0.104 m` to `0.016 m` while
lowering cost from `141` to `131`. Joint LMPC lowers cost again to `128` but has
`0.081 m` combined marine error, exposing a measurable completion-speed versus
regulation-accuracy tradeoff.

Sweep delay and dropout over five deterministic seeds and generate a robustness
heatmap:

```text
python run.py harbor-sweep
```

The network sweep separates safety from liveness. Physical dynamics invalidate
the previous reduced-model claim of universal safety through `50%` dropout:
some sparse-message trials fail safety or completion. The sweep now maps that
boundary rather than assuming it away.

The two-step block guidance default matches per-step completion cost `201` with
`109` instead of `229` guidance updates. Three- and four-step blocks remain safe
and complete but regress cost to `225` and `232`. Block size remains an explicit
closed-loop design variable rather than a pure compute optimization.

## Next Research Ablations

1. Sweep range, delay, update rate, dropout, and message TTL to map the safety
   envelope across additional geometries and disturbances.
2. Replace analytic time-to-go negotiation with learned risk/time-to-go and
   compare safety, makespan, and communication load.
3. Add platform-specific static geometry and currents/slip disturbances while
   preserving the shared observation contract.
4. Tag admitted LMPC safe trajectories with the network and model parameters
   under which they were demonstrated, then test transfer across conditions.
