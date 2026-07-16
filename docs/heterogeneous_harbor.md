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

- UGV: dynamic skid-steer model with left/right drive-side inputs and a 3-DOF `[x, y, yaw]`
  pose goal. RobEn (Jackal) and Inspector-Gadget (Husky) use separate named
  mass, inertia, damping, speed, actuation, and footprint profiles.
- USV: underactuated body-frame surge/sway/yaw marine dynamics, port/starboard
  waterjet controls, and a 3-DOF pose goal at the water surface.
- ROV: body-frame 6-DOF marine dynamics with inertia, Coriolis, linear and
  quadratic damping, hydrostatic restoring forces, eight T200 controls through
  a full-rank 6x8 allocation, and
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

## Asymmetric Actuator-Fault Study

The fault study applies a different hidden effectiveness vector to every named
platform. RobEn and Inspector-Gadget remain separate UGVs: each has its own
physical parameters, route, left/right drive fault, and local estimate.
The controller receives no configured fault value.

```text
python run.py harbor-fault-study
```

| Controller | Step sum | Effectiveness RMSE | Fallbacks |
| --- | ---: | ---: | ---: |
| Nominal MPC | 178 | 0.235 | 13 |
| Scalar-adaptive MPC | **156** | 0.166 | 7 |
| Passive diagonal MPC | 162 | 0.0212 | 0 |
| Active diagonal MPC | 169 | **0.0100** | 0 |
| One-pass active MPC | 163 | 0.0297 | 0 |
| Information-aware MPC | 163 | **0.0160** | 0 |
| Retained passive LMPC | 160 | 0.0184 | 0 |
| Retained active-ID LMPC | 161 | 0.0189 | 0 |
| Retained information-ID LMPC | 162 | **0.0120** | 0 |

All nine rollouts are complete and swept-safe with numerical-zero collision
slack. The nominal and scalar baselines require solver fallbacks and therefore
are not admitted; all seven actuator-wise trials are fallback-free and valid.
Round-robin active MPC reduces gain RMSE by `52.6%` relative to passive
diagonal MPC, while increasing completion cost by seven steps (`4.3%`). More
importantly, information-aware MPC reduces RMSE by `46.0%` relative to the
equal-budget one-pass comparator: both execute 14 direct probes and cost 163
steps. Across repeated runs, retained information-ID LMPC lowers RMSE by
`34.9%` relative to retained passive LMPC at a two-step (`1.3%`) cost increase.

The active controllers track normalized command energy, local transition
information, and a minimum direct-probe quota for each physical actuator
channel. A small alternating pulse is imposed
as a first-step equality inside the platform's own NLP only while moving and
clear of nearby agents. Dynamics, domain, actuator, and communicated collision
constraints therefore see the requested pulse. If it is infeasible, the same
agent immediately re-solves without it; repeatedly rejected channels are
disabled. The rejected probe attempt is reported separately; a successful
ordinary re-solve remains a clean control solve and is not an execution fallback.

The diagnostic exposes the intended observability limit. Every physical channel
receives two accepted direct probes in the round-robin active MPC trial and one
in the equal-budget one-pass and information-aware trials, including all eight
BlueROV2 Heavy thrusters. The
UGV model identifies physical left/right drive-side channels, the Heron model
identifies port/starboard waterjet channels, and the BlueROV2 Heavy model
identifies all eight thruster channels. Because eight actuator gains affect six
dynamic-rate observations, simultaneous passive identification is
underdetermined; channel-isolating active probes resolve that ambiguity over
multiple transitions. Probe counts, per-agent channel order, rejection counts,
and the linearized posterior-standard-deviation trace are saved in JSON.

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
