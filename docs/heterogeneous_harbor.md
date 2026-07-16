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

- UGV: planar acceleration-controlled unicycle with a 3-DOF `[x, y, yaw]`
  pose goal on its ground domain.
- USV: planar surge/yaw dynamics with drag and a 3-DOF `[x, y, yaw]` pose
  goal at the water surface.
- ROV: untethered damped 6-DOF motion with a 12-state pose/velocity model,
  six wrench controls, and `[x, y, z, roll, pitch, yaw]` goals inside the
  underwater volume.

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
| Guidance seed | 205 | 1.568 m | 0 | 0 |
| Distributed MPC | 171 | 0.874 m | 0 | 0 |
| Distributed LMPC 1 | 171 | 1.078 m | 0 | 0 |
| Distributed LMPC 2 | 169 | 0.921 m | 0 | 0 |

Plain MPC reduces completion cost by `16.6%` versus guidance. The second LMPC
iteration reaches `17.6%`, with terminal slack falling from `0.258` in LMPC 1
to `0.099` in LMPC 2. This is controlled evidence for the configured scenario,
not a general guarantee. The simulator and optimizer use matching reduced-order
equations; `harbor_dynamics.md` documents their limitations and the
higher-fidelity migration.

## Initial Evidence

Run the deterministic ablation without creating an output file:

```text
python run.py harbor
```

The default scenario produces:

| Policy | Goals | Violations | Min distance | Step sum |
| --- | ---: | ---: | ---: | ---: |
| Independent | 4/4 | 4 | 0.650 m | 180 |
| Reciprocal communication | 2/4 | 0 | 1.520 m | 462* |
| ETA-priority communication | 4/4 | 0 | 1.568 m | 205 |

`*` assigns `horizon + 1` to each incomplete platform. Communication removes all
swept violations. Reciprocal response remains safe but loses liveness;
ETA-priority is the only tested policy that is both complete and safe. The
independent controller is faster but violates the two UGV safety envelopes, so
it is not an admissible task-time baseline.

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

Sweep delay and dropout over five deterministic seeds and generate a robustness
heatmap:

```text
python run.py harbor-sweep
```

The default 150-trial sweep separates safety from liveness. All tested trials
remain collision-free, including eight delay steps and `50%` dropout, but final
completion falls under several delayed or sparse-message conditions. This is
evidence for safety robustness in the configured scenario, not a general
guarantee.

The two-step block guidance default retains zero violations, completion, and
the same `205` step sum as per-step guidance while reducing guidance updates
from `218` to `119` (`45.4%`). Four-step guidance loses final completion, so
block size remains an explicit experimental variable.

## Next Research Ablations

1. Sweep range, delay, update rate, dropout, and message TTL to map the safety
   envelope across additional geometries and disturbances.
2. Replace analytic time-to-go negotiation with learned risk/time-to-go and
   compare safety, makespan, and communication load.
3. Add platform-specific static geometry and currents/slip disturbances while
   preserving the shared observation contract.
4. Tag admitted LMPC safe trajectories with the network and model parameters
   under which they were demonstrated, then test transfer across conditions.
