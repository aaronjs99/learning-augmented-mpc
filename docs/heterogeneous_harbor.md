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

Each platform owns its state dimension, control dimension, dynamics, control
bounds, position/velocity extraction, and operating domain. Shared coordination
sees only world-frame position, velocity, and desired velocity.

- UGV: planar acceleration-controlled unicycle on its ground domain.
- USV: planar surge/yaw dynamics with drag at the water surface.
- ROV: untethered surge/heave/yaw dynamics inside an underwater volume.

The configured shoreline separates the operating media. The UGV remains on the
quay/ground above the shoreline, the USV remains in harbor water at `z=0`, and
the ROV remains in the same water region below the surface. The default safety
case is a near-surface USV/ROV crossing; the UGV communicates from the quay but
does not drive through water.

Communication changes information only. It is configured by range, update
interval, delay, message time-to-live, dropout probability, and random seed. No
model reads another platform's state during integration, and no constraint
binds one platform's pose to another.

## Initial Evidence

Run the deterministic ablation without creating an output file:

```text
python run.py harbor
```

The default scenario produces:

| Policy | Goals | Violations | Min distance | Step sum |
| --- | ---: | ---: | ---: | ---: |
| Independent | 3/3 | 6 | 0.611 m | 115 |
| Reciprocal communication | 3/3 | 0 | 1.662 m | 143 |
| ETA-priority communication | 3/3 | 0 | 1.407 m | 128 |

Communication removes all swept violations. ETA-priority negotiation then
reduces summed completion steps by `10.5%` relative to reciprocal coordination,
while retaining `0.557 m` clearance beyond the `0.85 m` USV/ROV requirement.
The independent controller is faster but unsafe, so it is not an admissible
task-time baseline.

Use `--mode communication` or `--mode independent` for one policy, and
`--policy reciprocal` for the prior communication baseline. Add
`--output results/tmp/harbor.json` only when a persistent JSON artifact is
useful. All experiment quantities live in `config/harbor.yaml`.

Generate an ignored comparison plot and coordinated GIF while experimenting:

```text
python run.py harbor --plot-dir results/tmp/harbor_eta
```

Sweep delay and dropout over five deterministic seeds and generate a robustness
heatmap:

```text
python run.py harbor-sweep
```

The default 150-trial sweep separates safety from liveness. All trials remain
collision-free through four delay steps across the tested `0-50%` dropout
range, but final completion rate falls as messages become sparse or stale. At
six and eight delay steps, safety also degrades under high dropout. Constant-
velocity delayed-message prediction remains configurable but is off by default:
in this turning USV/ROV geometry it can over-predict motion and is not a robust
improvement over stale-state anticipation.

The two-step block guidance default retains the same `128` summed first-hit
cost and zero violations as per-step guidance while reducing guidance updates
from `131` to `68` (`48.1%`). Four-step and longer blocks can lose final
completion or safety, so block size remains an explicit experimental variable.

## Next Research Ablations

1. Sweep range, delay, update rate, dropout, and message TTL to map the safety
   envelope across additional geometries and disturbances.
2. Replace analytic time-to-go negotiation with learned risk/time-to-go and
   compare safety, makespan, and communication load.
3. Add platform-specific static geometry and currents/slip disturbances while
   preserving the shared observation contract.
4. Use communication-conditioned safe sets, where terminal samples are tagged
   with the network state under which they were demonstrated.
