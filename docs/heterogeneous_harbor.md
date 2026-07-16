# Heterogeneous Harbor Coordination

## Research Question

Can delayed, range-limited communication improve safety for independently
actuated UGV, USV, and ROV platforms without imposing a tether or any other
relative-pose constraint?

The initial controlled experiment compares two policies with identical starts,
goals, dynamics, domains, and controller gains:

- `independent`: each platform tracks only its own goal.
- `communication`: platforms exchange position, velocity, and goal messages and
  apply decentralized priority/yield coordination.

## Modular Contract

Each platform owns its state dimension, control dimension, dynamics, control
bounds, position/velocity extraction, and operating domain. Shared coordination
sees only world-frame position, velocity, and desired velocity.

- UGV: planar acceleration-controlled unicycle on its ground domain.
- USV: planar surge/yaw dynamics with drag at the water surface.
- ROV: untethered surge/heave/yaw dynamics inside an underwater volume.

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

| Policy | Goals | Swept violations | Minimum distance |
| --- | ---: | ---: | ---: |
| Independent | 3/3 | 7 | 0.172 m |
| Communication | 3/3 | 0 | 1.634 m |

The safety improvement costs task time: the yielding UGV reaches its goal at
step 74 instead of step 41. That tradeoff is intentionally visible and gives
future learning or MPC policies an objective baseline to beat.

Use `--mode communication` or `--mode independent` for one policy. Add
`--output results/tmp/harbor.json` only when a persistent JSON artifact is
useful. All experiment quantities live in `config/harbor.yaml`.

## Next Research Ablations

1. Sweep range, delay, update rate, dropout, and message TTL to map the safety
   envelope of heterogeneous coordination.
2. Replace lexical fixed priority with learned risk/time-to-go negotiation and
   compare safety, makespan, and communication load.
3. Add platform-specific static geometry and currents/slip disturbances while
   preserving the shared observation contract.
4. Use communication-conditioned safe sets, where terminal samples are tagged
   with the network state under which they were demonstrated.
