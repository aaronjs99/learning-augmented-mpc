# Research Contributions

This project is an adaptation of decentralized LMPC to simplified robotic manta
agents. The implementation intentionally differs from the vehicle paper in
dynamics, safe-set construction, terminal constraints, and collision handling.

## Implemented Contributions

1. **Manta-specific LMPC adaptation**
   - Uses a 7-state manta model with Hopf oscillator states and bounded fin
     actuation.
   - Uses position-only terminal safe-set matching by default, because full
     oscillator-state matching over-constrains the short-horizon NLP.

2. **Control-memory warm starts**
   - APF and accepted LMPC safe sets now store both states and controls.
   - IPOPT warm starts blend a nominal constant control with stored safe-set
     controls through `warm_start_control_blend`.
   - This keeps the initial guess dynamically plausible without fully copying
     APF or previous-iteration artifacts.

3. **Strict safe-set admission**
   - Only complete, collision-free rollouts are added back into the learned
     safe set.
   - Safe-but-incomplete rollouts remain visible in diagnostics but do not
     become terminal data.

4. **Benchmark sweep surface**
   - `python run.py sweep` produces compact scenario-level validity, safety,
     fallback, clearance, and first-hit cost summaries.
   - This makes scenario tuning and method comparisons reproducible.

## Current Evidence

The strongest current success case is `manta_crossover`:

```text
python run.py sweep --scenario manta_crossover --iterations 2 --max-steps 230
```

The latest curated run selects LMPC iteration 2, stays collision-free, and
improves first-hit step proxies from `84/223` to `52/181`.

A short `manta_triangle` probe remains safe but incomplete:

```text
python run.py sweep --scenario manta_triangle --iterations 1 --max-steps 160
```

This suggests that the remaining 3-agent problem is not basic safety; it is
coordination and terminal reachability under decentralized hyperplane
constraints.

## High-Value Next Experiments

1. **Priority-aware hyperplanes**
   - Give right-of-way to agents closer to their goals or with fewer escape
     options.
   - Expected benefit: fewer traffic jams in 3-agent crossing scenarios.

2. **Adaptive slack penalties**
   - Increase hyperplane/static slack penalties when observed violations or
     near-contact events occur.
   - Expected benefit: preserve feasibility early while tightening safety later.

3. **Terminal reachability filtering**
   - Select terminal safe-set samples using both time index and reachable
     position/control consistency.
   - Expected benefit: avoid terminal points that are safe but unreachable from
     the current short horizon.

4. **Scenario-class benchmark table**
   - Track APF baseline, 2-agent success, 3-agent nominal crossing, narrow gate,
     and offset crossing separately.
   - Expected benefit: turns failure modes into measurable research claims.
