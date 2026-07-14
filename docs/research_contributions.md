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

5. **Performance-aware APF safe-set selection**
   - Staged APF candidates are now ranked by safety and total first-hit time,
     rather than safety margin alone.
   - This produces shorter complete seeds while keeping explicit validation.

6. **Priority-aware decentralized hyperplanes**
   - Pairwise SVM margins can be asymmetric while preserving the total pairwise
   separation budget.
   - This provides a configurable right-of-way mechanism for decentralized
     traffic-jam experiments.

7. **Swept-segment safety validation**
   - Safe-set admission evaluates the minimum synchronous agent separation and
     obstacle clearance along every interval between saved states.
   - This prevents endpoint-only sampling from accepting trajectories that
     cross another agent or obstacle between controller updates.

8. **Intersample safety collocation**
   - The nonlinear MPC enforces static-obstacle and separating-hyperplane
     constraints at configurable interior points within every control interval.
   - Validation remains independent and swept, so collocation density can be
     studied against measured intersample clearance rather than assumed safe.

9. **Independent execution-time safety filter**
   - Proposed joint transitions are swept-validated before they are executed.
   - Unsafe proposals are replaced by bounded APF actions and, if necessary,
     zero-translation holds; every intervention is recorded separately from
     solver failure.

## Current Evidence

The strongest current success case is `manta_crossover`:

```text
python run.py sweep --scenario manta_crossover --iterations 2 --max-steps 230
```

The latest curated run selects LMPC iteration 2, stays swept-collision-free,
and improves first-hit step proxies from `84/223` to `49/182`. It uses no IPOPT
fallbacks and one recorded execution-time safety intervention near the inflated
obstacle boundary. The selected rollout retains `0.018` minimum swept obstacle
clearance, and the intervention is visible rather than being silently admitted
as a safe optimization result.

A short `manta_triangle` probe remains safe but incomplete:

```text
python run.py sweep --scenario manta_triangle --iterations 1 --max-steps 160
```

With performance-aware APF seed selection, the triangle APF seed changed from
`344/118/423` first-hit steps to `115/338/414`. The one-iteration LMPC probe
then produced a safe partial rollout with costs around `96/314/361+`, but it
still did not become a complete selectable LMPC result. This suggests that the
remaining 3-agent problem is not basic safety; it is coordination and terminal
reachability under decentralized hyperplane constraints.

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

4. **Waypoint terminal repair**
   - A capped APF repair phase exists behind `repair_incomplete_with_apf`, but
     current probes show it is still too conservative or too slow for triangle.
   - Expected benefit after refinement: turn safe partial LMPC routes into
     complete learned safe-set members without hiding the repair status.

5. **Scenario-class benchmark table**
   - Track APF baseline, 2-agent success, 3-agent nominal crossing, narrow gate,
     and offset crossing separately.
   - Expected benefit: turns failure modes into measurable research claims.
