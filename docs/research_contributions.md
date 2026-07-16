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
   - `python run.py sweep` produces compact scenario-level runtime, selected
     and latest solver status, validity, safety, fallback, clearance, and
     first-hit cost summaries.
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

10. **Constraint-slack telemetry**
    - Every successful MPC solve exposes its maximum static-obstacle,
      pairwise-hyperplane, and absolute terminal slack instead of reducing
      solver behavior to `ok`.
    - Reports preserve maxima and nonzero-use counts per learning iteration,
      globally and per agent, enabling safety interventions and performance
      changes to be tied back to the optimizer's actual relaxation usage.

11. **Swept-safe compact APF staging**
    - Order-conditioned APF routes are delay-scheduled for concurrent execution
      using pairwise swept-distance tables and a minimum-makespan search.
    - Delayed controls are replayed through the nonlinear seven-state dynamics;
      the scheduler never splices position-only paths into the learned safe set.
    - Every compact candidate competes against the original sequential seed and
      must pass the same independent admission validator, so infeasible overlap
      automatically falls back to the established behavior.

12. **Untethered heterogeneous harbor coordination**
    - Two 3-DOF-pose UGVs, one 3-DOF-pose USV, and a 6-DOF-pose, 12-state ROV
      expose a common world-velocity guidance contract while retaining distinct
      controls, dynamics, goals, and operating domains.
    - Range, rate, delay, message lifetime, dropout, and random seed are YAML
      configuration, and communication changes information without introducing
      any physical relative-pose constraint.
    - In the shoreline-constrained harbor, coordination removes four swept UGV
      violations and raises minimum 3D separation from `0.650 m` to `1.568 m`
      while all four platforms satisfy position and attitude tolerances.
    - Reciprocal response is safe but leaves two platforms incomplete;
      ETA-priority is the tested policy that restores both safety and liveness.
    - A 150-trial seeded delay/dropout sweep remains safe throughout the tested
      grid but exposes a separate final-completion boundary. This is scoped
      empirical evidence, not a safety guarantee.
    - This package is currently a distributed guidance baseline. Successful
      rollout samples are visualized but are not yet used by an LMPC terminal
      set or learned cost-to-go.

13. **Configurable block guidance execution**
    - The useful block-replanning concept from the legacy `distmpc` prototype is
      retained as a platform-neutral guidance update interval with zero-order
      control hold and explicit update-count telemetry.
    - Unlike the legacy centralized GEKKO script, the experiment remains
      heterogeneous, communication-aware, swept-validated, and YAML-backed.
    - A two-step block cuts guidance updates from `218` to `119` (`45.4%`)
      while preserving completion, zero violations, and the same `205` step
      sum; four-step guidance loses completion.

## Current Evidence

The strongest current success case is `manta_crossover`:

```text
python run.py sweep --scenario manta_crossover --iterations 2 --max-steps 230
```

The latest verified run selects LMPC iteration 2, stays swept-collision-free,
and improves first-hit step proxies from `84/223` to `49/181`. It uses no IPOPT
fallbacks and one recorded execution-time safety intervention near the inflated
obstacle boundary. The selected rollout retains `0.018` minimum swept obstacle
clearance, and the intervention is visible rather than being silently admitted
as a safe optimization result. Slack telemetry reports zero static and
hyperplane relaxation throughout that selected iteration. Terminal slack peaks
at `0.00574` for agent 0 and `0.0783` for agent 1, localizing the remaining
terminal mismatch instead of conflating it with collision relaxation. Runtime
is `351` seconds.

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

An instrumented 60-step triangle probe isolates the active relaxation:

```text
python run.py sweep --scenario manta_triangle --iterations 1 --max-steps 60
```

All 180 agent-solves are IPOPT-clean with zero static and hyperplane slack and
no safety-filter intervention. Absolute terminal slack is nonzero on all 180
solves, with a maximum of `0.114` against the configured bound of `4.0`. The
terminal relaxation is therefore continuously active but not saturated; simply
raising its bound or retuning collision penalties is not supported by this
evidence.

Compact APF staging materially improves the triangle initialization before any
LMPC solve. The selected safe-set makespan falls from `414` first-hit steps for
the slowest agent in the prior sequential seed to `199` steps, with per-agent
first hits of `199/189/147`. Independent swept validation reports `0.639 m`
minimum pairwise separation against the configured `0.600 m` requirement and
`0.932 m` minimum clearance outside the inflated obstacle. The complete
seven-state/control sequence is also replay-checked one transition at a time in
the test suite. The two-agent crossover remains valid and shortens to `155`
steps, demonstrating that compact staging is not triangle-specific.

The compact initializer does not, by itself, solve the triangle LMPC problem.
In a 60-step controlled probe, the learned rollout remains swept-safe and
solver-clean, reaches agent 2 at step `55`, and uses no execution-time safety
interventions. It also exposes `0.0303` maximum hyperplane slack and `0.398`
maximum terminal slack, both larger than under strictly sequential staging. At
200 steps, agents 0 and 2 improve from APF first-hit costs `117/75` to `99/55`,
but agent 1 remains incomplete and 66 fallback actions occur after its clean
solve prefix. Compact staging is therefore retained for its objective APF
makespan improvement, while complete three-agent LMPC remains an open result.

A pruned discrete-terminal experiment was also rejected. Fixing the terminal
to the nominal sampled point took `120.6` seconds, slightly worsened agent 0's
60-step goal error from `4.114` to `4.257`, and left maximum terminal slack at
`0.114`. Solving nominal and future candidates took `229.5` seconds, recovered
the original goal error, and increased maximum terminal slack to `0.190`. The
convex-hull implementation remains the default because discrete enumeration did
not buy reachability or relaxation improvement commensurate with its runtime.

Increasing the convex-hull terminal slack penalty from `1000` to `10000` was
more effective. In a 20-step triangle A/B test, waiting-agent displacement fell
from `0.159/0.113` m to `0.0166/0.0119` m and maximum terminal slack fell by
roughly one order of magnitude, while all solves remained clean and runtime was
unchanged. A 60-step crossover A/B similarly reduced maximum terminal slack
from `0.0904` to `0.00963`, preserved the active agent's progress, and kept the
staged waiting agent near its start.

## High-Value Next Experiments

1. **Adaptive per-agent terminal regularization**
   - Increase terminal weight for agents expected to hold and relax it only for
     agents whose measured terminal mismatch grows during active motion.
   - Expected benefit: retain staged coordination without over-regularizing the
     agent currently earning task-time improvement.

2. **Control-rate constraints**
   - Add configurable fin-command slew limits and measure completion time,
     fallback rate, and control effort.
   - Expected benefit: suppress abrupt optimizer commands that are feasible in
     the model but difficult for physical manta actuators to track.

3. **Waypoint terminal repair**
   - A capped APF repair phase exists behind `repair_incomplete_with_apf`, but
     current probes show it is still too conservative or too slow for triangle.
   - Expected benefit after refinement: turn safe partial LMPC routes into
     complete learned safe-set members without hiding the repair status.

4. **Scenario-class benchmark table**
   - Track APF baseline, 2-agent success, 3-agent nominal crossing, narrow gate,
     and offset crossing separately.
   - Expected benefit: turns failure modes into measurable research claims.
