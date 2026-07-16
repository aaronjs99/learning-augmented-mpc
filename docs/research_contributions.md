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
    - With platform-scale separated quay lanes, both independent and ETA-
      priority runs are swept-safe. Coordination raises minimum 3D separation
      from `1.375 m` to `1.709 m` while reducing makespan from `54` to `53`.
    - Reciprocal response is safe but leaves the surface vessel incomplete;
      ETA-priority is the tested policy that restores both safety and liveness.
    - Delay/dropout sweeps now expose both safety and completion boundaries
      under inertial dynamics; the former universal reduced-model safety claim
      is intentionally retired.
    - Per-agent distributed MPC and LMPC now use local state plus received
      messages. A prior clean rollout supplies a position-safe terminal hull
      and learned time-to-go; full platform pose remains part of task success.
    - At `N=12`, MPC reduces completion cost from guidance `201` to `116`;
      complete and clean LMPC candidates at `127` are correctly rejected as
      regressions. At `N=15`, LMPC is admitted and improves MPC from `128` to
      `124`, demonstrating a horizon-dependent rather than universal benefit.
    - Numerical and CasADi transitions for dynamic skid-steer, 3-DOF marine,
      and 6-DOF
      marine models are regression-checked for machine-precision agreement.
    - Named profiles distinguish SRI Lab's Jackal-based RobEn and Husky-based
      Inspector-Gadget, plus a full-payload Heron and BlueROV2 Heavy.

13. **Configurable block guidance execution**
    - The useful block-replanning concept from the legacy `distmpc` prototype is
      retained as a platform-neutral guidance update interval with zero-order
      control hold and explicit update-count telemetry.
    - Unlike the legacy centralized GEKKO script, the experiment remains
      heterogeneous, communication-aware, swept-validated, and YAML-backed.
    - The two-step default matches per-step completion cost `201` with `109`
      rather than `229` updates. Three- and four-step blocks stay safe and
      complete but regress task cost, making update interval a control variable.

14. **Constraint-aware active fault identification inside distributed LMPC**
    - Each named platform can carry a hidden diagonal effectiveness vector over
      its physical actuator channels; RobEn and Inspector-Gadget retain independent
      UGV models, routes, left/right drive-side faults, and estimates.
    - A causal finite-difference sensitivity matrix and bounded least-squares
      update identify excited channels from only local state transition and
      prior command data. No controller reads the configured plant fault.
    - Energy and Fisher-information schedulers request alternating channel
      pulses only when a channel lacks excitation, direct calibration, or
      linearized actuator information.
      The selected pulse is a first-step NLP equality, so normal dynamics,
      domains, actuator limits, and communicated collision constraints remain
      active. Infeasible probes trigger a nominal re-solve and bounded channel
      rejection rather than a guidance fallback.
    - Round-robin active MPC lowers gain RMSE from `0.0212` to `0.0100`
      (`52.6%`) at a `162 -> 169` completion-cost tradeoff. At an equal budget
      of 14 direct probes, information-aware ordering lowers RMSE from `0.0297`
      to `0.0160` (`46.0%`) with identical completion cost `163`.
    - Retaining each run's own local model and safe rollout, information-ID LMPC
      lowers RMSE from `0.0184` to `0.0120` (`34.9%`) at a `160 -> 162` cost.
    - All nine matched trials are complete and swept-safe with numerical-zero
      collision slack. The seven actuator-wise trials are fallback-free; nominal
      and scalar baselines are not admitted because they require fallbacks.
      The claim is improved actuator identification with an explicit task-cost
      tradeoff, not universal dominance.
    - Active fault-diagnosis input design is established literature. The
      defensible contribution is its local constraint-aware integration and
      retained-model ablation in this untethered heterogeneous distributed-LMPC
      setting.

15. **Actuator-resolved heterogeneous control allocation**
    - RobEn/Jackal and Inspector-Gadget/Husky retain independent body parameters,
      effective tracks, drivetrain metadata, left/right commands, and fault states.
    - Heron maps port/starboard waterjet thrust into surge and yaw while preserving
      underactuated sway dynamics.
    - BlueROV2 Heavy maps eight bounded T200 channels through a full-rank 6x8
      allocation into its 12-state 6-DOF marine model. The overactuated 8-to-6
      contract creates a genuine passive-identification ambiguity that the active
      channel probes measurably reduce.

16. **Equal-budget information-aware probe scheduling**
    - Each agent accumulates its own normalized finite-difference Fisher
      information matrix and ranks safe direct probes by expected log-determinant
      gain weighted by current channel uncertainty.
    - The saved telemetry includes the physical channel sequence, accepted and
      rejected counts, and posterior-standard-deviation history, so scheduling
      decisions can be audited independently of final RMSE.
    - The posterior is explicitly treated as a linearized ranking proxy rather
      than a calibrated confidence interval. The defensible result is the
      matched 14-probe improvement over one-pass scheduling, not a universal
      statistical guarantee.

## Current Evidence

The nominal horizon study exposes where learning helps and where it does not:

```text
python run.py harbor-horizon-study
```

At `N=8`, both controllers are incomplete. At `N=12`, MPC is admitted at `116`
and LMPC is rejected at `127`. At `N=15`, LMPC is admitted and improves MPC
from `128` to `124` (`3.1%`). All reported runs are swept-safe and solver-clean.

### Joint local adaptation under model mismatch

The robustness experiment separates the nominal planning model from the
execution plant. A configurable current and per-platform actuator
effectiveness act only during execution. Each distributed controller first
fits bounded scalar control effectiveness from its velocity/rate response,
then fits a bounded world-frame position residual after accounting for that
gain. Both estimates use only prior local state, prior command, and current
onboard state. They enter that agent's horizon model while the true disturbance
remains hidden.

```text
python run.py harbor-robustness
```

With the configured opposing current, `0.92/0.88/0.88` hidden effectiveness,
and a 12-step goal hold, all controllers complete safely. Nominal and residual-
only MPC incur three and two fallbacks, respectively, and fail the validity
gate. Joint MPC and joint LMPC are fallback-free and recover the hidden gains
and current. Joint MPC reduces combined marine terminal error about `85%`
(`0.104 m` to `0.016 m`) and improves cost from `141` to `131`. Joint LMPC is
faster again at `128`, though its `0.081 m` combined marine error is worse than
joint MPC. This is an objective adaptation and speed/regulation tradeoff, not a
universal superiority claim.
Cross-current station keeping at a fixed USV yaw can be physically infeasible
for the underactuated no-sway-thruster model and is intentionally not presented
as a controller benchmark.

#### Literature positioning

The estimator split itself is not claimed as novel. Disturbance-observer MPC
has been demonstrated for underwater vehicle-manipulator systems
([IFAC-PapersOnLine, 2021](https://doi.org/10.1016/j.ifacol.2021.10.115));
distributed adaptive fault-tolerant control has addressed actuator
loss-of-effectiveness and disturbances in multi-AUV systems
([Ocean Engineering, 2022](https://doi.org/10.1016/j.oceaneng.2022.112924));
and separate disturbance/fault observers have been used for ships
([Ocean Engineering, 2023](https://doi.org/10.1016/j.oceaneng.2023.114662)).
Recent work also covers fault-tolerant networked heterogeneous marine surface
vessels ([Ocean Engineering, 2024](https://doi.org/10.1016/j.oceaneng.2024.119370))
and adaptive error-compensation NMPC for heterogeneous USV-AUV systems
([Ocean Engineering, 2026](https://doi.org/10.1016/j.oceaneng.2026.125938)).
Input design for active fault diagnosis is itself a mature field
([Annual Reviews in Control, 2019](https://doi.org/10.1016/j.arcontrol.2019.03.002)),
including integrated input-set design for multiplicative actuator faults
([IEEE Control Systems Letters, 2023](https://doi.org/10.1109/LCSYS.2023.3273912))
and minimally invasive state-constrained diagnosis for nonlinear systems
([Control Engineering Practice, 2024](https://doi.org/10.1016/j.conengprac.2024.106118)).

The defensible project contribution is therefore the integration and tested
ablation: local separated current/effectiveness identification plus bounded
constraint-aware excitation inside safe-set distributed LMPC for untethered
mixed-domain UGV/USV/ROV agents with different 3-DOF/6-DOF contracts, hard
communicated collision constraints, sustained-goal evaluation, and strict
rejection of fallback-contaminated data. The residual-only failure, clean joint-
adaptive result, and matched passive/active fault study support this architecture
in the configured benchmark. Broader novelty still requires comparison against
observer-based and optimal-input-design baselines, noisy Monte Carlo trials,
and hardware or higher-fidelity validation.

The strongest manta-specific case remains `manta_crossover`:

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
