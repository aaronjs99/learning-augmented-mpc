# Research Contributions

## Scope boundary

This repository is a simulator-first research benchmark. Results describe
deterministic software experiments with modeled vehicle dynamics, sensors,
communications, and disturbances; they are not hardware-validation claims.
Any physical deployment requires separate system identification, sensor timing,
state-estimation, safety review, and field trials.

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
    - Round-robin active MPC lowers gain RMSE from `0.0137` to `0.0100`
      (`26.6%`) at a `159 -> 166` completion-cost tradeoff. At an equal budget
      of 14 direct probes, information-aware ordering lowers RMSE from `0.0297`
      to `0.0218` (`26.8%`) with identical completion cost `160`.
    - Retained LMPC freezes the preceding local gain estimate and therefore
      preserves each source RMSE exactly. Its equal cost `168` across retained
      variants does not support a task-time learning-improvement claim here.
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
      gain, current channel uncertainty, and locally estimated departure from
      nominal effectiveness. Hidden plant values never enter the score.
    - The saved telemetry includes the physical channel sequence, accepted and
      rejected counts, and posterior-standard-deviation history, so scheduling
      decisions can be audited independently of final RMSE.
    - A five-case per-channel Latin-hypercube ensemble spans effectiveness
      `[0.55, 0.98]`. Information-aware probing wins all five equal-budget
      pairs, reducing paired mean relative RMSE by `37.96%` with `+0.2` mean
      sustained-completion steps. Its paired absolute reduction is `0.0201`
      with case-bootstrap 95% interval `[0.0080, 0.0360]`.
    - Both active policies are complete, swept-safe, and solver-clean in all
      five cases. Passive diagonal MPC is complete and safe but solver-valid in
      only three. The five-case result is a controlled generalization check,
      not a hardware or population-level guarantee.
    - The posterior is explicitly treated as a linearized ranking proxy rather
      than a calibrated confidence interval.

17. **Line-of-sight terminal recovery for underactuated harbor agents**
    - Planar UGV/USV MPC tracks the velocity line-of-sight yaw while moving and
      restores final requested yaw only for station keeping.
    - This removes a zero-control local solution that left Heron at `0.315 m`
      against a `0.300 m` goal tolerance under a severe asymmetric waterjet
      fault. After the correction, all ten active ensemble rollouts complete
      and remain swept-safe without fallback.

18. **Noise-robust local recursive actuator identification**
    - Seeded platform-specific observation channels separate controller-visible
      UGV/USV/ROV measurements from execution truth while preserving each
      platform's angle, domain, and dynamic-state contracts.
    - A normalized covariance-form recursive estimator uses forgetting,
      process uncertainty, Mahalanobis innovation gating, and a Joseph covariance
      update. Hidden plant effectiveness never enters the controller.
    - Across the same five stratified fault draws, recursive passive estimation
      wins all five comparisons and lowers mean RMSE from `0.1529` to `0.0392`
      (`72.42%` paired relative reduction; absolute bootstrap 95% interval
      `[0.0779, 0.1496]`). All rollouts are complete and collision-safe.
    - With the recursive-probe gate corrected, one-pass and information-aware
      probing reach `0.0748` and `0.0510`, respectively, while passive recursive
      RLS remains best at `0.0392`. Information-aware ordering beats one-pass in
      only two of five cases; its paired interval `[-0.0055, 0.0532]` crosses
      zero, although it lowers mean completion cost by `6.8` steps.
    - Fourteen of 15 recursive-controller rollouts are solver-clean; one-pass
      probing retains one fallback. The defensible contribution is robust local
      identification under heterogeneous noisy observations, not universal
      active-scheduling dominance or hardware validation.

19. **Intent-bounded communicated obstacle prediction**
    - Constant-velocity peer extrapolation is retained as an explicit ablation.
      The new predictor caps along-track motion at communicated intent only when
      current velocity is sufficiently aligned; unaligned motion remains
      constant velocity.
    - Each optimizer remains decentralized and keeps hard pairwise constraints.
      No future peer trajectory, centralized solve, or physical tether is added.
    - Across five matched noisy hidden-fault cases, goal-bounded prediction
      reduces recoverable fallbacks from `7` to `0`, improves fallback-free case
      rate from `80%` to `100%`, and changes mean completion cost by `-0.6`
      steps. All pairs complete and remain collision-safe.
    - Estimation RMSE is nearly unchanged (`0.03846 -> 0.03934`), isolating the
      contribution as reduced false collision infeasibility rather than better
      fault identification.

20. **Event-adaptive RLS and estimator-controller coupling under temporary faults**
    - YAML schedules change each platform's hidden per-channel plant
      effectiveness during execution. The local controller receives only noisy
      state transitions and prior commands; scheduled times and magnitudes are
      retained solely for offline scoring.
    - A one-step normalized innovation threshold and a chi-square CUSUM are
      explicit covariance-inflation comparators. Both recompute the recursive
      gain after model surprise; cooldown bounds repeated adaptation.
    - CUSUM-triggered probing keeps information requests dormant until a change
      statistic fires, then clears only that agent's excitation/information
      budget and requests one constraint-aware probe per physical channel.
    - Across three matched observation seeds, passive CUSUM wins all pairs and
      reduces degraded-interval RMSE from `0.1695` to `0.0979` (`42.27%`). The
      threshold comparator reaches `0.1006` (`40.63%`). Both reduce mean
      completion cost from `154.3` to `149.3`.
    - Triggered probing reaches `0.1533` recovery RMSE versus `0.1736` fixed
      (`11.70%` reduction) and lowers final RMSE by `9.61%`, at mean completion
      cost `150.3`. It executes 16 physical-channel probes per run.
    - All 12 initial rollouts complete, remain collision-safe, use numerical-zero
      collision slack, and require no solver fallback. Event recall remains
      only `50-58%` with `2.0-2.7` unmatched inflations per run, so this is an
      adaptive-tracking and re-identification result, not certified fault
      detection, isolation, or hardware validation.
    - A later stratified development ensemble varies all 14 actuator channels,
      per-agent onset/duration, and observation noise. Detector warmup,
      refractory timing, and loss-only probe arming were developed there, so it
      is not presented as untouched evidence.
    - On a separate five-case holdout, innovation-threshold RLS wins all five
      degraded intervals, reduces mean RMSE `0.18152 -> 0.11047` (`38.50%`),
      reduces completion cost by `13.2` steps, and remains solver-clean. The
      paired absolute-reduction bootstrap interval is `[0.05805, 0.09210]`.
    - CUSUM improves holdout tracking slightly further (`40.42%`) but causes 22
      recoverable solver fallbacks in one case and raises task cost. Triggered
      probing reaches `43.77%` tracking reduction but shares those fallbacks and
      adds cost. This counterexample shows that estimator accuracy alone does
      not establish closed-loop optimizer robustness.
    - The failure is reproducibly confined to the Clearpath Heron USV on steps
      `35-56`, with IPOPT status `Infeasible_Problem_Detected`, spanning the
      hidden restoration at step `40`. Both CUSUM variants share the interval;
      threshold RLS remains solver-clean. The artifact stores per-agent steps
      and status counts for direct audit.
    - Threshold RLS was selected only after holdout inspection, so a third,
      frozen ten-case confirmation ensemble compares it solely with fixed RLS
      under YAML-predeclared closed-loop gates.
    - It passes every confirmation gate: `10/10` degraded-interval wins, RMSE
      `0.17460 -> 0.10147` (`40.93%` paired relative reduction), absolute paired
      bootstrap interval `[0.05717, 0.09093]`, completion cost `147.8 -> 144.3`,
      and `100%` completion, safety, and fallback-free rates across 20 rollouts.
    - Recovery improvement remains unsupported because its interval crosses
      zero, and final RMSE worsens. Event recall is `73.75%` with `3.2` unmatched
      inflations per run. Thus the confirmed scope is degradation tracking and
      closed-loop task performance in simulation, not certified diagnosis,
      recovery convergence, population-level generalization, or hardware proof.

21. **Direction-gated channel-selective recovery regularization (ablation)**
    - A local detector event is classified by the mean diagonal-RLS update. Only
      aggregate positive events enable a nominal-health prior, and only channels
      with positive updates are moved. The controller remains blind to hidden
      fault schedules and values.
    - Five development cases show `5/5` recovery wins and `7.13%` lower mean
      post-recovery RMSE with a positive paired interval while preserving
      completion, safety, and fallback-free operation.
    - A separately frozen ten-case confirmation shows `9/10` recovery wins,
      `3.23%` mean recovery improvement, and paired interval
      `[0.00162, 0.00613]`. Degraded-interval RMSE also improves by `0.00123`.
    - The predeclared overall gate fails because final estimate RMSE worsens by
      `0.00411` and mean task cost increases by `0.4` steps. The result supports
      a repeatable short-horizon recovery effect, not promotion to the default
      controller or a broad novelty claim.

22. **Estimator-decoupled, rank-gated transient recovery accommodation**
    - Raw event-adaptive RLS remains the estimator state. A separate decaying
      controller offset acts only on positively recovering channels, so a
      nominal-health prior cannot recursively accumulate as estimator bias.
    - A model-derived full-column-rank gate rejects locally underdetermined
      recovery updates, including the eight-input/six-residual ROV map without
      platform-name special cases. Loss/recovery hysteresis adds a causal dwell
      and one-episode budget while remaining blind to hidden schedules.
    - Five development cases give `4/5` wins, `1.08%` lower recovery RMSE, and
      paired interval `[0.00060, 0.00257]` with no fault, final, cost, safety, or
      solver regression.
    - A separately frozen ten-case confirmation passes every predeclared gate:
      `10/10` recovery wins, `0.85%` mean relative improvement, paired interval
      `[0.000826, 0.001421]`, numerical-zero fault-interval delta, slightly
      better final RMSE, unchanged task cost, and `100%` completion, safety,
      and fallback-free rates.
    - The contribution is the heterogeneous estimator-controller architecture
      and independently gated evidence, not invention of RLS, rank tests,
      temporal hysteresis, or nominal priors in isolation.

23. **Hard-domain MPC with a separately audited interior warning band**
    - Land, water, and depth bounds remain hard for all predicted states. A
      bounded quadratic slack applies only to an interior warning margin and is
      reported independently from collision slack.
    - This resolves an underactuated Heron recursive-feasibility failure caused
      by outward momentum at the shoreline. The exact diagnostic case changes
      from 15 USV solver fallbacks to zero without allowing a predicted state
      outside the water domain.
    - The final ten-case confirmation is fully safe and fallback-free; maximum
      adaptive warning-band use is `7.07e-6 m`. This is a robust integration and
      auditability contribution, not a claim that soft constraint bands are new.

24. **Observability-aware range SLAM inside fault-adaptive distributed MPC**
    - A common simulator boundary now separates plant truth from controller
      belief. UGV/USV positions use 2D filters and BlueROV2 uses a 3D filter;
      range bias, Gaussian noise, range limit, update rate, dropout, odometry
      bias, and odometry noise are independently configurable.
    - The same estimator supports known-map localization and joint unknown-
      beacon EKF-SLAM. A sliding Jacobian window reports numerical rank,
      smallest singular value, and condition number. Unknown landmarks cannot
      influence navigation until fixed-anchor geometry spans the position
      subspace; high-uncertainty landmark updates preserve the pose block.
    - A third non-collinear fixed harbor beacon corrects the globally weak
      two-anchor geometry. In the isolated diagnostic, BlueROV2 RMSE is
      `0.579 m` for dead reckoning, `0.076 m` for the known map, and `0.105 m`
      for joint SLAM.
    - The joint study composes range SLAM with local diagonal actuator RLS,
      actuator-independent current estimation, communication, and distributed
      nonlinear MPC. Across three fresh cases, dead reckoning completes `0/3`,
      known-map ranges `2/3`, and joint SLAM `3/3`.
    - A separate belief-feasibility comparator retries only after a hard MPC
      failure and relaxes only internal dynamic-state envelopes. It preserves
      hard collision, actuator, medium, depth, and world-domain constraints.
      Joint SLAM plus retry is complete and collision-free in `3/3`, has zero
      fallbacks, uses one retry total, and reaches `0.00148` maximum slack. Its
      mean completion cost is `158` versus `157` for direct position sensing.
      These are development results; no novelty or generalization claim is made
      before an untouched confirmation ensemble and literature comparison.

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

### Joint uncertainty and rejected controller extensions

The temporary-fault generator now independently samples hidden marine current
without changing the established fault draws. Raw residual estimates remain
available for diagnostics while control-facing residuals can be restricted by
platform kind and projected into an underactuated USV's surge subspace. An
untouched ten-case joint-current/fault confirmation improved recovery RMSE in
all ten pairs but reached only `70%` completion and therefore failed its frozen
task gates.

That failure motivated three controlled extensions. Always-on residual
projection degraded the five-case development aggregate. Always-on elastic
velocity/rate envelopes caused one completion regression. A feedback-gated
elastic retry rescued the exposed USV case with a tiny measured relaxation, but
fresh development had `0/5` rescues, equal completion/cost, and slightly worse
yaw and recovery RMSE. No confirmation was run for either rejected candidate.
The research contribution here is the reproducible diagnosis, telemetry, and
falsification workflow; the confirmed dwell-gated transient actuator-recovery
method remains the supported adaptation result.

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
