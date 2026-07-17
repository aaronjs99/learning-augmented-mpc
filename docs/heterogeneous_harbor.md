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
| Nominal MPC | 182 | 0.235 | 13 |
| Scalar-adaptive MPC | 168 | 0.158 | 7 |
| Passive diagonal MPC | **159** | 0.0137 | 0 |
| Active diagonal MPC | 166 | **0.0100** | 0 |
| One-pass active MPC | 160 | 0.0297 | 0 |
| Information-aware MPC | 160 | **0.0218** | 0 |
| Retained passive LMPC | 168 | 0.0137 | 0 |
| Retained active-ID LMPC | 168 | **0.0100** | 0 |
| Retained information-ID LMPC | 168 | 0.0218 | 0 |

All nine rollouts are complete and swept-safe with numerical-zero collision
slack. The nominal and scalar baselines require solver fallbacks and therefore
are not admitted; all seven actuator-wise trials are fallback-free and valid.
Round-robin active MPC reduces gain RMSE by `26.6%` relative to passive
diagonal MPC, while increasing completion cost by seven steps (`4.4%`). The
equal-budget information-aware MPC reduces RMSE by `26.8%` relative to
one-pass probing: both execute 14 direct probes and cost 160 steps. Retained
LMPC trials freeze the preceding local actuator estimate, so each retained
trial preserves its source RMSE exactly instead of conflating model transfer
with a second low-excitation identification process. Their equal cost of 168
does not establish a task-time learning improvement in this fixed fault case.

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

## Stratified Fault Generalization

The generalization command replaces one hand-selected vector with five
deterministic per-channel Latin-hypercube draws over effectiveness `[0.55,
0.98]`. Every one of the 14 physical channels occupies each severity stratum
once, while each named platform retains a separate hidden vector.

```text
python run.py harbor-fault-generalization
```

| Controller | Mean RMSE | Mean sustained cost | Valid | Complete / safe |
| --- | ---: | ---: | ---: | ---: |
| Passive diagonal MPC | 0.0394 | **149.6** | 3/5 | 5/5 |
| One-pass active MPC | 0.0464 | 157.2 | 5/5 | 5/5 |
| Information-aware MPC | **0.0262** | 157.4 | 5/5 | 5/5 |

Fault-focused information scheduling beats equal-budget one-pass probing in
all five paired cases. Mean paired relative RMSE reduction is `37.96%`; mean
absolute reduction is `0.0201`, with a case-bootstrap 95% interval `[0.0080,
0.0360]`. Mean sustained-completion cost changes by only `+0.2` steps and by at
most two steps in any pair. Both active policies execute one probe per channel,
complete every task, remain swept-safe, and use no solver fallbacks. Passive
adaptation completes safely but incurs fallbacks in two cases.

Five stratified cases are a controlled generalization check, not a population-
level guarantee. The bootstrap interval describes this configured ensemble;
hardware trials, more fault draws, and time-varying failures are still required
for a broad statistical claim. The scheduled-fault study below supplies an
initial time-varying benchmark, but not broader population or hardware evidence.

## Noisy Local-Observation Study

```text
python run.py harbor-fault-noise-study
```

This repeats the same five hidden-fault draws with deterministic UGV/USV/ROV
state-observation noise. Controller inputs are noisy; safety and completion are
evaluated from plant truth. Recursive passive estimation reduces mean actuator
RMSE from `0.1529` for the instantaneous fit to `0.0392`, a mean paired relative
reduction of `72.42%`, and wins all five matched cases. The paired absolute
reduction is `0.1137` with bootstrap 95% interval `[0.0779, 0.1496]`.

After correcting the recursive-estimator probe gate, one-pass and information-
aware probing reach `0.0748` and `0.0510`; passive recursive RLS remains best at
`0.0392`. Information-aware ordering beats one-pass in only two of five cases.
Its paired mean reduction is `0.0238`, but bootstrap interval `[-0.0055,
0.0532]` crosses zero. It lowers completion cost by `6.8` steps on average.
Every rollout completes and remains collision-safe. Fourteen of 15 recursive
rollouts are solver-clean; one-pass retains one fallback in seed 37. The
instantaneous baseline retains one Inspector-Gadget fallback in seed 71.

## Intent-Bounded Peer Prediction

```text
python run.py harbor-prediction-study
```

This ablation holds the safe-memory seed, recursive estimator, hidden faults,
observation seeds, communication, and hard collision constraints fixed. The
legacy constant-velocity model incurs seven recoverable Inspector-Gadget
fallbacks in one of five cases. Capping aligned peer travel at its communicated
goal reduces fallbacks `7 -> 0`, raises fallback-free case rate `80% -> 100%`,
and changes mean completion cost by `-0.6` steps. All five pairs complete and
remain collision-safe. Mean actuator RMSE changes only `0.03846 -> 0.03934`,
supporting peer-prediction feasibility rather than improved identification as
the mechanism.

## Time-Varying Actuator-Fault Tracking

```text
python run.py harbor-time-varying-fault-study
```

This matched ablation switches hidden per-channel effectiveness vectors down
and back to nominal during active execution while retaining the same safe-
memory seed, communication model, schedule, and noisy observations for every
controller. The controller does not receive change timing or magnitude.
Passive chi-square CUSUM wins all three seeds during degradation: RMSE changes
`0.1695 -> 0.0979` (`42.27%` paired relative reduction), and completion cost
changes `154.3 -> 149.3`. A one-step innovation threshold reaches `0.1006` at
the same cost.

Change-triggered information probing remains dormant until CUSUM fires, then
reopens the local information budget and executes one safe request per physical
channel. Its recovery RMSE changes `0.1736 -> 0.1533` (`11.70%`), final RMSE
improves `9.61%`, and completion cost is `150.3`. All 12 rollouts complete,
remain collision-safe, solver-clean, and at numerical-zero collision slack.

Covariance-inflation events are reported for auditability but are not treated
as certified classifications. Mean causal recall is only `50-58%`, with
`2.0-2.7` unmatched inflations per run. Normalized innovations can respond to
noise and model mismatch as well as actuator loss. That original schedule
supports online tracking and recovery only for that configured benchmark; the
broader holdout below does not support a general recovery claim.

### Stratified development and holdout

```text
python run.py harbor-temporary-fault-generalization
python run.py harbor-temporary-fault-generalization --holdout
```

The five-case development ensemble varies all 14 physical actuator channels,
each platform's onset and duration, and observation noise. It motivated an
8-step detector warmup, 10-step cooldown, and loss-only probe arming, so its
results are development evidence rather than an untouched generalization test.

The separately seeded five-case holdout was then run once without further
tuning. All 20 matched rollouts completed and remained collision-safe.
Innovation-threshold RLS reduced mean degraded-interval RMSE from `0.18152` to
`0.11047` (`38.50%` paired relative reduction), won all five pairs, reduced
mean completion cost by `13.2` steps, and required no solver fallback. Its
absolute paired RMSE-reduction bootstrap interval was `[0.05805, 0.09210]`.

CUSUM reduced degraded-interval RMSE slightly further to `0.10803` (`40.42%`)
but incurred 22 recoverable solver fallbacks in holdout case `719` and increased
mean completion cost by `5.6` steps. Triggered probing reached `0.10074` but
shared the 22 fallbacks and increased cost by `7.8` steps. Recovery intervals
favor threshold RLS on average, but the paired interval crosses zero, and final
RMSE is unchanged. Therefore threshold RLS is the provisional robust candidate;
CUSUM and triggered probing remain informative ablations, not selected methods.
A fresh confirmation ensemble or hardware experiment is required because this
candidate was selected after inspecting the holdout.

The case-`719` failure is localized rather than inferred from an aggregate:
only the Clearpath Heron USV reports `Infeasible_Problem_Detected`, on steps
`35-56`, in both CUSUM variants. Its hidden effectiveness changes at steps `18`
and `40`, while CUSUM events occur at `20` and `42`. Threshold RLS remains
solver-clean and reaches the USV goal at step `46`; passive CUSUM reaches it at
step `111` through the deterministic guidance fallback. This establishes an
estimator/controller feasibility coupling around the restoration interval, but
does not by itself identify which estimated channel or constraint causes it.

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
5. Compare innovation-adaptive covariance against interacting-multiple-model,
   CUSUM, and observer-based fault estimators over larger onset and recovery
   ensembles, including false-alarm and detection-delay metrics.
