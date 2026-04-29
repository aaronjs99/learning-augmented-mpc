# Roadmap

## Goal
Deliver a minimal, reproducible decentralized LMPC pipeline for 3 collision-aware agents.

## Phase 0 (Done): Simulation and Evaluation Foundation
- Fixed dynamics for initial build: single-integrator.
- Implemented one reusable 3-agent environment.
- Implemented centralized named scenarios.
- Implemented metrics and plotting utilities.
- Added thin zero-control sanity-check script.

## Phase 1 (Done): Baseline Decentralized MPC
- Implemented one decentralized MPC controller interface.
- Reused existing simulation/scenario/metrics/plotting modules.
- Baseline command: `python scripts/run_baseline_mpc.py --scenario all`
- Acceptance:
  - agents reach goals in nominal scenario
  - no collisions under nominal settings
  - reproducible command + documented outputs

## Phase 2: LMPC Learning Components
- Add learned safe-set storage from successful trajectories.
- Add learned terminal cost-to-go from stored trajectories.
- Keep MPC/LMPC in one controller path with config switches only.

## Phase 3: Failure-Mode Analysis
- Evaluate shifted initial/goal and/or tighter safety settings.
- Document failure types (infeasibility, collision risk, goal failure).

## Phase 4: Optional BO (Only After 1-3)
- Optional tuning of selected hyperparameters only.

## Out of Scope for Initial Delivery
- Agent counts other than 3.
- Multiple duplicated controller or simulator codepaths.
