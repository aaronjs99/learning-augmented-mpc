# MAE 271D: Decentralized Learning MPC (3-Agent)

Minimal, reproducible project for decentralized LMPC with collision-aware trajectory optimization for exactly 3 agents.

## Scope (Required Story)
1. Baseline decentralized MPC for 3 agents.
2. Decentralized LMPC with learned safe sets and learned cost-to-go.
3. Failure-mode analysis under shifted initial/goal conditions and/or tighter safety constraints.
4. Optional BO only after 1-3 are complete.

## Current Implementation Status
Implemented now:
- simulation layer (single-integrator, 3 agents)
- centralized scenario definitions
- rollout metrics
- plotting utilities
- thin sanity-check script (zero-control/open-loop)

Not implemented yet:
- MPC and LMPC controllers

## Minimal Repository Layout
- `src/simulation/`: 3-agent environment and scenarios.
- `src/metrics/`: rollout metrics.
- `src/plotting/`: centralized plotting utilities.
- `src/mpc/`: reserved for baseline MPC (next step).
- `src/learning/`: reserved for LMPC learning objects (next step).
- `scripts/`: thin runnable scripts.
- `experiments/`: experiment records (purpose/command/outputs/interpretation).
- `results/`: generated artifacts.

## Setup
Recommended minimal dependencies:
- `numpy`
- `matplotlib`

## Run (Sanity Checks)
Run all scenarios with zero-control rollout:

`python scripts/run_sanity_checks.py --scenario all`

Run one scenario:

`python scripts/run_sanity_checks.py --scenario crossing_paths`

## Expected Outputs
Each run writes a timestamped folder under `results/` with:
- per-scenario `metrics.json`
- per-scenario `trajectories.png`
- per-scenario `pairwise_distances.png`
- top-level `summary.json`

## Next Steps
1. Implement baseline decentralized MPC on top of the existing simulation/scenario/metrics stack.
2. Add LMPC safe-set and cost-to-go learning objects with no duplicated controller path.
3. Run failure-mode experiments and document outcomes in `experiments/`.
