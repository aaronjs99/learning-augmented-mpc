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
- baseline decentralized MPC with CVXPY/OSQP

Not implemented yet:
- LMPC safe-set and learned cost-to-go components

## Minimal Repository Layout
- `src/simulation/`: 3-agent environment and scenarios.
- `src/metrics/`: rollout metrics.
- `src/plotting/`: centralized plotting utilities.
- `src/mpc/`: shared baseline decentralized MPC controller and constraints.
- `src/learning/`: reserved for LMPC learning objects (next step).
- `scripts/`: thin runnable scripts.
- `experiments/`: experiment records (purpose/command/outputs/interpretation).
- `results/`: generated artifacts.

## Setup
Recommended minimal dependencies:
- `numpy`
- `matplotlib`
- `cvxpy`
- `osqp`

Install with:

`pip install -r requirements.txt`

## Run (Sanity Checks)
Run all scenarios with zero-control rollout:

`python scripts/run_sanity_checks.py --scenario all`

Run one scenario:

`python scripts/run_sanity_checks.py --scenario crossing_paths`

## Run (Baseline MPC)
Run decentralized baseline MPC on all scenarios:

`python scripts/run_baseline_mpc.py --scenario all`

Run only the nominal scenario:

`python scripts/run_baseline_mpc.py --scenario nominal_triangle_rotation`

Run all baseline scenarios with animations:

`python scripts/run_baseline_mpc.py --scenario all --make-video`

Reproduce the hard-constraint crossing-path infeasibility:

`python scripts/run_baseline_mpc.py --scenario crossing_paths --collision-mode hard_linearized`

## Expected Outputs
Sanity checks write a timestamped folder under `results/` with:
- per-scenario `metrics.json`
- per-scenario `trajectories.png`
- per-scenario `pairwise_distances.png`
- top-level `summary.json`

Baseline MPC writes a timestamped folder under `results/baseline/` with:
- top-level `summary.json`
- per-scenario `metrics.json`
- per-scenario `states.csv`
- per-scenario `controls.csv`
- per-scenario `solver_statuses.json`
- per-scenario `trajectories.png`
- per-scenario `pairwise_distances.png`
- per-scenario `animation.gif` when `--make-video` is used

`solver_statuses.json` records the collision mode, status counts per agent, and every timestep's per-agent solver status. The crossing-path scenario defaults to `soft_penalty` because its straight-line reference collapses all agents to the same midpoint, making the hard linearized constraints infeasible at the default input bound.

## Next Steps
1. Add LMPC safe-set and cost-to-go learning objects with no duplicated controller path.
2. Run failure-mode experiments and document outcomes in `experiments/`.

## License
This project is licensed under the MIT License. See `LICENSE`.
