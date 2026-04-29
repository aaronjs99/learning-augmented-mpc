# Experiment: baseline_mpc

## Purpose
Evaluate the minimal decentralized 3-agent MPC baseline before adding LMPC safe sets or learned cost-to-go.

## Command
`python scripts/run_baseline_mpc.py --scenario all`

`python scripts/run_baseline_mpc.py --scenario all --make-video`

## Outputs
- `results/baseline/baseline_<timestamp>/summary.json`
- `results/baseline/baseline_<timestamp>/<scenario>/metrics.json`
- `results/baseline/baseline_<timestamp>/<scenario>/states.csv`
- `results/baseline/baseline_<timestamp>/<scenario>/controls.csv`
- `results/baseline/baseline_<timestamp>/<scenario>/solver_statuses.json`
- `results/baseline/baseline_<timestamp>/<scenario>/trajectories.png`
- `results/baseline/baseline_<timestamp>/<scenario>/pairwise_distances.png`
- Optional: `results/baseline/baseline_<timestamp>/<scenario>/animation.gif` from `--make-video`

## Interpretation
The nominal scenario should show all agents reaching their goals without collision.

The hard-linearized `crossing_paths` QPs are infeasible because the straight-line reference sends all agents through the same midpoint, while the initial spacing is safely above the threshold. The baseline therefore uses `soft_penalty` for `crossing_paths`; this restores solver feasibility and goal convergence, but the scenario remains a safety stress case if `collision_count` is nonzero.

The shifted-start scenario is retained as an expected stress/failure candidate for later failure-mode analysis.
