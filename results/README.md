# Results

Generated artifacts from experiment runs live here.

## Layout
- `results/sanity_<timestamp>/`: zero-control simulation checks.
- `results/baseline/baseline_<timestamp>/`: decentralized baseline MPC outputs.
- Per scenario: `metrics.json`, `states.csv`, `controls.csv`, `trajectories.png`, and `pairwise_distances.png`.
- Baseline runs also include `solver_statuses.json`.
- Baseline runs with `--make-video` also include `animation.gif`.

## Rules
- Do not manually edit raw run artifacts.
- Keep outputs traceable to one command and one commit.
- Keep only artifacts needed for reproducibility and reporting.

## Minimum Comparison Metrics
- success/failure per agent
- time-to-goal
- cumulative control effort
- minimum pairwise distance
- collision count
- solver feasibility rate
