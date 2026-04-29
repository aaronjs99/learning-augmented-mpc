# Results

Generated artifacts from experiment runs live here.

## Layout
- `results/<experiment_id>/metrics.json`
- `results/<experiment_id>/trajectories.csv` (or equivalent compact format)
- `results/<experiment_id>/plots/`
- `results/<experiment_id>/interpretation.md`

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
