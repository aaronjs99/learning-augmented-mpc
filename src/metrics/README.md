# Metrics

Run-level metrics for trajectory rollouts.

## Current metrics
- total cost proxy
- minimum pairwise distance
- collision count
- time-to-goal
- control effort (if controls are provided)

Keep metric computation in one module to avoid duplicate analysis logic.
