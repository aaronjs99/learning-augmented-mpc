# Plotting

Central plotting utilities for rollout diagnostics.

## Current plots
- 2D trajectories with start/goal markers
- pairwise distance time series with safety threshold
- rollout GIFs with agent positions, trails, goals, and safety radii

Plotting uses a noninteractive Matplotlib backend and an ignored local cache under `results/tmp/` so scripts can run from a clean terminal or CI-like environment.

No plotting logic should be duplicated in scripts.
