# Plotting

Central plotting utilities for manta rollout diagnostics.

Current outputs:
- 2D trajectories with start/goal markers.
- Pairwise distance time series with safety threshold.
- APF/LMPC learning progression plot.
- Cost-by-iteration plot.
- Optional final-iteration GIF with trajectory and wing-motion views.

Plotting uses a noninteractive Matplotlib backend so runs work from the terminal.
Shared primitives keep agent colors, obstacle layers, goal tolerances, workspace
limits, and output-directory handling consistent across static plots and GIFs.
