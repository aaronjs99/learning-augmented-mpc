# Results

Generated runs write timestamped folders under `results/` and are ignored by
git by default.

The one exception is `results/latest/`. Use that folder for the curated
result snapshot that should travel with the code and report. When a new run is
better, replace `results/latest/` with that run's key files.

Typical tracked snapshot files:
- `summary.json`
- `states_by_iteration.csv`
- `learning_progression.png`
- `cost_decrease.png`
- `final_trajectories.png`
- `pairwise_distances.png`
- optional `final_iteration.gif`

The physical harbor study keeps exactly five files under `latest/harbor/`:

- `metrics.json`: guidance, MPC, admitted LMPC, and rejected-attempt telemetry.
- `research_progress.png`: trajectories, safe reference, cost, clearance, and
  fallback diagnostics.
- `harbor_lmpc.gif`: best admitted rollout with yaw and ROV pitch/depth views.
- `horizon_study.json`: matched `N=8/12/15` MPC/LMPC evidence.
- `horizon_efficiency.png`: liveness, completion cost, and mean NLP latency.
