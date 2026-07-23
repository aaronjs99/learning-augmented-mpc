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

The physical harbor study keeps compact JSON/figure pairs under
`latest/harbor/`; individual rollout folders and redundant animations remain
ignored. Core artifacts include:

- `metrics.json`: guidance, MPC, admitted LMPC, and rejected-attempt telemetry.
- `research_progress.png`: trajectories, safe reference, cost, clearance, and
  fallback diagnostics.
- `harbor_lmpc.gif`: best admitted rollout with yaw and ROV pitch/depth views.
- `horizon_study.json`: matched `N=8/12/15` MPC/LMPC evidence.
- `horizon_efficiency.png`: liveness, completion cost, and mean NLP latency.
- `temporary_fault_generalization.{json,png}`: stratified development cases.
- `temporary_fault_holdout.{json,png}`: separately seeded one-way evaluation.
- `temporary_fault_confirmation.{json,png}`: frozen threshold-RLS confirmation
  with machine-readable acceptance gates.
- `station_keeping_development.{json,png}`: actuator-independent marine-current
  observer evidence with an untouched confirmation ensemble.
- `range_aided_slam_development.{json,png}`: dead-reckoning, known-map,
  joint-landmark, and high-dropout localization evidence.
- `joint_localization_development.{json,png}`: estimated-state distributed MPC
  under matched current, range, observation, and temporary-actuator uncertainty.
- `joint_localization_candidate.gif`: representative complete, safe,
  zero-fallback joint-SLAM and belief-retry rollout.
- `fixed_lag_development.{json,png}`: three matched robust fixed-lag-SLAM
  development cases after the terminal-heading and ROV-depth corrections. This
  is development evidence, not a frozen confirmation result.
