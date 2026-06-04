# Results

Generated runs write timestamped folders under `results/` and are ignored by
git by default.

The one exception is `results/latest_working/`. Use that folder for the curated
result snapshot that should travel with the code and report. When a new run is
better, replace `results/latest_working/` with that run's key files.

Typical tracked snapshot files:
- `summary.json`
- `states_by_iteration.csv`
- `learning_progression.png`
- `cost_decrease.png`
- `final_trajectories.png`
- `pairwise_distances.png`
- optional `final_iteration.gif`
