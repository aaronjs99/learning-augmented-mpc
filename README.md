# MAE 271D: Decentralized Learning MPC

Config-driven multi-agent manta LMPC implementation plus a modular
heterogeneous harbor research testbed. The manta state is:

`[x, y, theta, p_L, q_L, p_R, q_R]`

The default run is controlled by `config/manta.yaml`.

## Workflow
1. Generate iteration-0 safe trajectories with a staged APF autopilot.
2. Run decentralized CasADi/IPOPT LMPC with learned safe-set terminal hulls and time-to-go costs.
3. Warm-start IPOPT from a blend of nominal controls and stored safe-set controls.
4. Use priority-aware SVM spatial hyperplanes plus soft slacks for inter-agent avoidance and a softened circular static-obstacle constraint.
5. Add only complete, collision-free rollouts back into the learned safe set.
6. Select the latest APF or LMPC iteration that reaches all goals without pairwise or obstacle violations.

## Layout
- `run.py`: root command dispatcher.
- `config/`: YAML runtime configuration.
- `docs/`: research contribution notes and experiment roadmap.
- `scripts/dynamics/`: manta dynamics and RK4 integration.
- `scripts/simulation/`: manta environment and scenario dataclasses.
- `scripts/harbor/`: independent UGV/USV/ROV dynamics, domains, communication,
  and heterogeneous simulation.
- `scripts/learning/`: APF initialization, safe-set sampling, SVM hyperplanes, runner.
- `scripts/mpc/`: CasADi LMPC agent builder.
- `scripts/metrics/`, `scripts/plotting/`: diagnostics, plots, and GIFs.
- `scripts/reporting/`: stable summary, CSV, plot, and animation artifact generation.
- `tests/`: fast regression tests for evaluation, safe-set admission, priority, and warm starts.
- `results/`: generated outputs.

## Setup
`python3 -m pip install -r requirements.txt`

## Run
Full manta LMPC using `config/manta.yaml`:

`python3 run.py`

APF baseline only, equivalent to iteration 0:

`python3 run.py baseline`

Zero-control manta sanity check:

`python3 run.py sanity`

Compact benchmark sweep, APF-only by default:

`python3 run.py sweep`

Fast regression tests (no IPOPT solve):

`python3 run.py test`

Heterogeneous harbor communication ablation (writes no file by default):

`python3 run.py harbor`

Useful overrides:

`python3 run.py --iterations 1`

`python3 run.py --scenario lane_swap --iterations 2 --max-steps 240 --no-video --output-dir results/lmpc/test_lane_swap`

Validated 2-agent LMPC benchmark:

`python3 run.py --scenario manta_crossover --iterations 2 --max-steps 230 --no-video --output-dir results/lmpc/test_manta_crossover`

Equivalent sweep summary:

`python3 run.py sweep --scenario manta_crossover --iterations 2 --max-steps 230`

`python3 run.py --make-video`

`python3 run.py --config config/manta.yaml --output-dir results/debug_run`

Outputs are written under the output root set in `config/manta.yaml`.
`summary.json` records every candidate iteration in `validation_by_iteration`.
`selected_iteration` is the APF or LMPC iteration used for the final plots.
`solver_clean: false` means the run used a solver fallback or an execution-time
safety intervention. The trajectory can still be valid if the recorded states
are complete and swept-collision-free; inspect both counts in `summary.json`.
`safe: true` with `valid: false` means the rollout avoided collisions but did
not reach every goal, so it is not reused as learned safe-set data.

## Tracked Results and Data
Timestamped run folders are ignored by git. Keep only the current report-ready
snapshot in `results/latest/`.

The current workflow does not require external data files. `data/` is available
for optional local inputs or scratch exports, but its contents are ignored
except for `data/README.md`.

## License
MIT. See `LICENSE`.
