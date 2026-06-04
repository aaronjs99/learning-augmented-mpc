# MAE 271D: Decentralized Learning MPC

Config-driven 3-agent manta LMPC implementation. The active state is:

`[x, y, theta, p_L, q_L, p_R, q_R]`

The default run is controlled by `config/manta.yaml`.

## Workflow
1. Generate iteration-0 safe trajectories with an APF autopilot.
2. Run decentralized CasADi/IPOPT LMPC with learned safe-set terminal hulls and time-to-go costs.
3. Use SVM spatial hyperplanes plus soft slacks for inter-agent avoidance and a softened circular static-obstacle constraint.

## Layout
- `run.py`: root command dispatcher.
- `config/`: YAML runtime configuration.
- `scripts/dynamics/`: manta dynamics and RK4 integration.
- `scripts/simulation/`: manta environment and scenario dataclasses.
- `scripts/learning/`: APF initialization, safe-set sampling, SVM hyperplanes, runner.
- `scripts/mpc/`: CasADi LMPC agent builder.
- `scripts/metrics/`, `scripts/plotting/`: diagnostics, plots, and GIFs.
- `results/`: generated outputs.

## Setup
`pip install -r requirements.txt`

## Run
Full manta LMPC using `config/manta.yaml`:

`python run.py`

APF baseline only, equivalent to iteration 0:

`python run.py baseline`

Zero-control manta sanity check:

`python run.py sanity`

Useful overrides:

`python run.py --iterations 1`

`python run.py --make-video`

`python run.py --config config/manta.yaml --output-dir results/debug_run`

Outputs are written under `results/manta_lmpc/` by default.

## Tracked Results and Data
Timestamped run folders are ignored by git. Keep only the current report-ready
snapshot in `results/latest_working/`.

The current workflow does not require external data files. `data/` is available
for optional local inputs or scratch exports, but its contents are ignored
except for `data/README.md`.

## License
MIT. See `LICENSE`.
