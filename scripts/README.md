# Scripts Package

This package contains both runnable entry points and reusable modules.

## Entry Points
- `run_manta_lmpc.py`: full APF + LMPC workflow, driven by `config/manta.yaml`.
- `run_baseline_mpc.py`: APF iteration-0 baseline shortcut.
- `run_sanity_checks.py`: zero-control manta simulation check.

Prefer the root dispatcher:

`python run.py`

`python run.py baseline`

`python run.py sanity`

## Modules
- `dynamics/`: shared manta/CPG dynamics.
- `simulation/`: environment and scenario dataclasses.
- `learning/`: staged APF safe sets, SVM hyperplanes, validation, run loop.
- `mpc/`: CasADi/IPOPT LMPC optimizer.
- `metrics/`: rollout metrics from position dimensions.
- `plotting/`: trajectory, learning-progress, cost, and GIF outputs.

Runtime constants should live in YAML under `config/` unless they are structural code assumptions.
