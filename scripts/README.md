# Scripts Package

This package contains both runnable entry points and reusable modules.

## Entry Points
- `run_manta_lmpc.py`: full APF + LMPC workflow, driven by `config/manta.yaml`.
- `run_baseline_mpc.py`: APF iteration-0 baseline shortcut.
- `run_sanity_checks.py`: zero-control manta simulation check.
- `run_sweep.py`: compact APF/LMPC benchmark table without plot generation.
- `run_tests.py`: dependency-light unit-test discovery.

Prefer the root dispatcher:

`python run.py`

`python run.py baseline`

`python run.py sanity`

`python run.py sweep`

`python run.py test`

## Modules
- `dynamics/`: shared manta/CPG dynamics.
- `simulation/`: environment and scenario dataclasses.
- `learning/`: staged APF safe sets, SVM hyperplanes, policies, run loop.
- `mpc/`: CasADi/IPOPT LMPC optimizer.
- `metrics/`: shared history conversion, rollout validation, costs, and metrics.
- `plotting/`: trajectory, learning-progress, cost, and GIF outputs.
- `reporting/`: reusable summary preparation and run artifact serialization.

Runtime constants should live in YAML under `config/` unless they are structural code assumptions.
