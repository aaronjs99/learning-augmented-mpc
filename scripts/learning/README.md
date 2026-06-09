# Learning

Learning and initialization modules for manta LMPC.

- `apf.py`: iteration-0 APF autopilot and one-step APF backup control.
- `safe_sets.py`: dynamically valid staggered safe-set construction and terminal sampling.
- `hyperplanes.py`: SVM spatial hyperplanes for pairwise avoidance.
- `runner.py`: APF baseline plus repeated decentralized LMPC iterations.

Tuning values for these modules are loaded from `config/manta.yaml`.
