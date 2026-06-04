# Learning

Learning and initialization modules for manta LMPC.

- `apf.py`: iteration-0 APF autopilot.
- `safe_sets.py`: staggered initial safe-set construction and terminal sampling.
- `hyperplanes.py`: SVM spatial hyperplanes for pairwise avoidance.
- `runner.py`: APF baseline plus repeated decentralized LMPC iterations.

Tuning values for these modules are loaded from `config/manta.yaml`.
