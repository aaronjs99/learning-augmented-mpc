# Learning

Learning and initialization modules for manta LMPC.

- `apf.py`: iteration-0 APF autopilot with optional extra static obstacles and one-step APF backup control.
- `safe_sets.py`: dynamically valid staged safe-set construction and terminal sampling.
- `hyperplanes.py`: SVM spatial hyperplanes for pairwise avoidance.
- `runner.py`: APF baseline plus repeated decentralized LMPC iterations, validation, and safe-set updates.

Tuning values for these modules are loaded from `config/manta.yaml`.

Only rollouts that are both collision-free and complete are added back into the
learned safe set. Safe-but-incomplete LMPC attempts are still reported in
`summary.json`, but they are not used as terminal safe-set data.
