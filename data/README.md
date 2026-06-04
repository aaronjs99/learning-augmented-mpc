# Data

The current manta LMPC workflow does not require external data files. Scenario
states, goals, obstacles, dynamics, and controller tuning live in
`config/manta.yaml`.

Use this folder only for optional local inputs, exports, or scratch data. Its
contents are ignored by git so generated or machine-specific files do not get
committed by accident.
