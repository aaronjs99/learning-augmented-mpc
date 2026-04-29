# Experiments

This directory stores reproducible experiment records for the 3-agent decentralized LMPC project.

## Required Record Per Experiment
- Purpose
- Command
- Outputs (files/paths)
- Interpretation
- Commit hash

## Naming
Use one directory or markdown record per experiment id, for example:
- `exp_YYYY-MM-DD_baseline_mpc`
- `exp_YYYY-MM-DD_lmpc_learning`
- `exp_YYYY-MM-DD_failure_shifted_init_goal`

## Required Experiment Sequence
1. Baseline MPC (3 agents)
2. LMPC with learned safe set + learned cost-to-go
3. Failure-mode analysis (shifted conditions and/or tighter safety)
4. Optional BO tuning only after 1-3

## Canonical Planned Commands
- `python scripts/run_baseline_mpc.py --scenario all`
- `python scripts/run_experiment.py --scenario baseline --controller lmpc`
- `python scripts/run_experiment.py --scenario shifted_init_goal --controller lmpc`

## Minimal Record Template
- Purpose:
- Command:
- Outputs:
- Interpretation:
- Commit:
