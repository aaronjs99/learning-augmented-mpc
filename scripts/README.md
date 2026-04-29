# Scripts

Thin runnable entry points only.

## Available
- `run_sanity_checks.py`: zero-control/open-loop checks for simulation, metrics, and plotting.
- `run_baseline_mpc.py`: closed-loop decentralized baseline MPC for all named scenarios.

## Usage
`python scripts/run_sanity_checks.py --scenario all`

`python scripts/run_baseline_mpc.py --scenario all`

`python scripts/run_baseline_mpc.py --scenario all --make-video`
