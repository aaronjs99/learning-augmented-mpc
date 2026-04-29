# Source Layout

Keep implementation minimal, reusable, and non-duplicated.

## Implemented
- `src/simulation/`
  - `environment.py`: reusable 3-agent single-integrator environment and rollout
  - `scenarios.py`: named scenarios
- `src/metrics/`
  - `core.py`: rollout metric computation and pairwise distances
- `src/plotting/`
  - `trajectories.py`: trajectory and pairwise-distance plots
  - `animations.py`: optional rollout GIFs for baseline visual checks
- `src/mpc/`
  - `controller.py`: shared decentralized MPC QP controller
  - `constraints.py`: input bounds and linearized collision constraints

## Reserved for Next Phases
- `src/learning/`: LMPC safe-set and cost-to-go objects

## Rules
- One simulator path.
- One scenario-definition path.
- One metrics module.
- One plotting module.
- Thin scripts in `scripts/` only.
