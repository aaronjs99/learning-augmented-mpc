# MPC

Shared decentralized MPC implementation for the 3-agent baseline.

## Files
- `controller.py`: one reusable CVXPY/OSQP QP controller used for each agent.
- `constraints.py`: shared input-bound and linearized collision constraints.

## Baseline Formulation
Each agent solves its own convex QP with single-integrator dynamics, quadratic goal tracking, quadratic control effort, componentwise input bounds, and pairwise collision constraints linearized around the previous predicted trajectories.

## Collision Modes
- `hard_linearized`: linearized pairwise collision constraints are hard QP constraints.
- `soft_penalty`: the same linearized constraints get nonnegative slacks with a quadratic penalty, preserving feasibility when the linearization reference is poor.
