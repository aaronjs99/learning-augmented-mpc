# MPC

Controller builders for the project.

- `manta_lmpc.py`: active CasADi/IPOPT LMPC agent for the 7-state manta dynamics.
- `controller.py` and `constraints.py`: legacy 2D CVXPY baseline retained only as reference code.

LMPC horizons, costs, slack bounds, SVM distance thresholds, warm starts, solver
limits, and tolerances belong in `config/manta.yaml`.

The default terminal constraint matches the learned convex-hull state in
position only. Full 7-state terminal matching can be enabled with
`terminal_position_only: false`, but it is much easier to make infeasible for
the manta oscillator states.
