# MPC

Controller builders for the project.

- `manta_lmpc.py`: active CasADi/IPOPT LMPC agent for the 7-state manta dynamics.

LMPC horizons, costs, slack bounds, SVM distance thresholds, warm starts, solver
limits, and tolerances belong in `config/manta.yaml`.

Control warm starts are provided by the learning runner, which blends nominal
controls with stored controls from the current safe-set trajectory before each
IPOPT solve.

The default terminal constraint matches the learned convex-hull state in
position only. Full 7-state terminal matching can be enabled with
`terminal_position_only: false`, but it is much easier to make infeasible for
the manta oscillator states.
