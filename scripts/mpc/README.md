# MPC

Controller builders for the project.

- `manta_lmpc.py`: active CasADi/IPOPT LMPC agent for the 7-state manta dynamics.
- `controller.py` and `constraints.py`: legacy 2D CVXPY baseline retained only as reference code.

LMPC horizons, costs, slack bounds, SVM distance thresholds, warm starts, solver
limits, and tolerances belong in `config/manta.yaml`.
