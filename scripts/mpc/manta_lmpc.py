"""CasADi/IPOPT LMPC agent for the config-driven manta dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.dynamics import MantaDynamicsConfig
from scripts.dynamics.manta import rk4_step_ca
from scripts.simulation import StaticObstacle


@dataclass(frozen=True)
class MantaLMPCConfig:
    """Runtime LMPC settings loaded from ``config/manta.yaml``."""

    prediction_horizon: int = 20
    k_hull: int = 10
    dt: float = 0.2
    max_steps: int = 300
    iterations: int = 4
    goal_tolerance: float = 0.65
    hyperplane_safety_margin: float = 0.3
    hyperplane_slack_bound: float = 2.0
    static_slack_bound: float = 2.0
    terminal_slack_weight: float = 1000.0
    safety_slack_weight: float = 10000.0
    safe_cost_weight: float = 10.0
    state_cost_weights: tuple[float, ...] = (1.0, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0)
    control_cost_weights: tuple[float, ...] = (0.5, 0.5)
    hyperplane_ignore_distance: float = 4.0
    warm_start_control: float = 1.0
    ipopt_max_iter: int = 200
    ipopt_print_level: int = 0
    log_interval: int = 5


class MantaAgentOptimizer:
    """Single-agent manta LMPC NLP reused for each decentralized agent."""

    def __init__(
        self,
        *,
        config: MantaLMPCConfig,
        num_obstacles: int,
        obstacle: StaticObstacle,
        dynamics_config: MantaDynamicsConfig = MantaDynamicsConfig(),
    ) -> None:
        """Build one CasADi Opti problem from the loaded YAML settings."""
        try:
            import casadi as ca
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "CasADi is required for LMPC. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc

        self.ca = ca
        self.config = config
        self.num_obstacles = num_obstacles
        self.obstacle = obstacle
        self.dynamics_config = dynamics_config
        self._build_problem()

    def solve_step(
        self,
        *,
        current_state: np.ndarray,
        goal_state: np.ndarray,
        hyperplanes: list[tuple[np.ndarray, np.ndarray]],
        safe_states: np.ndarray,
        safe_costs: np.ndarray,
        warm_states: np.ndarray,
        warm_controls: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve one receding-horizon step and return ``(control, next_state)``."""
        opti = self.opti
        opti.set_value(self.p_init, current_state)
        opti.set_value(self.p_goal, goal_state)
        opti.set_value(self.p_safe_states, safe_states)
        opti.set_value(self.p_safe_costs, safe_costs)

        for obs_idx, (H_mat, h_vec) in enumerate(hyperplanes):
            opti.set_value(self.pH_list[obs_idx], H_mat)
            opti.set_value(self.ph_list[obs_idx], h_vec)

        opti.set_initial(self.X_state, warm_states)
        opti.set_initial(self.U, warm_controls)
        opti.set_initial(self.lambdas, 1.0 / self.config.k_hull)
        opti.set_initial(self.terminal_slack, 0.0)
        opti.set_initial(self.slack_hyper, 0.0)
        opti.set_initial(self.slack_static, 0.0)

        sol = opti.solve()
        return_status = str(opti.stats().get("return_status", ""))
        if _is_interrupted_status(return_status):
            raise KeyboardInterrupt
        return (
            np.asarray(sol.value(self.U[:, 0]), dtype=float).reshape(2),
            np.asarray(sol.value(self.X_state[:, 1]), dtype=float).reshape(7),
        )

    def _build_problem(self) -> None:
        ca = self.ca
        cfg = self.config
        dyn = self.dynamics_config
        N = cfg.prediction_horizon
        K = cfg.k_hull
        if len(cfg.state_cost_weights) != 7:
            raise ValueError("lmpc.state_cost_weights must contain 7 values")
        if len(cfg.control_cost_weights) != 2:
            raise ValueError("lmpc.control_cost_weights must contain 2 values")

        opti = ca.Opti()
        X_state = opti.variable(7, N + 1)
        U = opti.variable(2, N)

        opti.subject_to(opti.bounded(dyn.mu_min, U, dyn.mu_max))
        opti.subject_to(
            opti.bounded(-dyn.oscillator_bound, X_state[3:7, :], dyn.oscillator_bound)
        )

        p_init = opti.parameter(7, 1)
        p_goal = opti.parameter(7, 1)
        pH_list = [opti.parameter(N, 2) for _ in range(self.num_obstacles)]
        ph_list = [opti.parameter(N, 1) for _ in range(self.num_obstacles)]
        p_safe_states = opti.parameter(7, K)
        p_safe_costs = opti.parameter(K, 1)

        lambdas = opti.variable(K, 1)
        terminal_slack = opti.variable(7, 1)
        slack_hyper = opti.variable(self.num_obstacles, N)
        slack_static = opti.variable(1, N)

        opti.subject_to(opti.bounded(0.0, lambdas, 1.0))
        opti.subject_to(ca.sum1(lambdas) == 1.0)
        opti.subject_to(opti.bounded(0.0, slack_hyper, cfg.hyperplane_slack_bound))
        opti.subject_to(opti.bounded(0.0, slack_static, cfg.static_slack_bound))

        for k in range(N):
            x_next = rk4_step_ca(X_state[:, k], U[:, k], cfg.dt, dyn)
            opti.subject_to(X_state[:, k + 1] == x_next)

            pos_k = ca.vertcat(X_state[0, k + 1], X_state[1, k + 1])
            for obs_idx in range(self.num_obstacles):
                opti.subject_to(
                    ca.mtimes(pH_list[obs_idx][k, :], pos_k) + ph_list[obs_idx][k]
                    <= slack_hyper[obs_idx, k]
                )

            obs_x, obs_y = self.obstacle.center
            dist_sq = (X_state[0, k + 1] - obs_x) ** 2 + (
                X_state[1, k + 1] - obs_y
            ) ** 2
            opti.subject_to(dist_sq >= (self.obstacle.radius - slack_static[0, k]) ** 2)

        opti.subject_to(X_state[:, 0] == p_init)
        opti.subject_to(
            X_state[:, N] == ca.mtimes(p_safe_states, lambdas) + terminal_slack
        )

        Q = ca.DM(np.diag(cfg.state_cost_weights))
        R = ca.DM(np.diag(cfg.control_cost_weights))
        cost = 0
        for k in range(N):
            err = X_state[:, k] - p_goal
            cost += ca.dot(err, ca.mtimes(Q, err))
            cost += ca.dot(U[:, k], ca.mtimes(R, U[:, k]))

        cost += cfg.safe_cost_weight * ca.mtimes(lambdas.T, p_safe_costs)
        cost += cfg.terminal_slack_weight * ca.dot(terminal_slack, terminal_slack)
        cost += cfg.safety_slack_weight * ca.sum1(ca.vec(slack_hyper))
        cost += cfg.safety_slack_weight * ca.sum1(ca.vec(slack_static))

        opti.minimize(cost)
        opti.solver(
            "ipopt",
            {
                "print_time": 0,
                "ipopt.sb": "yes",
                "ipopt.max_iter": cfg.ipopt_max_iter,
                "ipopt.print_level": cfg.ipopt_print_level,
            },
        )

        self.opti = opti
        self.X_state = X_state
        self.U = U
        self.p_init = p_init
        self.p_goal = p_goal
        self.pH_list = pH_list
        self.ph_list = ph_list
        self.p_safe_states = p_safe_states
        self.p_safe_costs = p_safe_costs
        self.lambdas = lambdas
        self.terminal_slack = terminal_slack
        self.slack_hyper = slack_hyper
        self.slack_static = slack_static


def _is_interrupted_status(return_status: str) -> bool:
    return (
        "KeyboardInterrupt" in return_status
        or "KeyboardInterruptException" in return_status
        or "User_Requested_Stop" in return_status
    )
