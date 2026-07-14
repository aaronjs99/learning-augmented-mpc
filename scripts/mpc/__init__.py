"""Decentralized MPC package for multi-agent trajectory optimization."""

from .manta_lmpc import MantaAgentOptimizer, MantaLMPCConfig

try:
    from .controller import MPCController
except ModuleNotFoundError as exc:
    if exc.name != "cvxpy":
        raise
    MPCController = None  # type: ignore[assignment]

__all__ = ["MantaAgentOptimizer", "MantaLMPCConfig", "MPCController"]
