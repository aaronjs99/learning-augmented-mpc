"""Decentralized MPC package for the 3-agent project."""

from .manta_lmpc import MantaAgentOptimizer, MantaLMPCConfig

try:
    from .controller import MPCController
except ModuleNotFoundError as exc:
    if exc.name != "cvxpy":
        raise
    MPCController = None  # type: ignore[assignment]

__all__ = ["MantaAgentOptimizer", "MantaLMPCConfig", "MPCController"]
