"""Decentralized MPC package for multi-agent trajectory optimization."""

from .manta_lmpc import MantaAgentOptimizer, MantaLMPCConfig, MantaStepSolution

__all__ = ["MantaAgentOptimizer", "MantaLMPCConfig", "MantaStepSolution"]
