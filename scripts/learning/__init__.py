"""Learning and initialization helpers for manta LMPC."""

from .apf import (
    APFConfig,
    backup_apf_control,
    compute_apf_control,
    simulate_manta_autopilot,
    simulate_manta_autopilot_with_controls,
)
from .hyperplanes import get_symmetric_hyperplanes_spatial
from scripts.metrics import cost_by_iteration

from .policies import priority_margins, warm_start_from_safe_set
from .recovery import RecoveryResult, repair_incomplete_with_apf, safe_fallback_apf_step
from .runner import MantaLMPCRunResult, run_manta_lmpc
from .safe_sets import (
    build_staggered_safe_sets,
    hold_controls,
    hold_trajectory,
    sample_terminal_safe_set,
)

__all__ = [
    "APFConfig",
    "backup_apf_control",
    "build_staggered_safe_sets",
    "compute_apf_control",
    "cost_by_iteration",
    "get_symmetric_hyperplanes_spatial",
    "hold_controls",
    "hold_trajectory",
    "MantaLMPCRunResult",
    "priority_margins",
    "RecoveryResult",
    "repair_incomplete_with_apf",
    "run_manta_lmpc",
    "safe_fallback_apf_step",
    "sample_terminal_safe_set",
    "simulate_manta_autopilot",
    "simulate_manta_autopilot_with_controls",
    "warm_start_from_safe_set",
]
