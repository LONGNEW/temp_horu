from __future__ import annotations

"""Legacy import shim for historical checkpoints."""

from .trial.subspace_trial_rowgate_v3 import (
    SubspaceTrialRowGateCommonBasisMethod,
    SubspaceTrialRowGateV3Method,
)

__all__ = [
    "SubspaceTrialRowGateCommonBasisMethod",
    "SubspaceTrialRowGateV3Method",
]
