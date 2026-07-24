from __future__ import annotations

"""Legacy import shim for historical checkpoints.

`subspace_trial_core` historically exposed many helpers/classes, so this shim keeps
wildcard re-export to preserve pickle compatibility.
"""

from .trial.subspace_trial_core import *  # noqa: F401,F403
