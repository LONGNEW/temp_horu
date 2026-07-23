from __future__ import annotations

from dataclasses import dataclass

import torch

from .subspace_trial import SubspaceTrialMethod
from .subspace_trial_core import row_normalize


@dataclass
class SubspaceTrialRowGateMethod(SubspaceTrialMethod):
    gate_alpha: float = 1.0
    gate_min: float = 0.1
    gate_max: float = 0.9

    def _row_gate_terms(
        self,
        payload: dict[str, torch.Tensor | str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total = payload["class_total_counts"].to(self.encoder.device)
        wrong = payload["class_wrong_counts"].to(self.encoder.device)

        error_ratio = wrong / total.clamp_min(1.0)
        rollback_gate = (1.0 - (float(self.gate_alpha) * error_ratio)).clamp(
            min=float(self.gate_min),
            max=float(self.gate_max),
        )
        rollback_gate = torch.where(total > 0, rollback_gate, torch.zeros_like(rollback_gate))
        follow_ratio = torch.where(total > 0, 1.0 - rollback_gate, torch.zeros_like(rollback_gate))
        return rollback_gate, follow_ratio

    def _server_shared_coord_sync(
        self,
        payload: dict[str, torch.Tensor | str],
        global_shared: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_shared = payload["shared_coords"].to(self.encoder.device)
        _, follow_ratio = self._row_gate_terms(payload)
        updated = local_shared + (float(self.global_lr) * follow_ratio).unsqueeze(1) * (global_shared - local_shared)
        return row_normalize(updated), follow_ratio.to(torch.float32)
