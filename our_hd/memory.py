from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ClassMemory:
    weight: torch.Tensor

    @classmethod
    def zeros(
        cls,
        num_classes: int,
        hd_dim: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "ClassMemory":
        weight = torch.zeros(num_classes, hd_dim, device=device, dtype=dtype)
        return cls(weight=weight)

    @classmethod
    def from_encoded(
        cls,
        x_hv: torch.Tensor,
        y: torch.Tensor,
        num_classes: int,
    ) -> "ClassMemory":
        weight = torch.zeros(num_classes, x_hv.shape[1], device=x_hv.device, dtype=x_hv.dtype)
        weight.index_add_(0, y.long(), x_hv)
        return cls(weight=weight)

    def clone(self) -> "ClassMemory":
        return ClassMemory(weight=self.weight.clone())

    def normalize_(self, eps: float = 1e-8) -> "ClassMemory":
        self.weight = self.weight / (torch.linalg.norm(self.weight, dim=1, keepdim=True) + eps)
        return self

    def normalize_masked_(self, mask: torch.Tensor, eps: float = 1e-8) -> "ClassMemory":
        if mask.ndim != 1 or mask.shape[0] != self.weight.shape[1]:
            raise ValueError(
                f"mask must be 1D with length {self.weight.shape[1]}, got shape {tuple(mask.shape)}"
            )
        mask = mask.to(device=self.weight.device, dtype=torch.bool)
        if not torch.any(mask):
            return self
        active = self.weight[:, mask]
        norms = torch.linalg.norm(active, dim=1, keepdim=True).clamp_min(eps)
        self.weight[:, mask] = active / norms
        return self
