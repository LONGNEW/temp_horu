from __future__ import annotations

from typing import Literal

import torch

SimilarityMetric = Literal["dot", "cos"]


def similarity_scores(
    x_hv: torch.Tensor,
    class_hv: torch.Tensor,
    metric: SimilarityMetric = "dot",
    eps: float = 1e-8,
) -> torch.Tensor:
    if metric == "dot":
        return x_hv @ class_hv.T
    if metric == "cos":
        numerator = x_hv @ class_hv.T
        x_norms = torch.linalg.norm(x_hv, dim=1, keepdim=True).clamp_min(eps)
        class_norms = torch.linalg.norm(class_hv, dim=1).clamp_min(eps)
        return numerator / (x_norms * class_norms.unsqueeze(0))
    raise ValueError(f"Unsupported similarity metric: {metric}")
