"""Metric helpers that deliberately distinguish bootstrap and round costs."""
from __future__ import annotations
import hashlib
import torch

def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {"mean_accuracy": sum(values) / len(values), "p10_accuracy": ordered[max(0, int(.1 * (len(values) - 1)))], "worst_accuracy": ordered[0]}
