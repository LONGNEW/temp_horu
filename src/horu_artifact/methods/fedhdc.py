"""FedHDC's USER_SPECIFIED bootstrap and stale-batch local update."""
from __future__ import annotations

import time
import torch

from ..hdc.prototype import PrototypeMemory


def normalize_rows(memory: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.vector_norm(memory, dim=1, keepdim=True)
    return torch.where(norms > 0, memory / norms.clamp_min(torch.finfo(memory.dtype).eps), memory)


def bundled_model(encoded: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Sum each class's encoded samples, then normalize non-empty rows."""
    model = torch.zeros((num_classes, encoded.shape[1]), dtype=torch.float32, device=encoded.device)
    model.index_add_(0, labels, encoded)
    return normalize_rows(model)


def weighted_aggregate(models: list[torch.Tensor], sample_counts: list[int]) -> torch.Tensor:
    if not models or len(models) != len(sample_counts) or any(n <= 0 for n in sample_counts):
        raise ValueError("models and positive sample_counts must align")
    total = sum(sample_counts)
    combined = torch.zeros_like(models[0])
    for model, count in zip(models, sample_counts):
        if model.shape != combined.shape: raise ValueError("all models must have the same shape")
        combined.add_(model, alpha=count / total)
    return normalize_rows(combined)


def train_batches(model: torch.Tensor, encoded: torch.Tensor, labels: torch.Tensor, learning_rate: float, batch_size: int = 16, timing: dict[str, float] | None = None) -> int:
    """Fixed-prediction batch deltas; normalize only rows changed in each batch."""
    if batch_size <= 0: raise ValueError("batch_size must be positive")
    memory = PrototypeMemory(model)
    updates = 0
    similarity_ns = update_ns = 0
    for start in range(0, encoded.shape[0], batch_size):
        features, target = encoded[start:start + batch_size], labels[start:start + batch_size]
        begun = time.perf_counter_ns()
        predicted = memory.predict(features, "dot")
        similarity_ns += time.perf_counter_ns() - begun
        begun = time.perf_counter_ns()
        wrong = predicted != target
        if not torch.any(wrong):
            update_ns += time.perf_counter_ns() - begun
            continue
        unit = features[wrong] / torch.linalg.vector_norm(features[wrong], dim=1, keepdim=True).clamp_min(torch.finfo(features.dtype).eps)
        delta = torch.zeros_like(model)
        delta.index_add_(0, target[wrong], unit)
        delta.index_add_(0, predicted[wrong], -unit)
        changed = torch.linalg.vector_norm(delta, dim=1) > 0
        model.add_(delta, alpha=learning_rate)
        model[changed] = normalize_rows(model[changed])
        updates += int(wrong.sum().item())
        update_ns += time.perf_counter_ns() - begun
    if timing is not None:
        timing["similarity_ms"] = timing.get("similarity_ms", 0.0) + similarity_ns / 1e6
        timing["update_ms"] = timing.get("update_ms", 0.0) + update_ns / 1e6
    return updates
