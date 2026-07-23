"""Paper-faithful HyperFeel central-AM bootstrap and delta retraining.

The operations here deliberately use raw encoded hypervectors and never row
normalize a prototype.  The federated runner owns persistence and transport.
"""
from __future__ import annotations

import time
import torch

from ..hdc.prototype import PrototypeMemory


def bundled_model(encoded: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Build a local associative memory by class-wise raw-vector bundling."""
    memory = torch.zeros((num_classes, encoded.shape[1]), dtype=torch.float32, device=encoded.device)
    memory.index_add_(0, labels, encoded)
    return memory


def normalize_rows(memory: torch.Tensor) -> torch.Tensor:
    """Diagnostic-only row normalization; paper-faithful mode never calls it."""
    norms = torch.linalg.vector_norm(memory, dim=1, keepdim=True)
    return torch.where(norms > 0, memory / norms.clamp_min(torch.finfo(memory.dtype).eps), memory)


def sum_deltas(deltas: list[torch.Tensor]) -> torch.Tensor:
    """Algorithm 1 server aggregation: element-wise sum, not an average."""
    if not deltas:
        raise ValueError("at least one delta is required")
    return torch.stack(deltas).sum(dim=0)


def personalization_weights(errors: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Eq. (3): classes absent from the local pass receive zero weight."""
    return torch.where(counts > 0, errors / counts, torch.zeros_like(errors))


def apply_personalization(memory: torch.Tensor, previous_delta: torch.Tensor, weights: torch.Tensor, learning_rate: float) -> None:
    """Apply the prior global delta class by class in-place."""
    memory.add_(previous_delta * weights[:, None], alpha=learning_rate)


def retrain_samplewise(memory: torch.Tensor, encoded: torch.Tensor, labels: torch.Tensor, learning_rate: float, normalize_update_hypervectors: bool = False, normalize_prototypes: bool = False, timing: dict[str, float] | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Sequentially process samples with the current personalized AM.

    Returns ``(client_delta, error_counts, class_counts, mistakes)``.  Counts
    are by true class, matching the class-specific personalization weight.
    """
    delta = torch.zeros_like(memory)
    errors = torch.zeros(memory.shape[0], dtype=torch.float32, device=memory.device)
    counts = torch.zeros_like(errors)
    mistakes = 0
    predictor = PrototypeMemory(memory)
    similarity_ns = update_ns = 0
    for q, target in zip(encoded, labels):
        begun = time.perf_counter_ns()
        target_index = int(target.item())
        counts[target_index] += 1
        update_ns += time.perf_counter_ns() - begun
        begun = time.perf_counter_ns()
        predicted = int(predictor.predict(q.unsqueeze(0), "dot").item())
        similarity_ns += time.perf_counter_ns() - begun
        if predicted == target_index:
            continue
        begun = time.perf_counter_ns()
        update = q
        if normalize_update_hypervectors:
            update = q / torch.linalg.vector_norm(q).clamp_min(torch.finfo(q.dtype).eps)
        memory[target_index].add_(update, alpha=learning_rate)
        memory[predicted].add_(update, alpha=-learning_rate)
        delta[target_index].add_(update, alpha=learning_rate)
        delta[predicted].add_(update, alpha=-learning_rate)
        if normalize_prototypes:
            memory[[target_index, predicted]] = normalize_rows(memory[[target_index, predicted]])
        errors[target_index] += 1
        mistakes += 1
        update_ns += time.perf_counter_ns() - begun
    if timing is not None:
        timing["similarity_ms"] = timing.get("similarity_ms", 0.0) + similarity_ns / 1e6
        timing["update_ms"] = timing.get("update_ms", 0.0) + update_ns / 1e6
    return delta, errors, counts, mistakes


def retrain_batches(memory: torch.Tensor, encoded: torch.Tensor, labels: torch.Tensor, learning_rate: float, batch_size: int = 32, normalize_update_hypervectors: bool = False, normalize_prototypes: bool = False, timing: dict[str, float] | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Retrain with vectorized predictions and one stale-model update per batch."""
    delta = torch.zeros_like(memory)
    errors = torch.zeros(memory.shape[0], dtype=torch.float32, device=memory.device)
    counts = torch.bincount(labels, minlength=memory.shape[0]).to(torch.float32)
    mistakes = 0
    similarity_ns = update_ns = 0
    predictor = PrototypeMemory(memory)
    for begin in range(0, labels.numel(), batch_size):
        features = encoded[begin:begin + batch_size]
        targets = labels[begin:begin + batch_size]
        begun = time.perf_counter_ns()
        predictions = predictor.predict(features, "dot")
        similarity_ns += time.perf_counter_ns() - begun
        begun = time.perf_counter_ns()
        wrong = predictions != targets
        if bool(wrong.any()):
            updates = features[wrong]
            if normalize_update_hypervectors:
                updates = updates / torch.linalg.vector_norm(
                    updates, dim=1, keepdim=True
                ).clamp_min(torch.finfo(updates.dtype).eps)
            true_labels = targets[wrong]
            predicted_labels = predictions[wrong]
            batch_delta = torch.zeros_like(memory)
            batch_delta.index_add_(0, true_labels, updates)
            batch_delta.index_add_(0, predicted_labels, -updates)
            batch_delta.mul_(learning_rate)
            memory.add_(batch_delta)
            delta.add_(batch_delta)
            errors.add_(torch.bincount(true_labels, minlength=memory.shape[0]))
            mistakes += int(wrong.sum().item())
            if normalize_prototypes:
                memory.copy_(normalize_rows(memory))
        update_ns += time.perf_counter_ns() - begun
    if timing is not None:
        timing["similarity_ms"] = timing.get("similarity_ms", 0.0) + similarity_ns / 1e6
        timing["update_ms"] = timing.get("update_ms", 0.0) + update_ns / 1e6
    return delta, errors, counts, mistakes
