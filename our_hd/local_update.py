from __future__ import annotations

from dataclasses import dataclass

import time
import torch

from .memory import ClassMemory
from .similarity import SimilarityMetric, similarity_scores


@dataclass
class LocalHDUpdater:
    epochs: int = 1
    batch_size: int = 32
    lr: float = 1.0
    metric: SimilarityMetric = "cos"

    def _prepare_update_inputs(
        self,
        weight: torch.Tensor,
        *,
        update_mask: torch.Tensor | None = None,
        update_basis: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        mask = None
        if update_mask is not None:
            if update_mask.ndim != 1 or update_mask.shape[0] != weight.shape[1]:
                raise ValueError(
                    f"update_mask must be 1D with length {weight.shape[1]}, got shape {tuple(update_mask.shape)}"
                )
            mask = update_mask.to(device=weight.device, dtype=weight.dtype).unsqueeze(0)

        basis = None
        if update_basis is not None:
            if update_basis.ndim != 2 or update_basis.shape[1] != weight.shape[1]:
                raise ValueError(
                    f"update_basis must have shape (rank, {weight.shape[1]}), got {tuple(update_basis.shape)}"
                )
            basis = update_basis.to(device=weight.device, dtype=weight.dtype)
        return mask, basis

    def step(
        self,
        memory: ClassMemory,
        x_hv: torch.Tensor,
        y: torch.Tensor,
        *,
        update_mask: torch.Tensor | None = None,
        update_basis: torch.Tensor | None = None,
    ) -> ClassMemory:
        updated, _ = self.profiled_step(
            memory,
            x_hv,
            y,
            update_mask=update_mask,
            update_basis=update_basis,
        )
        return updated

    def profiled_step(
        self,
        memory: ClassMemory,
        x_hv: torch.Tensor,
        y: torch.Tensor,
        *,
        update_mask: torch.Tensor | None = None,
        update_basis: torch.Tensor | None = None,
    ) -> tuple[ClassMemory, dict[str, float]]:
        weight = memory.weight.clone()
        num_classes = weight.shape[0]
        num_samples = x_hv.shape[0]
        timings = {
            "updater_shuffle_ms": 0.0,
            "updater_batch_slice_ms": 0.0,
            "updater_similarity_ms": 0.0,
            "updater_error_update_ms": 0.0,
            "updater_normalize_ms": 0.0,
            "updater_step_ms": 0.0,
            "updater_wrong_batches": 0.0,
            "updater_wrong_samples": 0.0,
        }
        if num_samples == 0:
            return ClassMemory(weight=weight), timings

        mask, basis = self._prepare_update_inputs(weight, update_mask=update_mask, update_basis=update_basis)
        total_started = time.perf_counter()

        for _ in range(self.epochs):
            shuffle_started = time.perf_counter()
            order = torch.randperm(num_samples, device=x_hv.device)
            timings["updater_shuffle_ms"] += (time.perf_counter() - shuffle_started) * 1000.0
            for start in range(0, num_samples, self.batch_size):
                batch_started = time.perf_counter()
                idx = order[start:start + self.batch_size]
                x_batch = x_hv[idx]
                y_batch = y[idx].long()
                timings["updater_batch_slice_ms"] += (time.perf_counter() - batch_started) * 1000.0

                similarity_started = time.perf_counter()
                pred = similarity_scores(x_batch, weight, self.metric).argmax(dim=1)
                timings["updater_similarity_ms"] += (time.perf_counter() - similarity_started) * 1000.0
                wrong = pred != y_batch
                if not torch.any(wrong):
                    continue

                timings["updater_wrong_batches"] += 1.0
                timings["updater_wrong_samples"] += float(wrong.sum().item())

                # Mirror HD Zoo's retraining rule:
                # build class-wise +/- masks for misclassified samples, then bundle updates.
                update_started = time.perf_counter()
                wrong_mask = wrong.repeat(num_classes, 1).T
                eye = torch.eye(num_classes, dtype=torch.bool, device=x_batch.device)
                correct_update = wrong_mask & eye[y_batch]
                wrong_update = wrong_mask & eye[pred]
                updates = (correct_update.float() - wrong_update.float()).T @ x_batch
                if mask is not None:
                    updates = updates * mask
                if basis is not None:
                    updates = (updates @ basis.T) @ basis
                weight.add_(updates, alpha=self.lr)
                timings["updater_error_update_ms"] += (time.perf_counter() - update_started) * 1000.0

        normalize_started = time.perf_counter()
        updated = ClassMemory(weight=weight)
        if update_mask is not None:
            updated = updated.normalize_masked_(update_mask)
        else:
            updated = updated.normalize_()
        timings["updater_normalize_ms"] = (time.perf_counter() - normalize_started) * 1000.0
        timings["updater_step_ms"] = (time.perf_counter() - total_started) * 1000.0
        return updated, timings
