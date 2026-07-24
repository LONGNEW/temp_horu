"""Class prototype memory for nonlinear HDC."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class PrototypeMemory:
    """Per-client class memory with cosine prediction and push-pull updates."""

    def __init__(self, memory: torch.Tensor) -> None:
        if memory.ndim != 2 or memory.shape[0] == 0 or memory.shape[1] == 0:
            raise ValueError("memory must be a non-empty rank-2 tensor")
        if memory.dtype != torch.float32:
            raise TypeError("memory must have torch.float32 dtype")
        self.memory = memory

    @classmethod
    def initialize(
        cls,
        encoded: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int,
        normalize_rows: bool = True,
    ) -> "PrototypeMemory":
        _validate_batch(encoded, labels, num_classes)
        memory = torch.zeros((num_classes, encoded.shape[1]), dtype=torch.float32, device=encoded.device)
        for label in range(num_classes):
            values = encoded[labels == label]
            if values.numel():
                memory[label] = values.mean(dim=0)
        if normalize_rows:
            norms = torch.linalg.vector_norm(memory, dim=1, keepdim=True)
            memory = torch.where(norms > 0, memory / norms.clamp_min(torch.finfo(memory.dtype).eps), memory)
        return cls(memory)

    @property
    def num_classes(self) -> int:
        return self.memory.shape[0]

    def predict(self, encoded: torch.Tensor, similarity: str = "cosine") -> torch.Tensor:
        if encoded.ndim != 2 or encoded.shape[0] == 0 or encoded.shape[1] != self.memory.shape[1]:
            raise ValueError("encoded must be a non-empty batch matching memory dimension")
        if encoded.device != self.memory.device:
            raise ValueError("encoded and memory must be on the same device")
        if encoded.dtype != torch.float32:
            raise TypeError("encoded must have torch.float32 dtype")
        if similarity not in {"cosine", "dot"}:
            raise ValueError("similarity must be cosine or dot")
        norms = torch.linalg.vector_norm(self.memory, dim=1)
        if not torch.any(norms > 0):
            raise RuntimeError("cannot predict with all-zero prototype memory")
        if similarity == "cosine":
            query = F.normalize(encoded, p=2, dim=1, eps=torch.finfo(encoded.dtype).eps)
            prototypes = self.memory / norms[:, None].clamp_min(torch.finfo(encoded.dtype).eps)
            scores = query @ prototypes.T
        else:
            scores = encoded @ self.memory.T
        scores[:, norms == 0] = -torch.inf
        return scores.argmax(dim=1)

    def update(
        self,
        encoded: torch.Tensor,
        labels: torch.Tensor,
        learning_rate: float,
        similarity: str = "cosine",
        normalize_hypervectors: bool = False,
    ) -> int:
        _validate_batch(encoded, labels, self.num_classes)
        if encoded.shape[1] != self.memory.shape[1] or encoded.device != self.memory.device:
            raise ValueError("encoded batch does not match memory shape or device")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        updates = 0
        for index in range(encoded.shape[0]):
            predicted = int(self.predict(encoded[index : index + 1], similarity)[0])
            actual = int(labels[index])
            if predicted != actual:
                update_vector = encoded[index]
                if normalize_hypervectors:
                    update_vector = F.normalize(
                        update_vector.unsqueeze(0),
                        p=2,
                        dim=1,
                        eps=torch.finfo(encoded.dtype).eps,
                    )[0]
                self.memory[actual].add_(update_vector, alpha=learning_rate)
                self.memory[predicted].add_(update_vector, alpha=-learning_rate)
                updates += 1
        return updates

    def update_hdzoo_batch(
        self,
        encoded: torch.Tensor,
        labels: torch.Tensor,
        learning_rate: float,
        similarity: str = "dot",
        normalize_hypervectors: bool = False,
    ) -> int:
        _validate_batch(encoded, labels, self.num_classes)
        if encoded.shape[1] != self.memory.shape[1] or encoded.device != self.memory.device:
            raise ValueError("encoded batch does not match memory shape or device")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        predictions = self.predict(encoded, similarity)
        wrong = predictions != labels
        if not torch.any(wrong):
            return 0
        update_vectors = encoded[wrong]
        if normalize_hypervectors:
            update_vectors = F.normalize(update_vectors, p=2, dim=1, eps=torch.finfo(encoded.dtype).eps)
        updates = torch.zeros_like(self.memory)
        updates.index_add_(0, labels[wrong], update_vectors)
        updates.index_add_(0, predictions[wrong], -update_vectors)
        self.memory.add_(updates, alpha=learning_rate)
        return int(wrong.sum().item())


def _validate_batch(encoded: torch.Tensor, labels: torch.Tensor, num_classes: int) -> None:
    if encoded.ndim != 2 or encoded.shape[0] == 0:
        raise ValueError("encoded must be a non-empty rank-2 tensor")
    if labels.ndim != 1 or labels.shape[0] != encoded.shape[0]:
        raise ValueError("labels must be rank-1 and align with encoded")
    if labels.device != encoded.device:
        raise ValueError("labels and encoded must be on the same device")
    if labels.dtype != torch.long:
        raise TypeError("labels must have torch.long dtype")
    if num_classes <= 0 or torch.any(labels < 0) or torch.any(labels >= num_classes):
        raise ValueError("labels are outside class range")
