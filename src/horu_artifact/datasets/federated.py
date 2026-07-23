"""Portable, hash-validated federated dataset cache contract.

The T006 suite never infers a split at training time.  Every loader writes the
same immutable cache format so all methods consume identical client order,
sample ids, and train/test membership.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def _hash_bytes(*parts: bytes) -> str:
    h = hashlib.sha256()
    for part in parts: h.update(part)
    return h.hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    return _hash_bytes(str(value.dtype).encode(), str(tuple(value.shape)).encode(), value.numpy().tobytes())


@dataclass
class ClientData:
    train_x: torch.Tensor
    train_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    train_ids: torch.Tensor
    test_ids: torch.Tensor

    def validate(self, features: int, classes: int) -> None:
        for x, y, ids in ((self.train_x, self.train_y, self.train_ids), (self.test_x, self.test_y, self.test_ids)):
            if x.ndim != 2 or x.shape[1] != features or y.ndim != 1 or ids.ndim != 1:
                raise ValueError("invalid federated client tensor shape")
            if x.shape[0] != y.numel() or y.numel() != ids.numel() or not torch.isfinite(x).all():
                raise ValueError("invalid federated client tensors")
            if y.numel() and (int(y.min()) < 0 or int(y.max()) >= classes):
                raise ValueError("client labels are outside class range")


@dataclass
class FederatedDataset:
    name: str
    clients: dict[str, ClientData]
    num_features: int
    num_classes: int
    manifest: dict[str, Any]

    def validate(self) -> None:
        if not self.clients or list(self.clients) != sorted(self.clients):
            raise ValueError("client ids must be nonempty and lexically ordered")
        for client in self.clients.values(): client.validate(self.num_features, self.num_classes)
        expected = self.split_hash()
        if self.manifest.get("split_sha256") and self.manifest["split_sha256"] != expected:
            raise ValueError("cache split hash does not match tensors")

    def split_hash(self) -> str:
        parts: list[bytes] = [self.name.encode(), str(self.num_features).encode(), str(self.num_classes).encode()]
        for cid, client in self.clients.items():
            parts.extend([cid.encode(), tensor_hash(client.train_ids).encode(), tensor_hash(client.test_ids).encode()])
        return _hash_bytes(*parts)

    def statistics(self) -> dict[str, Any]:
        return {"dataset": self.name, "clients": len(self.clients), "features": self.num_features,
                "classes": self.num_classes, "train_samples": sum(x.train_y.numel() for x in self.clients.values()),
                "test_samples": sum(x.test_y.numel() for x in self.clients.values()), "split_sha256": self.split_hash(),
                "client_sizes": {cid: {"train": c.train_y.numel(), "test": c.test_y.numel()} for cid,c in self.clients.items()}}


def write_cache(dataset: FederatedDataset, data_root: str | Path) -> FederatedDataset:
    dataset.manifest = dict(dataset.manifest)
    dataset.manifest["split_sha256"] = dataset.split_hash()
    dataset.validate()
    root = Path(data_root) / dataset.name
    (root / "processed").mkdir(parents=True, exist_ok=True)
    payload = {"name": dataset.name, "clients": dataset.clients, "num_features": dataset.num_features,
               "num_classes": dataset.num_classes, "manifest": dataset.manifest}
    torch.save(payload, root / "processed" / "federated.pt")
    (root / "manifest.json").write_text(json.dumps(dataset.manifest, indent=2, sort_keys=True) + "\n")
    (root / "statistics.json").write_text(json.dumps(dataset.statistics(), indent=2, sort_keys=True) + "\n")
    return dataset


def load_federated(data_root: str | Path, name: str) -> FederatedDataset:
    path = Path(data_root) / name / "processed" / "federated.pt"
    if not path.is_file(): raise FileNotFoundError(f"prepared {name} cache is absent: {path}")
    dataset = FederatedDataset(**torch.load(path, map_location="cpu", weights_only=False))
    dataset.validate()
    return dataset


def stratified_split(labels: torch.Tensor, test_ratio: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic class-wise split; a singleton remains train-only."""
    train, test = [], []
    for label in torch.unique(labels, sorted=True).tolist():
        ix = torch.nonzero(labels == label, as_tuple=False).flatten()
        if ix.numel() < 2: train.append(ix); continue
        g = torch.Generator().manual_seed(seed + int(label) * 104729)
        ix = ix[torch.randperm(ix.numel(), generator=g)]
        ntest = max(1, min(ix.numel() - 1, round(ix.numel() * test_ratio)))
        test.append(ix[:ntest]); train.append(ix[ntest:])
    return torch.cat(train).sort().values, torch.cat(test).sort().values
