"""ISOLET parser with the T006 Dirichlet client partition."""
from __future__ import annotations
from pathlib import Path
import hashlib
import torch
from .federated import ClientData, FederatedDataset, stratified_split, write_cache


def _read(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = [float(x) for x in line.strip().split(",")]
        if len(values) != 618: raise ValueError(f"ISOLET row does not have 617 features: {path}")
        rows.append(values)
    data = torch.tensor(rows, dtype=torch.float32)
    return data[:, :-1], data[:, -1].long() - 1


def _partition(labels: torch.Tensor, seed: int) -> list[torch.Tensor]:
    # Rejection has a fixed seed-derived sequence; attempt count is recorded.
    for attempt in range(10000):
        g = torch.Generator().manual_seed(seed + attempt)
        buckets = [[] for _ in range(8)]
        for y in range(26):
            ix = torch.nonzero(labels == y, as_tuple=False).flatten()
            order = ix[torch.randperm(ix.numel(), generator=g)]
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + attempt * 1009 + y)
                weights = torch.distributions.Dirichlet(torch.full((8,), .05)).sample()
            counts = torch.multinomial(weights, order.numel(), replacement=True, generator=g).bincount(minlength=8)
            cursor = 0
            for client, count in enumerate(counts.tolist()): buckets[client].append(order[cursor:cursor + count]); cursor += count
        output = [torch.cat(x) for x in buckets]
        if all(x.numel() >= 50 and torch.unique(labels[x]).numel() >= 2 for x in output): return output, attempt
    raise RuntimeError("ISOLET partition did not meet minimum client constraints in 10,000 attempts")


def prepare_data(data_root: str | Path, source_root: str | Path, seed: int = 0) -> FederatedDataset:
    source = Path(source_root)
    a, b = source / "isolet1+2+3+4.data", source / "isolet5.data"
    if not a.is_file() or not b.is_file(): raise FileNotFoundError("ISOLET requires isolet1+2+3+4.data and isolet5.data")
    xa, ya = _read(a); xb, yb = _read(b); x, y = torch.cat([xa, xb]), torch.cat([ya, yb])
    if x.shape != (7797, 617) or torch.unique(y).numel() != 26: raise ValueError("ISOLET official dimensions mismatch")
    x = torch.nn.functional.normalize(x, p=2, dim=1)
    partitions, attempt = _partition(y, seed)
    clients = {}
    histograms = {}
    for i, indices in enumerate(partitions):
        train, test = stratified_split(y[indices], .3, seed + i)
        ids = indices.long()
        clients[f"{i:03d}"] = ClientData(x[ids[train]], y[ids[train]], x[ids[test]], y[ids[test]], ids[train], ids[test])
        histograms[str(i)] = torch.bincount(y[ids], minlength=26).tolist()
    digest = hashlib.sha256(a.read_bytes() + b.read_bytes()).hexdigest()
    manifest = {"source": str(source), "license": "UCI ISOLET", "raw_sha256": digest, "parser": "isolet_csv_v1",
                "clients": 8, "features": 617, "classes": 26, "partition": "classwise_dirichlet", "alpha": .05,
                "seed": seed, "partition_attempt": attempt, "client_class_histogram": histograms, "normalization": "samplewise_l2",
                "provenance": "USER_SPECIFIED_DIRICHLET_SPLIT"}
    return write_cache(FederatedDataset("isolet", clients, 617, 26, manifest), data_root)
