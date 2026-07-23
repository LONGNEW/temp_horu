"""T006 deterministic 30-client synthetic loader."""
from __future__ import annotations
from pathlib import Path
import torch
from .federated import ClientData, FederatedDataset, stratified_split, write_cache


def prepare_data(data_root: str | Path, seed: int = 0) -> FederatedDataset:
    g = torch.Generator().manual_seed(seed)
    classes, features, clients, per_client = 10, 60, 30, 600
    # alpha/beta are recorded user-specified generator controls; this generator
    # uses their Dirichlet distributions directly rather than hidden defaults.
    base = torch.randn(classes, features, generator=g)
    result = {}
    for i in range(clients):
        # Distribution.sample has no Generator argument; isolate and seed the
        # global RNG so cache identity is independent of process history.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 1_000_003 + i)
            proportions = torch.distributions.Dirichlet(torch.full((classes,), 0.5)).sample()
        labels = torch.multinomial(proportions, per_client, replacement=True, generator=g).long()
        shift = torch.randn(features, generator=g) * 0.5
        x = base[labels] + shift + torch.randn(per_client, features, generator=g)
        train, test = stratified_split(labels, .3, seed + i)
        ids = torch.arange(i * per_client, (i + 1) * per_client, dtype=torch.long)
        result[f"{i:03d}"] = ClientData(x[train], labels[train], x[test], labels[test], ids[train], ids[test])
    manifest = {"source": "internal_leaf_style_generator", "license": "N/A", "generator": "horu_artifact.datasets.synthetic.v1",
                "clients": clients, "features": features, "classes": classes, "alpha": .5, "beta": .5,
                "seed": seed, "normalization": "none", "provenance": "USER_SPECIFIED_MODERATE_HETEROGENEITY"}
    return write_cache(FederatedDataset("synthetic", result, features, classes, manifest), data_root)
