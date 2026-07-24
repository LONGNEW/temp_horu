from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from our_hd import ClientData, CosineProjectionEncoder, LocalHDUpdater
from our_hd.methods import FedHDCMethod, HoRUMethod, HyperFeelMethod


HD_METHODS = ["horu_hd", "fedhdc", "hyperfeel"]
RECONSTRUCTION_DATASETS = [
    "uci_har",
    "isolet_raw",
    "femnist",
    "wisdm",
    "synthetic",
    "ninapro_db1",
]

SOURCE_DATA_ROOT = Path(
    os.environ.get(
        "HORU_SOURCE_DATA_ROOT",
        os.environ.get("LONGNEW_DATA_ROOT", "/home/longnew/data") + "/datasets/horu-paper-main/source",
    )
)

DATASET_ROOTS = {
    "uci_har": str(SOURCE_DATA_ROOT / "data/tiers/on_device_hdc/uci_har/UCI HAR Dataset"),
    "isolet_raw": str(SOURCE_DATA_ROOT / "data/raw/isolet"),
    "femnist": str(SOURCE_DATA_ROOT / "data/tiers/standard_pfl/femnist"),
    "wisdm": str(SOURCE_DATA_ROOT / "data/tiers/on_device_hdc/wisdm"),
    "synthetic": str(SOURCE_DATA_ROOT / "data/leaf_synthetic/data"),
    "ninapro_db1": str(SOURCE_DATA_ROOT / "data/tiers/on_device_hdc/ninapro_db1"),
}

DATASET_NORMALIZATION_OVERRIDES = {
    "uci_har": "l2",
    "isolet_raw": "l2",
    "femnist": "l2",
    "wisdm": "standardize",
    "synthetic": "none",
    "ninapro_db1": "standardize",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset_specs() -> dict[str, dict]:
    with open("configs/dataset_specs.json", "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    specs = {
        item["name"]: copy.deepcopy(item)
        for item in payload["datasets"]
        if item["name"] in RECONSTRUCTION_DATASETS
    }
    for name, spec in specs.items():
        spec["root"] = DATASET_ROOTS[name]
        spec["normalization"] = DATASET_NORMALIZATION_OVERRIDES[name]
        if name == "uci_har":
            spec["preserve_original_split"] = True
        if name == "isolet_raw":
            spec["seed"] = 13
        if name == "femnist":
            spec["cache_limit_clients"] = max(int(spec.get("limit_clients", 200)), 200)
            spec["cache_dir"] = "cache/femnist"
            spec["selection_seed"] = 13
            spec["max_train_samples_per_client"] = int(spec.get("max_train_samples_per_client", 256))
            spec["preserve_original_split"] = True
        if name == "wisdm":
            spec["archive"] = str(Path(spec["root"]) / "wisdm-dataset.zip")
            spec["seed"] = 13
        if name in {"synthetic", "ninapro_db1"}:
            spec["seed"] = 13
    return specs


def apply_seed_to_dataset(dataset_cfg: dict, seed: int) -> dict:
    dataset_cfg = copy.deepcopy(dataset_cfg)
    if dataset_cfg["name"] in {"wisdm", "isolet_raw", "synthetic", "ninapro_db1"}:
        dataset_cfg["seed"] = seed
    if dataset_cfg["name"] == "femnist":
        dataset_cfg["selection_seed"] = seed
    return dataset_cfg


def build_config(dataset_cfg: dict, method_key: str, args: object) -> dict:
    if method_key not in HD_METHODS:
        raise ValueError(f"Unsupported reconstruction method: {method_key}")
    method_name = "horu" if method_key == "horu_hd" else method_key
    method_cfg: dict[str, object] = {"name": method_name}
    if method_key == "horu_hd":
        method_cfg.update(
            {
                "shared_rank": int(args.subspace_shared_rank),
                "personal_rank": int(args.subspace_personal_rank),
                "val_fraction": float(args.subspace_val_fraction),
                "alpha_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
                "gate_alpha": float(args.subspace_rowgate_alpha),
                "gate_min": float(args.subspace_rowgate_min),
                "gate_max": float(args.subspace_rowgate_max),
                "intersection_rank": int(args.subspace_intersection_rank),
                "enable_wasserstein_sync": bool(getattr(args, "enable_wasserstein_sync", False)),
                "wasserstein_atoms": int(getattr(args, "wasserstein_atoms", 3)),
                "wasserstein_beta": float(getattr(args, "wasserstein_beta", 0.0)),
                "wasserstein_max_iters": int(getattr(args, "wasserstein_max_iters", 20)),
                "wasserstein_interval": int(getattr(args, "wasserstein_interval", 1)),
            }
        )
    return {
        "dataset": copy.deepcopy(dataset_cfg),
        "model": {
            "encoder": "cosine_projection",
            "hd_dim": int(args.hd_dim),
            "binary": False,
            "cosine_random_phase": bool(getattr(args, "hd_cosine_random_phase", False)),
            "metric": "cos",
        },
        "method": method_cfg,
        "train": {
            "rounds": int(args.rounds),
            "local_epochs": int(args.local_epochs),
            "client_participation": float(args.client_participation),
            "batch_size": int(args.batch_size),
            "lr": float(args.hd_lr),
        },
        "runtime": {"device": args.device},
    }


def build_hd_method(cfg: dict, input_dim: int, num_classes: int, device: torch.device):
    method_cfg = cfg["method"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    method_name = str(method_cfg["name"])
    encoder = CosineProjectionEncoder(
        input_dim=input_dim,
        hd_dim=int(model_cfg["hd_dim"]),
        binary=bool(model_cfg.get("binary", False)),
        random_phase=bool(model_cfg.get("cosine_random_phase", False)),
        device=device,
    )
    metric = str(model_cfg.get("metric", "cos"))
    updater = LocalHDUpdater(
        epochs=int(train_cfg["local_epochs"]),
        batch_size=int(train_cfg["batch_size"]),
        lr=float(train_cfg["lr"]),
        metric=metric,
    )
    if method_name == "horu":
        method = HoRUMethod(
            encoder=encoder,
            num_classes=num_classes,
            shared_rank=int(method_cfg.get("shared_rank", 32)),
            personal_rank=int(method_cfg.get("personal_rank", 64)),
            local_epochs=int(train_cfg["local_epochs"]),
            batch_size=int(train_cfg["batch_size"]),
            global_lr=float(train_cfg["lr"]),
            personal_lr=float(train_cfg["lr"]),
            val_fraction=float(method_cfg.get("val_fraction", 0.0)),
            alpha_grid=tuple(float(value) for value in method_cfg.get("alpha_grid", [0.0])),
            gate_alpha=float(method_cfg.get("gate_alpha", 1.0)),
            gate_min=float(method_cfg.get("gate_min", 0.1)),
            gate_max=float(method_cfg.get("gate_max", 0.9)),
            intersection_rank=int(method_cfg.get("intersection_rank", 24)),
            enable_wasserstein_sync=bool(method_cfg.get("enable_wasserstein_sync", False)),
            wasserstein_atoms=int(method_cfg.get("wasserstein_atoms", 3)),
            wasserstein_beta=float(method_cfg.get("wasserstein_beta", 0.0)),
            wasserstein_max_iters=int(method_cfg.get("wasserstein_max_iters", 20)),
            wasserstein_interval=int(method_cfg.get("wasserstein_interval", 1)),
        )
        return method, "mean_personalized_accuracy"
    if method_name == "fedhdc":
        method = FedHDCMethod(
            encoder=encoder,
            updater=updater,
            num_classes=num_classes,
            metric=metric,
            debug=bool(method_cfg.get("debug", False)),
        )
        return method, "global_test_accuracy"
    if method_name == "hyperfeel":
        method = HyperFeelMethod(
            encoder=encoder,
            num_classes=num_classes,
            local_epochs=int(train_cfg["local_epochs"]),
            batch_size=int(train_cfg["batch_size"]),
            lr=float(train_cfg["lr"]),
            metric=metric,
            debug=bool(method_cfg.get("debug", False)),
        )
        return method, "mean_personalized_accuracy"
    raise ValueError(f"Unsupported reconstruction method name: {method_name}")


def mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std())}


def total_train_samples(clients: list[ClientData]) -> int:
    return int(sum(int(client.y_train.numel()) for client in clients))


def allocate_train_sample_budget(train_sizes: list[int], total_budget: int) -> list[int]:
    if not train_sizes:
        return []
    total_budget = max(0, int(total_budget))
    total_available = int(sum(train_sizes))
    if total_budget >= total_available:
        return [int(size) for size in train_sizes]
    weights = np.asarray(train_sizes, dtype=np.float64)
    raw = weights / max(float(weights.sum()), 1.0) * float(total_budget)
    allocation = np.floor(raw).astype(int)
    allocation = np.minimum(allocation, weights.astype(int))
    positive = weights > 0
    if total_budget >= int(positive.sum()):
        allocation = np.where((positive) & (allocation == 0), 1, allocation)
        allocation = np.minimum(allocation, weights.astype(int))
    while int(allocation.sum()) > total_budget:
        reducible = np.where(allocation > 1)[0]
        if reducible.size == 0:
            reducible = np.where(allocation > 0)[0]
            if reducible.size == 0:
                break
        target = reducible[np.argmax(allocation[reducible])]
        allocation[target] -= 1
    fractional = raw - np.floor(raw)
    remaining_budget = total_budget - int(allocation.sum())
    while remaining_budget > 0:
        growable = np.where(allocation < weights)[0]
        if growable.size == 0:
            break
        target = growable[np.argmax(fractional[growable])]
        allocation[target] += 1
        fractional[target] = 0.0
        remaining_budget -= 1
    return allocation.astype(int).tolist()


def stratified_subsample_client_train_data(client: ClientData, max_samples: int, *, seed: int) -> ClientData:
    max_samples = int(max_samples)
    train_size = int(client.y_train.numel())
    if max_samples <= 0 or train_size <= max_samples:
        return client
    y_cpu = client.y_train.detach().cpu()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    classes, counts = torch.unique(y_cpu, sorted=True, return_counts=True)
    if int(classes.numel()) == 0:
        return client
    if max_samples >= int(classes.numel()):
        proportions = counts.to(torch.float64) / float(counts.sum().item())
        allocation = torch.floor(proportions * float(max_samples)).to(torch.long)
        allocation = torch.maximum(allocation, torch.ones_like(allocation))
        allocation = torch.minimum(allocation, counts)
        while int(allocation.sum().item()) > max_samples:
            reducible = torch.nonzero(allocation > 1, as_tuple=False).squeeze(1)
            if reducible.numel() == 0:
                reducible = torch.nonzero(allocation > 0, as_tuple=False).squeeze(1)
                if reducible.numel() == 0:
                    break
            target = reducible[torch.argmax(allocation[reducible])]
            allocation[target] -= 1
        while int(allocation.sum().item()) < max_samples:
            remaining = counts - allocation
            growable = torch.nonzero(remaining > 0, as_tuple=False).squeeze(1)
            if growable.numel() == 0:
                break
            target = growable[torch.argmax(remaining[growable])]
            allocation[target] += 1
        selected_parts: list[torch.Tensor] = []
        for cls, take in zip(classes.tolist(), allocation.tolist()):
            if int(take) <= 0:
                continue
            cls_idx = torch.nonzero(y_cpu == int(cls), as_tuple=False).squeeze(1)
            permutation = torch.randperm(int(cls_idx.numel()), generator=generator)
            selected_parts.append(cls_idx[permutation[: int(take)]])
        chosen = torch.sort(torch.cat(selected_parts, dim=0)).values
    else:
        chosen = torch.sort(torch.randperm(train_size, generator=generator)[:max_samples]).values
    x_index = chosen.to(device=client.x_train.device)
    y_index = chosen.to(device=client.y_train.device)
    return ClientData(
        client_id=client.client_id,
        x_train=client.x_train.index_select(0, x_index),
        y_train=client.y_train.index_select(0, y_index),
        x_test=client.x_test,
        y_test=client.y_test,
    )


def maybe_cap_large_dataset_train_data(
    clients: list[ClientData],
    *,
    threshold: int | None,
    total_cap: int | None,
    seed: int,
) -> tuple[list[ClientData], dict[str, int | bool | None]]:
    total_before = total_train_samples(clients)
    info: dict[str, int | bool | None] = {
        "applied": False,
        "activation_threshold": None if threshold is None else int(threshold),
        "target_total_train_samples": None if total_cap is None else int(total_cap),
        "total_train_samples_before": int(total_before),
        "total_train_samples_after": int(total_before),
        "clients_modified": 0,
    }
    if threshold is None or total_cap is None or total_before <= int(threshold):
        return clients, info
    allocation = allocate_train_sample_budget(
        [int(client.y_train.numel()) for client in clients],
        int(total_cap),
    )
    capped_clients = [
        stratified_subsample_client_train_data(client, allocation[idx], seed=int(seed) + (idx * 10007))
        for idx, client in enumerate(clients)
    ]
    total_after = total_train_samples(capped_clients)
    info.update(
        {
            "applied": True,
            "total_train_samples_after": int(total_after),
            "clients_modified": int(
                sum(allocation[idx] < int(client.y_train.numel()) for idx, client in enumerate(clients))
            ),
        }
    )
    return capped_clients, info
