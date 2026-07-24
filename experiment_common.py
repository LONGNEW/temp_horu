from __future__ import annotations

import json
from pathlib import Path

import torch

from our_hd import FEMNISTAdapter, ISOLETAdapter, NinaProDB1Adapter, SyntheticAdapter, UCIHARAdapter, WISDMAdapter


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    return torch.device(device_name)


def resolve_dataset_normalization(dataset_cfg: dict, *, default: str) -> str:
    if dataset_cfg.get("normalization") is not None:
        return str(dataset_cfg["normalization"])
    if dataset_cfg.get("standardize") is not None:
        return "standardize" if bool(dataset_cfg["standardize"]) else "none"
    return default


def build_dataset_adapter(cfg: dict, device: torch.device):
    dataset_cfg = cfg["dataset"]
    dataset_name = dataset_cfg["name"]
    if dataset_name == "uci_har":
        return UCIHARAdapter(
            root=dataset_cfg["root"],
            test_size=dataset_cfg.get("test_size", 0.3),
            min_client_samples=dataset_cfg.get("min_client_samples", 10),
            limit_clients=dataset_cfg.get("limit_clients"),
            normalization=resolve_dataset_normalization(dataset_cfg, default="l2"),
            max_train_samples_per_client=dataset_cfg.get("max_train_samples_per_client"),
            max_test_samples_per_client=dataset_cfg.get("max_test_samples_per_client"),
            preserve_original_split=dataset_cfg.get("preserve_original_split", False),
            device=device,
        )
    if dataset_name == "isolet_raw":
        return ISOLETAdapter(
            root=dataset_cfg["root"],
            num_clients=dataset_cfg.get("num_clients", 8),
            alpha=dataset_cfg.get("alpha", 5.0),
            test_size=dataset_cfg.get("test_size", 0.3),
            min_client_samples=dataset_cfg.get("min_client_samples", 10),
            limit_clients=dataset_cfg.get("limit_clients"),
            normalization=resolve_dataset_normalization(dataset_cfg, default="l2"),
            max_train_samples_per_client=dataset_cfg.get("max_train_samples_per_client"),
            max_test_samples_per_client=dataset_cfg.get("max_test_samples_per_client"),
            preserve_original_split=dataset_cfg.get("preserve_original_split", False),
            seed=dataset_cfg.get("seed", 13),
            device=device,
        )
    if dataset_name == "wisdm":
        return WISDMAdapter(
            root=dataset_cfg["root"],
            wisdm_modality=dataset_cfg.get("wisdm_modality", "phone_accel"),
            test_size=dataset_cfg.get("test_size", 0.3),
            min_client_samples=dataset_cfg.get("min_client_samples", 20),
            limit_clients=dataset_cfg.get("limit_clients", 51),
            normalization=resolve_dataset_normalization(dataset_cfg, default="standardize"),
            max_train_samples_per_client=dataset_cfg.get("max_train_samples_per_client"),
            max_test_samples_per_client=dataset_cfg.get("max_test_samples_per_client"),
            seed=dataset_cfg.get("seed", 13),
            device=device,
        )
    if dataset_name == "ninapro_db1":
        return NinaProDB1Adapter(
            root=dataset_cfg["root"],
            ninapro_modality=dataset_cfg.get("ninapro_modality", "emg_glove"),
            test_size=dataset_cfg.get("test_size", 0.3),
            min_client_samples=dataset_cfg.get("min_client_samples", 20),
            limit_clients=dataset_cfg.get("limit_clients", 27),
            normalization=resolve_dataset_normalization(dataset_cfg, default="standardize"),
            max_train_samples_per_client=dataset_cfg.get("max_train_samples_per_client"),
            max_test_samples_per_client=dataset_cfg.get("max_test_samples_per_client"),
            seed=dataset_cfg.get("seed", 13),
            device=device,
        )
    if dataset_name == "femnist":
        return FEMNISTAdapter(
            root=dataset_cfg["root"],
            limit_clients=dataset_cfg.get("limit_clients", 200),
            cache_limit_clients=dataset_cfg.get("cache_limit_clients"),
            cache_dir=dataset_cfg.get("cache_dir", "cache/femnist"),
            selection_seed=dataset_cfg.get("selection_seed"),
            min_client_samples=dataset_cfg.get("min_client_samples", 20),
            normalization=resolve_dataset_normalization(dataset_cfg, default="l2"),
            max_train_samples_per_client=dataset_cfg.get("max_train_samples_per_client"),
            max_test_samples_per_client=dataset_cfg.get("max_test_samples_per_client"),
            device=device,
        )
    if dataset_name == "synthetic":
        return SyntheticAdapter(
            root=dataset_cfg["root"],
            limit_clients=dataset_cfg.get("limit_clients"),
            min_client_samples=dataset_cfg.get("min_client_samples", 10),
            normalization=resolve_dataset_normalization(dataset_cfg, default="none"),
            max_train_samples_per_client=dataset_cfg.get("max_train_samples_per_client"),
            max_test_samples_per_client=dataset_cfg.get("max_test_samples_per_client"),
            seed=dataset_cfg.get("seed", 13),
            device=device,
        )
    raise ValueError(f"Unsupported reconstruction dataset: {dataset_name}")
