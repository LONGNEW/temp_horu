from __future__ import annotations

import json
from pathlib import Path

import torch

from our_hd import (
    CIFAR10Adapter,
    CIFAR100Adapter,
    EMNISTAdapter,
    FEMNISTAdapter,
    FlambyHeartDiseaseAdapter,
    FlambyTcgaBrcaAdapter,
    HHARAdapter,
    ISOLETAdapter,
    MHEALTHAdapter,
    NinaProDB1Adapter,
    PAMAP2Adapter,
    SyntheticAdapter,
    UCIHARAdapter,
    USCHADAdapter,
    WISDMAdapter,
)


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
    if dataset_name == "emnist":
        return EMNISTAdapter(
            root=dataset_cfg.get("root", "../fisher_FL_hd_unzipped/data/tiers/standard_pfl/emnist"),
            split=dataset_cfg.get("split", "balanced"),
            num_clients=dataset_cfg.get("num_clients", 100),
            alpha=dataset_cfg.get("alpha", 0.5),
            test_size=dataset_cfg.get("test_size", 0.3),
            min_client_samples=dataset_cfg.get("min_client_samples", 10),
            limit_clients=dataset_cfg.get("limit_clients"),
            normalization=resolve_dataset_normalization(dataset_cfg, default="none"),
            max_train_samples_per_client=dataset_cfg.get("max_train_samples_per_client"),
            max_test_samples_per_client=dataset_cfg.get("max_test_samples_per_client"),
            seed=dataset_cfg.get("seed", 13),
            download=dataset_cfg.get("download", True),
            device=device,
        )
    if dataset_name == "flamby_tcga_brca":
        return FlambyTcgaBrcaAdapter(
            root=dataset_cfg.get("root"),
            num_clients=dataset_cfg.get("num_clients", 100),
            alpha=dataset_cfg.get("alpha", 0.5),
            test_size=dataset_cfg.get("test_size", 0.3),
            min_client_samples=dataset_cfg.get("min_client_samples", 10),
            limit_clients=dataset_cfg.get("limit_clients", 100),
            preserve_native_clients=dataset_cfg.get("preserve_native_clients", True),
            normalization=resolve_dataset_normalization(dataset_cfg, default="standardize"),
            max_train_samples_per_client=dataset_cfg.get("max_train_samples_per_client"),
            max_test_samples_per_client=dataset_cfg.get("max_test_samples_per_client"),
            seed=dataset_cfg.get("seed", 13),
            auto_accept_license=dataset_cfg.get("auto_accept_license", False),
            device=device,
        )
    if dataset_name == "flamby_heart_disease":
        return FlambyHeartDiseaseAdapter(
            root=dataset_cfg.get("root"),
            num_clients=dataset_cfg.get("num_clients", 100),
            alpha=dataset_cfg.get("alpha", 0.5),
            test_size=dataset_cfg.get("test_size", 0.3),
            min_client_samples=dataset_cfg.get("min_client_samples", 10),
            limit_clients=dataset_cfg.get("limit_clients", 100),
            preserve_native_clients=dataset_cfg.get("preserve_native_clients", True),
            normalization=resolve_dataset_normalization(dataset_cfg, default="standardize"),
            max_train_samples_per_client=dataset_cfg.get("max_train_samples_per_client"),
            max_test_samples_per_client=dataset_cfg.get("max_test_samples_per_client"),
            seed=dataset_cfg.get("seed", 13),
            auto_accept_license=dataset_cfg.get("auto_accept_license", False),
            device=device,
        )
    if dataset_name == "cifar10":
        return CIFAR10Adapter(
            root=dataset_cfg.get("root", "../fisher_FL_hd_unzipped/data/tiers/standard_pfl/cifar10/cifar-10-batches-py"),
            num_clients=dataset_cfg.get("num_clients", 37),
            alpha=dataset_cfg.get("alpha", 0.5),
            test_size=dataset_cfg.get("test_size", 0.3),
            min_client_samples=dataset_cfg.get("min_client_samples", 10),
            limit_clients=dataset_cfg.get("limit_clients"),
            normalization=resolve_dataset_normalization(dataset_cfg, default="none"),
            max_train_samples_per_client=dataset_cfg.get("max_train_samples_per_client"),
            max_test_samples_per_client=dataset_cfg.get("max_test_samples_per_client"),
            seed=dataset_cfg.get("seed", 13),
            device=device,
        )
    if dataset_name == "cifar100":
        return CIFAR100Adapter(
            root=dataset_cfg.get("root", "../fisher_FL_hd_unzipped/data/tiers/standard_pfl/cifar100/cifar-100-python"),
            label_type=dataset_cfg.get("label_type", "coarse"),
            num_clients=dataset_cfg.get("num_clients", 33),
            alpha=dataset_cfg.get("alpha", 1.0),
            test_size=dataset_cfg.get("test_size", 0.3),
            min_client_samples=dataset_cfg.get("min_client_samples", 10),
            limit_clients=dataset_cfg.get("limit_clients"),
            normalization=resolve_dataset_normalization(dataset_cfg, default="none"),
            max_train_samples_per_client=dataset_cfg.get("max_train_samples_per_client"),
            max_test_samples_per_client=dataset_cfg.get("max_test_samples_per_client"),
            seed=dataset_cfg.get("seed", 13),
            device=device,
        )
    if dataset_name == "pamap2":
        return PAMAP2Adapter(
            root=dataset_cfg.get("root", "../fisher_FL_hd_unzipped/data/tiers/on_device_hdc/pamap2/PAMAP2_Dataset/Protocol"),
            test_size=dataset_cfg.get("test_size", 0.3),
            min_client_samples=dataset_cfg.get("min_client_samples", 20),
            limit_clients=dataset_cfg.get("limit_clients", 8),
            normalization=resolve_dataset_normalization(dataset_cfg, default="standardize"),
            max_train_samples_per_client=dataset_cfg.get("max_train_samples_per_client"),
            max_test_samples_per_client=dataset_cfg.get("max_test_samples_per_client"),
            seed=dataset_cfg.get("seed", 13),
            device=device,
        )
    if dataset_name == "mhealth":
        return MHEALTHAdapter(
            root=dataset_cfg.get("root", "../fisher_FL_hd_unzipped/data/tiers/on_device_hdc/mhealth"),
            split_mode=dataset_cfg.get("split_mode", "per_activity_chrono"),
            window_size=dataset_cfg.get("window_size", 128),
            window_stride=dataset_cfg.get("window_stride", 64),
            strict_windows=dataset_cfg.get("strict_windows", False),
            drop_ecg=dataset_cfg.get("drop_ecg", True),
            train_fraction=dataset_cfg.get("train_fraction", 0.7),
            val_fraction=dataset_cfg.get("val_fraction", 0.1),
            include_val_in_train=dataset_cfg.get("include_val_in_train", False),
            min_client_samples=dataset_cfg.get("min_client_samples", 20),
            limit_clients=dataset_cfg.get("limit_clients", 10),
            normalization=resolve_dataset_normalization(dataset_cfg, default="standardize"),
            max_train_samples_per_client=dataset_cfg.get("max_train_samples_per_client"),
            max_test_samples_per_client=dataset_cfg.get("max_test_samples_per_client"),
            device=device,
        )
    if dataset_name == "usc_had":
        return USCHADAdapter(
            root=dataset_cfg.get("root", "../fisher_FL_hd_unzipped/data/tiers/on_device_hdc/usc_had"),
            window_size=dataset_cfg.get("window_size", 128),
            window_stride=dataset_cfg.get("window_stride", 64),
            train_trials=tuple(dataset_cfg.get("train_trials", [1, 2, 3])),
            val_trials=tuple(dataset_cfg.get("val_trials", [4])),
            test_trials=tuple(dataset_cfg.get("test_trials", [5])),
            include_val_in_train=dataset_cfg.get("include_val_in_train", False),
            strict_windows=dataset_cfg.get("strict_windows", False),
            min_client_samples=dataset_cfg.get("min_client_samples", 20),
            limit_clients=dataset_cfg.get("limit_clients", 14),
            normalization=resolve_dataset_normalization(dataset_cfg, default="standardize"),
            max_train_samples_per_client=dataset_cfg.get("max_train_samples_per_client"),
            max_test_samples_per_client=dataset_cfg.get("max_test_samples_per_client"),
            device=device,
        )
    if dataset_name == "hhar":
        return HHARAdapter(
            root=dataset_cfg.get("root", "../fisher_FL_hd_unzipped/data/tiers/on_device_hdc/hhar"),
            hhar_source=dataset_cfg.get("hhar_source", "watch"),
            client_mode=dataset_cfg.get("client_mode", "user"),
            window_size=dataset_cfg.get("window_size", 128),
            window_stride=dataset_cfg.get("window_stride", 64),
            strict_windows=dataset_cfg.get("strict_windows", False),
            train_fraction=dataset_cfg.get("train_fraction", 0.7),
            val_fraction=dataset_cfg.get("val_fraction", 0.1),
            include_val_in_train=dataset_cfg.get("include_val_in_train", False),
            resample_hz=dataset_cfg.get("resample_hz", 50.0),
            min_client_samples=dataset_cfg.get("min_client_samples", 20),
            limit_clients=dataset_cfg.get("limit_clients", 9),
            normalization=resolve_dataset_normalization(dataset_cfg, default="standardize"),
            max_train_samples_per_client=dataset_cfg.get("max_train_samples_per_client"),
            max_test_samples_per_client=dataset_cfg.get("max_test_samples_per_client"),
            device=device,
        )
    if dataset_name == "wisdm":
        return WISDMAdapter(
            root=dataset_cfg.get("root", "../fisher_FL_hd_unzipped/data/tiers/on_device_hdc/wisdm"),
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
            root=dataset_cfg.get("root", "../fisher_FL_hd_unzipped/data/tiers/on_device_hdc/ninapro_db1"),
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
            root=dataset_cfg.get("root", "data/leaf_synthetic/data"),
            limit_clients=dataset_cfg.get("limit_clients"),
            min_client_samples=dataset_cfg.get("min_client_samples", 10),
            normalization=resolve_dataset_normalization(dataset_cfg, default="none"),
            max_train_samples_per_client=dataset_cfg.get("max_train_samples_per_client"),
            max_test_samples_per_client=dataset_cfg.get("max_test_samples_per_client"),
            seed=dataset_cfg.get("seed", 13),
            device=device,
        )
    raise ValueError(f"Unsupported dataset for now: {dataset_name}")
