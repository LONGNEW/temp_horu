from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from experiment_common import build_dataset_adapter, resolve_device
from hardware_energy import RunEnergyMonitor
from our_hd import (
    ClientData,
    CosineProjectionEncoder,
    FederatedRunner,
    GroupLearnedCosineProjectionEncoder,
    LocalHDUpdater,
    MaskedCosineProjectionEncoder,
    PackedAdditiveCosineEncoder,
    ResidualPackedCosineEncoder,
)
from our_hd.methods import (
    CentralizedHDMethod,
    FedHDCMethod,
    HoRUEGLiteMethod,
    HoRUMethod,
    HyperFeelMethod,
    LocalHDMethod,
    MLPGroupLearnedHDMethod,
    MaskingAntiCollapseHDMethod,
    NaiveGroupEnsembleHDMethod,
    PackedAdditiveHDMethod,
    ResidualPackedHDMethod,
)
from our_nn import NNFederatedRunner
from our_nn.methods import (
    DFLCNNMethod,
    DFLMLPMethod,
    DittoCNNMethod,
    DittoMLPMethod,
    FedAvgCNNMethod,
    FedAvgMLPMethod,
    FedProxCNNMethod,
    FedProxMLPMethod,
    PFedMeCNNMethod,
    PFedMeMLPMethod,
)

CANONICAL_COMMONBASIS_METHOD_KEY = "horu_hd"
CANONICAL_COMMONBASIS_METHOD_NAME = "horu"

COMMONBASIS_METHOD_SPECS: dict[str, dict[str, object]] = {
    CANONICAL_COMMONBASIS_METHOD_KEY: {
        "name": CANONICAL_COMMONBASIS_METHOD_NAME,
        "class": HoRUMethod,
    },
    "horu_eg_lite_hd": {
        "name": "horu_eg_lite",
        "class": HoRUEGLiteMethod,
        "method_defaults": {
            "eg_group_preset": "auto",
            "eg_num_groups": 4,
            "eg_weight_temperature": 0.35,
            "eg_weight_prior_blend": 0.5,
            "eg_enable_interactions": True,
            "eg_interaction_weight": 0.10,
            "eg_interaction_pairs": ["hand+ankle", "chest+ankle"],
        },
    },
}
COMMONBASIS_METHOD_KEYS = tuple(COMMONBASIS_METHOD_SPECS.keys())
COMMONBASIS_METHOD_NAMES = {
    str(spec["name"])
    for spec in COMMONBASIS_METHOD_SPECS.values()
}
COMMONBASIS_METHOD_CLASS_BY_NAME = {
    str(spec["name"]): spec["class"]
    for spec in COMMONBASIS_METHOD_SPECS.values()
}

HD_METHODS = [
    *COMMONBASIS_METHOD_KEYS,
]

OPTIONAL_HD_METHODS = [
    "local_hd",
    "fedhdc",
    "hyperfeel",
    "centralized_hd",
    "masking_anticollapse_hd",
    "mlp_group_learned_hd",
    "block_packed_group_hd",
    "superposition_packed_group_hd",
    "hash_packed_group_hd",
    "naive_group_ensemble",
    "residual_packed_group_hd",
]

NN_METHODS = [
    "fedavg_mlp",
    "fedavg_cnn",
    "fedprox_mlp",
    "fedprox_cnn",
    "ditto_mlp",
    "ditto_cnn",
    "pfedme_mlp",
    "pfedme_cnn",
    "dfl_mlp",
    "dfl_cnn",
]

ALL_METHODS = [*HD_METHODS, *NN_METHODS]
METHOD_CHOICES = [*HD_METHODS, *OPTIONAL_HD_METHODS, *NN_METHODS]

SOURCE_DATA_ROOT = Path(
    os.environ.get(
        "HORU_SOURCE_DATA_ROOT",
        os.environ.get("LONGNEW_DATA_ROOT", "/home/longnew/data") + "/datasets/horu-paper-main/source",
    )
)

DATASET_ROOTS = {
    "uci_har": str(SOURCE_DATA_ROOT / "data/tiers/on_device_hdc/uci_har/UCI HAR Dataset"),
    "wisdm": str(SOURCE_DATA_ROOT / "data/tiers/on_device_hdc/wisdm"),
    "isolet_raw": str(SOURCE_DATA_ROOT / "data/raw/isolet"),
    "ninapro_db1": str(SOURCE_DATA_ROOT / "data/tiers/on_device_hdc/ninapro_db1"),
    "emnist": str(SOURCE_DATA_ROOT / "data/tiers/standard_pfl/emnist"),
    "flamby_tcga_brca": "external/FLamby/flamby/datasets/fed_tcga_brca",
    "flamby_heart_disease": "external/FLamby/flamby/datasets/fed_heart_disease/dataset_creation_scripts/heart_disease_dataset",
    "cifar10": str(SOURCE_DATA_ROOT / "data/tiers/standard_pfl/cifar10/cifar-10-batches-py"),
    "cifar100": str(SOURCE_DATA_ROOT / "data/tiers/standard_pfl/cifar100/cifar-100-python"),
    "pamap2": str(SOURCE_DATA_ROOT / "data/tiers/on_device_hdc/pamap2/PAMAP2_Dataset/Protocol"),
    "mhealth": str(SOURCE_DATA_ROOT / "data/tiers/on_device_hdc/mhealth"),
    "usc_had": str(SOURCE_DATA_ROOT / "data/tiers/on_device_hdc/usc_had"),
    "hhar": str(SOURCE_DATA_ROOT / "data/tiers/on_device_hdc/hhar"),
    "femnist": str(SOURCE_DATA_ROOT / "data/tiers/standard_pfl/femnist"),
    "synthetic": str(SOURCE_DATA_ROOT / "data/leaf_synthetic/data"),
}

DATASET_NORMALIZATION_OVERRIDES = {
    "uci_har": "l2",
    "wisdm": "standardize",
    "isolet_raw": "l2",
    "ninapro_db1": "standardize",
    "emnist": "none",
    "flamby_tcga_brca": "standardize",
    "flamby_heart_disease": "standardize",
    "cifar10": "none",
    "cifar100": "none",
    "pamap2": "standardize",
    "mhealth": "standardize",
    "usc_had": "standardize",
    "hhar": "standardize",
    "femnist": "l2",
    "synthetic": "none",
}

SUBSPACE_METHOD_KEYS = {
    *COMMONBASIS_METHOD_KEYS,
}


def is_commonbasis_method_key(method_key: str) -> bool:
    return method_key in COMMONBASIS_METHOD_SPECS


def is_commonbasis_method_name(method_name: str) -> bool:
    return method_name in COMMONBASIS_METHOD_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the canonical 7-dataset HD/NN benchmark matrix.")
    parser.add_argument(
        "--analysis",
        choices=["benchmark", "shared_subspace"],
        default="benchmark",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["uci_har", "wisdm", "isolet_raw", "ninapro_db1", "pamap2", "femnist"],
    )
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=METHOD_CHOICES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[13])
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hd-dim", type=int, default=5000)
    parser.add_argument("--hd-lr", type=float, default=0.01)
    parser.add_argument("--hd-cosine-random-phase", action="store_true")
    parser.add_argument("--nn-lr", type=float, default=0.001)
    parser.add_argument("--fedprox-mu", type=float, default=0.01)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-rounds", type=int, default=5)
    parser.add_argument("--pilot-local-epochs", type=int, default=1)
    parser.add_argument("--pilot-max-clients", type=int, default=12)
    parser.add_argument("--collapse-factor", type=float, default=1.5)
    parser.add_argument("--json-out", default="results/seven_dataset_benchmark_latest.json")
    parser.add_argument("--md-out", default="results/seven_dataset_benchmark_latest.md")
    parser.add_argument("--shared-ranks", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--encoder-seeds", nargs="+", type=int, default=[13])
    parser.add_argument("--null-repeats", type=int, default=3)
    parser.add_argument("--analysis-out-dir", default="results/shared_subspace_existence")
    parser.add_argument("--subspace-shared-rank", type=int, default=32)
    parser.add_argument("--subspace-personal-rank", type=int, default=64)
    parser.add_argument("--subspace-val-fraction", type=float, default=0.0)
    parser.add_argument("--subspace-fusion-alpha", type=float, default=None)
    parser.add_argument("--subspace-rowgate-alpha", type=float, default=1.0)
    parser.add_argument("--subspace-rowgate-min", type=float, default=0.1)
    parser.add_argument("--subspace-rowgate-max", type=float, default=0.9)
    parser.add_argument("--subspace-explore-rounds", type=int, default=3)
    parser.add_argument("--subspace-refresh-interval", type=int, default=0)
    parser.add_argument("--subspace-intersection-rank", type=int, default=24)
    parser.add_argument("--subspace-intersection-ratio", type=float, default=None)
    parser.add_argument("--subspace-initial-intersection-ratio", type=float, default=0.25)
    parser.add_argument("--subspace-basis-demotion-ratio", type=float, default=0.5)
    parser.add_argument("--eg-num-groups", type=int, default=None)
    parser.add_argument("--eg-weight-temperature", type=float, default=None)
    parser.add_argument("--eg-weight-prior-blend", type=float, default=None)
    parser.add_argument("--eg-weight-update-momentum", type=float, default=None)
    parser.add_argument("--eg-gate-lambda", type=float, default=None)
    parser.add_argument(
        "--eg-approach-preset",
        type=str,
        default=None,
        choices=[
            "anchor",
            "discover_blend",
            "discover_conservative",
            "hybrid_personalized",
            "group_only",
        ],
    )
    parser.add_argument("--eg-enable-local-group-norm", dest="eg_enable_local_group_norm", action="store_true", default=None)
    parser.add_argument("--eg-disable-local-group-norm", dest="eg_enable_local_group_norm", action="store_false")
    parser.add_argument("--eg-enable-group-weight-learning", dest="eg_enable_group_weight_learning", action="store_true", default=None)
    parser.add_argument("--eg-disable-group-weight-learning", dest="eg_enable_group_weight_learning", action="store_false")
    parser.add_argument("--eg-uniform-group-weights", dest="eg_uniform_group_weights", action="store_true", default=None)
    parser.add_argument("--eg-nonuniform-group-weights", dest="eg_uniform_group_weights", action="store_false")
    parser.add_argument("--eg-enable-interactions", dest="eg_enable_interactions", action="store_true", default=None)
    parser.add_argument("--eg-disable-interactions", dest="eg_enable_interactions", action="store_false")
    parser.add_argument("--eg-enable-policy-blend-selection", dest="eg_policy_use_blend_selection", action="store_true", default=None)
    parser.add_argument("--eg-disable-policy-blend-selection", dest="eg_policy_use_blend_selection", action="store_false")
    parser.add_argument("--eg-group-preset", type=str, default=None)
    parser.add_argument("--eg-group-policy", type=str, default=None)
    parser.add_argument("--eg-group-manifest-path", type=str, default=None)
    parser.add_argument("--eg-policy-min-gain", type=float, default=None)
    parser.add_argument("--eg-policy-calibration-max-samples", type=int, default=None)
    parser.add_argument("--eg-discovery-accept-gamma", type=float, default=None)
    parser.add_argument("--eg-discovery-min-consensus", type=float, default=None)
    parser.add_argument("--eg-discovery-min-stability", type=float, default=None)
    parser.add_argument("--eg-discovery-fallback", type=str, default=None, choices=["horu", "group"])
    parser.add_argument("--eg-discovery-max-group-size", type=int, default=None)
    parser.add_argument("--eg-discovery-task-effect-weight", type=float, default=None)
    parser.add_argument("--eg-discovery-drift-weight", type=float, default=None)
    parser.add_argument("--eg-discovery-client-sample-ratio", type=float, default=None)
    parser.add_argument("--eg-discovery-row-sample-ratio", type=float, default=None)
    parser.add_argument("--eg-discovery-bootstrap-runs", type=int, default=None)
    parser.add_argument("--eg-interaction-weight", type=float, default=None)
    parser.add_argument("--dchb-num-packets", type=int, default=16)
    parser.add_argument("--dchb-val-fraction", type=float, default=0.10)
    parser.add_argument("--dchb-gate-update-interval", type=int, default=3)
    parser.add_argument("--dchb-gate-temperature", type=float, default=1.0)
    parser.add_argument("--dchb-open-threshold", type=float, default=0.70)
    parser.add_argument("--dchb-close-threshold", type=float, default=0.30)
    parser.add_argument("--dchb-gate-ema", type=float, default=0.20)
    parser.add_argument("--dchb-struct-threshold", type=float, default=0.0)
    parser.add_argument("--dchb-val-threshold", type=float, default=0.0)
    parser.add_argument("--dchb-utility-alpha", type=float, default=1.0)
    parser.add_argument("--dchb-utility-beta", type=float, default=0.5)
    parser.add_argument("--dchb-utility-gamma", type=float, default=0.5)
    parser.add_argument("--dchb-utility-delta", type=float, default=0.1)
    parser.add_argument("--dchb-utility-eta", type=float, default=0.05)
    parser.add_argument("--dchb-bootstrap-max-samples", type=int, default=20000)
    parser.add_argument("--dchb-gate-eval-max-samples", type=int, default=4096)
    parser.add_argument("--dchb-runtime-seed", type=int, default=13)
    parser.add_argument("--lgdro-num-experts", type=int, default=3)
    parser.add_argument("--lgdro-router-temperature", type=float, default=1.0)
    parser.add_argument("--lgdro-responsibility-temperature", type=float, default=0.75)
    parser.add_argument("--lgdro-dro-step-size", type=float, default=0.5)
    parser.add_argument("--lgdro-dro-temperature", type=float, default=1.0)
    parser.add_argument("--lgdro-router-lr", type=float, default=0.1)
    parser.add_argument("--lgdro-entropy-lambda", type=float, default=0.02)
    parser.add_argument("--lgdro-balance-lambda", type=float, default=0.1)
    parser.add_argument("--lgdro-min-router-prob", type=float, default=1e-4)
    parser.add_argument("--lgdro-runtime-seed", type=int, default=13)
    parser.add_argument("--mg-interaction-dim", type=int, default=0)
    parser.add_argument("--mg-alpha-init", type=float, default=0.5)
    parser.add_argument("--mg-alpha-lr", type=float, default=0.05)
    parser.add_argument("--mg-alpha-l2", type=float, default=0.01)
    parser.add_argument("--mg-alpha-max", type=float, default=4.0)
    parser.add_argument("--mg-tail-gamma", type=float, default=0.1)
    parser.add_argument("--mg-tail-tau", type=float, default=0.1)
    parser.add_argument("--mg-interaction-seed", type=int, default=13)
    parser.add_argument("--mg-update-mode", choices=["separate", "joint_score"], default="separate")
    parser.add_argument("--hmg-interaction-weight", type=float, default=0.15)
    parser.add_argument("--hmg-interaction-update-mode", choices=["separate", "joint_score"], default="joint_score")
    parser.add_argument("--hmg-alpha-lr", type=float, default=0.05)
    parser.add_argument("--hmg-alpha-l2", type=float, default=0.01)
    parser.add_argument("--hmg-alpha-max", type=float, default=4.0)
    parser.add_argument("--hmg-tail-gamma", type=float, default=0.1)
    parser.add_argument("--hmg-tail-tau", type=float, default=0.1)
    parser.add_argument("--fd-mask-subspace-size", type=int, default=64)
    parser.add_argument("--fd-mask-seed", type=int, default=13)
    parser.add_argument("--fd-group-count", type=int, default=4)
    parser.add_argument("--fd-group-feature-topk", type=int, default=None)
    parser.add_argument("--fd-group-mlp-hidden-dim", type=int, default=None)
    parser.add_argument("--fd-group-mlp-epochs", type=int, default=30)
    parser.add_argument("--fd-group-mlp-lr", type=float, default=0.01)
    parser.add_argument("--fd-group-seed", type=int, default=13)
    parser.add_argument("--fd-packed-num-groups", type=int, default=4)
    parser.add_argument(
        "--fd-packed-feature-group-mode",
        type=str,
        default="random_partition",
        choices=["random_partition", "random_topk", "contiguous_partition", "mlp"],
    )
    parser.add_argument("--fd-packed-feature-topk", type=int, default=None)
    parser.add_argument("--fd-packed-source-dim", type=int, default=None)
    parser.add_argument("--fd-packed-mlp-hidden-dim", type=int, default=None)
    parser.add_argument("--fd-packed-mlp-epochs", type=int, default=30)
    parser.add_argument("--fd-packed-mlp-lr", type=float, default=0.01)
    parser.add_argument("--fd-packed-seed", type=int, default=13)
    parser.add_argument(
        "--residual-packed-mode",
        type=str,
        default="hash",
        choices=["block", "superposition", "hash"],
    )
    parser.add_argument("--residual-eta", type=float, default=0.5)
    parser.add_argument(
        "--residual-eta-mode",
        type=str,
        default="fixed",
        choices=["fixed", "auto_margin_var"],
    )
    parser.add_argument("--residual-eta-grid", type=str, default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--residual-eta-beta", type=float, default=0.25)
    parser.add_argument("--residual-group-count", type=int, default=4)
    parser.add_argument(
        "--residual-feature-group-mode",
        type=str,
        default="random_partition",
        choices=["random_partition", "random_topk", "contiguous_partition", "mlp"],
    )
    parser.add_argument("--residual-feature-topk", type=int, default=None)
    parser.add_argument("--residual-source-dim", type=int, default=None)
    parser.add_argument("--residual-mlp-hidden-dim", type=int, default=None)
    parser.add_argument("--residual-mlp-epochs", type=int, default=30)
    parser.add_argument("--residual-mlp-lr", type=float, default=0.01)
    parser.add_argument("--residual-seed", type=int, default=13)
    parser.add_argument("--enable-wasserstein-sync", action="store_true")
    parser.add_argument("--wasserstein-atoms", type=int, default=3)
    parser.add_argument("--wasserstein-beta", type=float, default=0.0)
    parser.add_argument("--wasserstein-max-iters", type=int, default=20)
    parser.add_argument("--wasserstein-interval", type=int, default=1)
    parser.add_argument("--large-dataset-train-threshold", type=int, default=None)
    parser.add_argument("--large-dataset-train-cap", type=int, default=50000)
    parser.add_argument(
        "--client-regime",
        choices=["native", "pooled_all"],
        default="native",
        help=(
            "How to treat loaded clients before training/evaluation. "
            "'native' keeps dataset-provided client splits; "
            "'pooled_all' concatenates all clients into one pooled client."
        ),
    )
    parser.add_argument("--measure-energy", action="store_true")
    parser.add_argument("--power-sample-interval-ms", type=float, default=200.0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset_specs() -> dict[str, dict]:
    with open("configs/dataset_specs.json", "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    specs = {item["name"]: copy.deepcopy(item) for item in payload["datasets"]}
    for name, spec in specs.items():
        spec["root"] = DATASET_ROOTS[name]
        spec["normalization"] = DATASET_NORMALIZATION_OVERRIDES[name]
        if name == "uci_har":
            spec["preserve_original_split"] = True
        if name == "femnist":
            spec["cache_limit_clients"] = max(int(spec.get("limit_clients", 200)), 200)
            spec["cache_dir"] = "cache/femnist"
            spec["selection_seed"] = 13
            spec["max_train_samples_per_client"] = int(spec.get("max_train_samples_per_client", 256))
            spec["preserve_original_split"] = True
        if name == "wisdm":
            spec["archive"] = str(Path(spec["root"]) / "wisdm-dataset.zip")
        if name in {"cifar10", "cifar100", "isolet_raw", "emnist", "pamap2", "flamby_tcga_brca", "flamby_heart_disease"}:
            spec["seed"] = 13
    return specs


def apply_seed_to_dataset(dataset_cfg: dict, seed: int) -> dict:
    dataset_cfg = copy.deepcopy(dataset_cfg)
    if dataset_cfg["name"] in {"wisdm", "isolet_raw", "synthetic", "ninapro_db1", "pamap2", "cifar10", "cifar100", "emnist", "flamby_tcga_brca", "flamby_heart_disease"}:
        dataset_cfg["seed"] = seed
    if dataset_cfg["name"] == "femnist":
        dataset_cfg["selection_seed"] = seed
    return dataset_cfg


def apply_pilot_overrides(dataset_cfg: dict, args: argparse.Namespace) -> dict:
    dataset_cfg = copy.deepcopy(dataset_cfg)
    max_clients = int(args.pilot_max_clients)
    if dataset_cfg["name"] == "femnist":
        dataset_cfg["limit_clients"] = min(int(dataset_cfg.get("limit_clients", max_clients)), max_clients)
        dataset_cfg["cache_limit_clients"] = max(int(dataset_cfg["limit_clients"]), int(dataset_cfg.get("cache_limit_clients", max_clients)))
        dataset_cfg["max_train_samples_per_client"] = min(
            int(dataset_cfg.get("max_train_samples_per_client", 256)),
            256,
        )
    elif dataset_cfg["name"] in {"uci_har", "pamap2", "ninapro_db1", "synthetic", "mhealth", "usc_had", "hhar"}:
        dataset_cfg["limit_clients"] = min(int(dataset_cfg.get("limit_clients", max_clients)), max_clients)
    elif dataset_cfg["name"] in {"isolet_raw", "cifar10", "cifar100", "emnist", "flamby_tcga_brca", "flamby_heart_disease"}:
        dataset_cfg["num_clients"] = min(int(dataset_cfg.get("num_clients", max_clients)), max_clients)
        if dataset_cfg.get("limit_clients") is not None:
            dataset_cfg["limit_clients"] = min(int(dataset_cfg["limit_clients"]), max_clients)
    return dataset_cfg


def apply_family_dataset_overrides(dataset_cfg: dict, *, family: str) -> dict:
    dataset_cfg = copy.deepcopy(dataset_cfg)
    # LEAF FEMNIST stores grayscale pixels in [0, 1]; NN baselines should use raw images.
    if family == "nn" and dataset_cfg["name"] == "femnist":
        dataset_cfg["normalization"] = "none"
        # Draw selected clients from the full LEAF user pool instead of caching only the first 200 users.
        dataset_cfg["cache_limit_clients"] = None
        dataset_cfg["max_train_samples_per_client"] = None
    return dataset_cfg


def build_base_train(args: argparse.Namespace, *, family: str) -> dict:
    rounds = args.pilot_rounds if args.pilot else args.rounds
    local_epochs = args.pilot_local_epochs if args.pilot else args.local_epochs
    train = {
        "rounds": int(rounds),
        "local_epochs": int(local_epochs),
        "client_participation": float(getattr(args, "client_participation", 1.0)),
        "batch_size": int(args.batch_size),
    }
    if family == "hd":
        train["lr"] = float(args.hd_lr)
    else:
        train["optimizer"] = "adam"
        train["lr"] = float(args.nn_lr)
    return train


def build_hd_model_cfg(hd_dim: int) -> dict:
    return {
        "encoder": "cosine_projection",
        "hd_dim": int(hd_dim),
        "binary": False,
        "cosine_random_phase": False,
        "metric": "cos",
    }


def build_nn_model_cfg(method_key: str, dataset_name: str) -> dict:
    if method_key in {"fedavg_cnn", "fedprox_cnn", "ditto_cnn", "pfedme_cnn"}:
        if dataset_name != "femnist":
            raise ValueError(f"CNN baseline {method_key} is only supported for femnist, got {dataset_name}")
        return {
            "architecture": "cnn",
            "hidden_dim": 2048,
            "dropout": 0.0,
        }
    if method_key == "dfl_cnn":
        if dataset_name != "femnist":
            raise ValueError(f"dfl_cnn is only supported for femnist, got {dataset_name}")
        return {
            "architecture": "dfl_cnn",
            "hidden_dim": 2048,
            "dropout": 0.0,
        }
    if method_key == "dfl_mlp":
        return {
            "architecture": "dfl_mlp",
            "hidden_dim": 256,
            "branch_layers": 4,
            "dropout": 0.0,
        }
    return {
        "architecture": "mlp",
        "hidden_dim": 256,
        "num_hidden_layers": 4,
        "dropout": 0.0,
    }


def _build_commonbasis_method_kwargs(method_cfg: dict, train_cfg: dict) -> dict[str, object]:
    return {
        "shared_rank": int(method_cfg.get("shared_rank", 16)),
        "personal_rank": int(method_cfg.get("personal_rank", 16)),
        "local_epochs": int(train_cfg["local_epochs"]),
        "batch_size": train_cfg["batch_size"],
        "global_lr": float(train_cfg["lr"]),
        "personal_lr": float(train_cfg.get("personal_lr", train_cfg["lr"])),
        "val_fraction": float(method_cfg.get("val_fraction", 0.2)),
        "alpha_grid": tuple(float(value) for value in method_cfg.get("alpha_grid", [0.0])),
        "gate_alpha": float(method_cfg.get("gate_alpha", 1.0)),
        "gate_min": float(method_cfg.get("gate_min", 0.1)),
        "gate_max": float(method_cfg.get("gate_max", 0.9)),
        "intersection_rank": int(method_cfg.get("intersection_rank", 8)),
        "intersection_ratio": (
            None
            if method_cfg.get("intersection_ratio") is None
            else float(method_cfg.get("intersection_ratio"))
        ),
        "transport_mode": str(method_cfg.get("transport_mode", "fp32")),
        "coordinate_dropout": float(method_cfg.get("coordinate_dropout", 0.0)),
        "upload_skip": float(method_cfg.get("upload_skip", 0.0)),
        "bit_flip_pct": float(method_cfg.get("bit_flip_pct", 0.0)),
        "stale_upload_rounds": int(method_cfg.get("stale_upload_rounds", 0)),
        "enable_inround_checkpoints": bool(method_cfg.get("enable_inround_checkpoints", False)),
        "enable_subspace_diagnostics": bool(method_cfg.get("enable_subspace_diagnostics", False)),
        "enable_system_profiling": bool(method_cfg.get("enable_system_profiling", False)),
        "trace_rounds": tuple(int(value) for value in method_cfg.get("trace_rounds", [])),
        "runtime_seed": int(method_cfg.get("runtime_seed", 13)),
        "enable_wasserstein_sync": bool(method_cfg.get("enable_wasserstein_sync", False)),
        "wasserstein_atoms": int(method_cfg.get("wasserstein_atoms", 3)),
        "wasserstein_beta": float(method_cfg.get("wasserstein_beta", 0.0)),
        "wasserstein_max_iters": int(method_cfg.get("wasserstein_max_iters", 20)),
        "wasserstein_interval": int(method_cfg.get("wasserstein_interval", 1)),
        "eg_group_preset": str(method_cfg.get("eg_group_preset", "auto")),
        "eg_num_groups": int(method_cfg.get("eg_num_groups", 4)),
        "eg_weight_temperature": float(method_cfg.get("eg_weight_temperature", 0.35)),
        "eg_weight_prior_blend": float(method_cfg.get("eg_weight_prior_blend", 0.5)),
        "eg_weight_update_momentum": float(method_cfg.get("eg_weight_update_momentum", 0.35)),
        "eg_gate_lambda": float(method_cfg.get("eg_gate_lambda", 1.0)),
        "eg_enable_local_group_norm": bool(method_cfg.get("eg_enable_local_group_norm", False)),
        "eg_local_group_norm_eps": float(method_cfg.get("eg_local_group_norm_eps", 1e-5)),
        "eg_enable_group_weight_learning": bool(method_cfg.get("eg_enable_group_weight_learning", True)),
        "eg_uniform_group_weights": bool(method_cfg.get("eg_uniform_group_weights", False)),
        "eg_group_policy": str(method_cfg.get("eg_group_policy", "preset")),
        "eg_group_manifest_path": method_cfg.get("eg_group_manifest_path"),
        "eg_discovery_bootstrap_runs": int(method_cfg.get("eg_discovery_bootstrap_runs", 8)),
        "eg_discovery_client_sample_ratio": float(method_cfg.get("eg_discovery_client_sample_ratio", 0.7)),
        "eg_discovery_row_sample_ratio": float(method_cfg.get("eg_discovery_row_sample_ratio", 0.6)),
        "eg_discovery_min_consensus": float(method_cfg.get("eg_discovery_min_consensus", 0.55)),
        "eg_discovery_accept_gamma": float(method_cfg.get("eg_discovery_accept_gamma", 1.05)),
        "eg_discovery_min_stability": float(method_cfg.get("eg_discovery_min_stability", 0.0)),
        "eg_discovery_fallback": str(method_cfg.get("eg_discovery_fallback", "horu")),
        "eg_discovery_max_group_size": int(method_cfg.get("eg_discovery_max_group_size", 0)),
        "eg_discovery_task_effect_weight": float(method_cfg.get("eg_discovery_task_effect_weight", 1.0)),
        "eg_discovery_drift_weight": float(method_cfg.get("eg_discovery_drift_weight", 0.5)),
        "eg_discovery_seed": int(method_cfg.get("eg_discovery_seed", 13)),
        "eg_policy_use_blend_selection": bool(method_cfg.get("eg_policy_use_blend_selection", True)),
        "eg_policy_lambda_grid": tuple(float(value) for value in method_cfg.get("eg_policy_lambda_grid", [0.0, 0.25, 0.5, 0.75, 1.0])),
        "eg_policy_min_gain": float(method_cfg.get("eg_policy_min_gain", 0.0)),
        "eg_policy_calibration_split": str(method_cfg.get("eg_policy_calibration_split", "train")),
        "eg_policy_calibration_max_samples": int(method_cfg.get("eg_policy_calibration_max_samples", 1024)),
        "eg_enable_interactions": bool(method_cfg.get("eg_enable_interactions", True)),
        "eg_interaction_weight": float(method_cfg.get("eg_interaction_weight", 0.10)),
        "eg_interaction_pairs": tuple(str(value) for value in method_cfg.get("eg_interaction_pairs", ["hand+ankle", "chest+ankle"])),
        "ss_block_preset": str(method_cfg.get("ss_block_preset", "auto")),
        "ss_num_blocks": int(method_cfg.get("ss_num_blocks", 4)),
        "ss_conflict_threshold": float(method_cfg.get("ss_conflict_threshold", 0.35)),
        "ss_residual_weight": float(method_cfg.get("ss_residual_weight", 0.30)),
        "ss_server_momentum": float(method_cfg.get("ss_server_momentum", 0.50)),
        "ss_cache_train_hv": bool(method_cfg.get("ss_cache_train_hv", False)),
        "interaction_dim": int(method_cfg.get("interaction_dim", 0)),
        "interaction_weight": float(method_cfg.get("interaction_weight", 0.15)),
        "interaction_seed": int(method_cfg.get("interaction_seed", 13)),
        "interaction_update_mode": str(method_cfg.get("interaction_update_mode", "joint_score")),
        "interaction_alpha_lr": float(method_cfg.get("interaction_alpha_lr", 0.05)),
        "interaction_alpha_l2": float(method_cfg.get("interaction_alpha_l2", 0.01)),
        "interaction_alpha_max": float(method_cfg.get("interaction_alpha_max", 4.0)),
        "interaction_tail_gamma": float(method_cfg.get("interaction_tail_gamma", 0.1)),
        "interaction_tail_tau": float(method_cfg.get("interaction_tail_tau", 0.1)),
        "debug": method_cfg.get("debug", False),
    }


def _base_subspace_method_cfg(args: argparse.Namespace, alpha_grid: list[float]) -> dict[str, object]:
    return {
        "shared_rank": int(args.subspace_shared_rank),
        "personal_rank": int(args.subspace_personal_rank),
        "val_fraction": float(args.subspace_val_fraction),
        "alpha_grid": alpha_grid,
        "gate_alpha": float(args.subspace_rowgate_alpha),
        "gate_min": float(args.subspace_rowgate_min),
        "gate_max": float(args.subspace_rowgate_max),
    }


def _apply_commonbasis_method_cfg_overrides(method_cfg: dict[str, object], args: argparse.Namespace) -> None:
    method_cfg["intersection_rank"] = int(getattr(args, "subspace_intersection_rank", 8))
    intersection_ratio = getattr(args, "subspace_intersection_ratio", None)
    if intersection_ratio is not None:
        method_cfg["intersection_ratio"] = float(intersection_ratio)
    method_cfg["transport_mode"] = str(getattr(args, "transport_mode", "fp32"))
    method_cfg["coordinate_dropout"] = float(getattr(args, "transport_coordinate_dropout", 0.0))
    method_cfg["upload_skip"] = float(getattr(args, "transport_upload_skip", 0.0))
    method_cfg["bit_flip_pct"] = float(getattr(args, "transport_bit_flip_pct", 0.0))
    method_cfg["stale_upload_rounds"] = int(getattr(args, "stale_upload_rounds", 0))
    method_cfg["enable_inround_checkpoints"] = bool(getattr(args, "enable_inround_checkpoints", False))
    method_cfg["enable_subspace_diagnostics"] = bool(getattr(args, "enable_subspace_diagnostics", False))
    method_cfg["enable_system_profiling"] = bool(getattr(args, "enable_system_profiling", False))
    method_cfg["trace_rounds"] = [int(value) for value in getattr(args, "trace_rounds", [])]
    method_cfg["runtime_seed"] = int(getattr(args, "runtime_seed", getattr(args, "seed", 13)))
    method_cfg["enable_wasserstein_sync"] = bool(getattr(args, "enable_wasserstein_sync", False))
    method_cfg["wasserstein_atoms"] = int(getattr(args, "wasserstein_atoms", 3))
    method_cfg["wasserstein_beta"] = float(getattr(args, "wasserstein_beta", 0.0))
    method_cfg["wasserstein_max_iters"] = int(getattr(args, "wasserstein_max_iters", 20))
    method_cfg["wasserstein_interval"] = int(getattr(args, "wasserstein_interval", 1))
    method_cfg["interaction_dim"] = int(getattr(args, "mg_interaction_dim", 0))
    method_cfg["interaction_seed"] = int(getattr(args, "mg_interaction_seed", 13))
    method_cfg["interaction_weight"] = float(getattr(args, "hmg_interaction_weight", 0.15))
    method_cfg["interaction_update_mode"] = str(getattr(args, "hmg_interaction_update_mode", "joint_score"))
    method_cfg["interaction_alpha_lr"] = float(getattr(args, "hmg_alpha_lr", 0.05))
    method_cfg["interaction_alpha_l2"] = float(getattr(args, "hmg_alpha_l2", 0.01))
    method_cfg["interaction_alpha_max"] = float(getattr(args, "hmg_alpha_max", 4.0))
    method_cfg["interaction_tail_gamma"] = float(getattr(args, "hmg_tail_gamma", 0.1))
    method_cfg["interaction_tail_tau"] = float(getattr(args, "hmg_tail_tau", 0.1))
    eg_num_groups = getattr(args, "eg_num_groups", None)
    if eg_num_groups is not None:
        method_cfg["eg_num_groups"] = int(eg_num_groups)
    eg_weight_temperature = getattr(args, "eg_weight_temperature", None)
    if eg_weight_temperature is not None:
        method_cfg["eg_weight_temperature"] = float(eg_weight_temperature)
    eg_weight_prior_blend = getattr(args, "eg_weight_prior_blend", None)
    if eg_weight_prior_blend is not None:
        method_cfg["eg_weight_prior_blend"] = float(eg_weight_prior_blend)
    eg_weight_update_momentum = getattr(args, "eg_weight_update_momentum", None)
    if eg_weight_update_momentum is not None:
        method_cfg["eg_weight_update_momentum"] = float(eg_weight_update_momentum)
    eg_gate_lambda = getattr(args, "eg_gate_lambda", None)
    if eg_gate_lambda is not None:
        method_cfg["eg_gate_lambda"] = float(eg_gate_lambda)
    eg_approach_preset = getattr(args, "eg_approach_preset", None)
    if eg_approach_preset is not None:
        preset = str(eg_approach_preset).strip().lower()
        if preset == "anchor":
            method_cfg.update(
                {
                    "eg_group_policy": "preset",
                    "eg_policy_use_blend_selection": False,
                    "eg_discovery_fallback": "horu",
                    "eg_gate_lambda": 1.0,
                    "eg_enable_local_group_norm": False,
                    "eg_enable_group_weight_learning": True,
                    "eg_uniform_group_weights": False,
                    "eg_enable_interactions": True,
                    "eg_interaction_weight": 0.10,
                }
            )
        elif preset == "discover_blend":
            method_cfg.update(
                {
                    "eg_group_policy": "discover",
                    "eg_policy_use_blend_selection": True,
                    "eg_policy_min_gain": 0.0,
                    "eg_discovery_fallback": "horu",
                    "eg_gate_lambda": 1.0,
                    "eg_enable_local_group_norm": False,
                    "eg_enable_group_weight_learning": True,
                    "eg_uniform_group_weights": False,
                    "eg_enable_interactions": True,
                    "eg_interaction_weight": 0.10,
                }
            )
        elif preset == "discover_conservative":
            method_cfg.update(
                {
                    "eg_group_policy": "discover",
                    "eg_policy_use_blend_selection": True,
                    "eg_policy_min_gain": 0.005,
                    "eg_discovery_fallback": "horu",
                    "eg_discovery_accept_gamma": 1.10,
                    "eg_discovery_min_consensus": 0.60,
                    "eg_discovery_min_stability": 0.55,
                    "eg_gate_lambda": 1.0,
                    "eg_enable_local_group_norm": True,
                    "eg_enable_group_weight_learning": False,
                    "eg_uniform_group_weights": True,
                    "eg_enable_interactions": False,
                    "eg_interaction_weight": 0.0,
                }
            )
        elif preset == "hybrid_personalized":
            method_cfg.update(
                {
                    "eg_group_policy": "discover",
                    "eg_policy_use_blend_selection": True,
                    "eg_policy_min_gain": 0.0,
                    "eg_discovery_fallback": "horu",
                    "eg_gate_lambda": 1.0,
                    "eg_enable_local_group_norm": True,
                    "eg_enable_group_weight_learning": True,
                    "eg_uniform_group_weights": False,
                    "eg_enable_interactions": True,
                    "eg_interaction_weight": 0.05,
                }
            )
        elif preset == "group_only":
            method_cfg.update(
                {
                    "eg_group_policy": "preset",
                    "eg_policy_use_blend_selection": False,
                    "eg_discovery_fallback": "group",
                    "eg_gate_lambda": 1.0,
                    "eg_enable_local_group_norm": False,
                    "eg_enable_group_weight_learning": True,
                    "eg_uniform_group_weights": False,
                    "eg_enable_interactions": True,
                    "eg_interaction_weight": 0.10,
                }
            )
    eg_enable_local_group_norm = getattr(args, "eg_enable_local_group_norm", None)
    if eg_enable_local_group_norm is not None:
        method_cfg["eg_enable_local_group_norm"] = bool(eg_enable_local_group_norm)
    eg_enable_group_weight_learning = getattr(args, "eg_enable_group_weight_learning", None)
    if eg_enable_group_weight_learning is not None:
        method_cfg["eg_enable_group_weight_learning"] = bool(eg_enable_group_weight_learning)
    eg_uniform_group_weights = getattr(args, "eg_uniform_group_weights", None)
    if eg_uniform_group_weights is not None:
        method_cfg["eg_uniform_group_weights"] = bool(eg_uniform_group_weights)
    eg_enable_interactions = getattr(args, "eg_enable_interactions", None)
    if eg_enable_interactions is not None:
        method_cfg["eg_enable_interactions"] = bool(eg_enable_interactions)
    eg_policy_use_blend_selection = getattr(args, "eg_policy_use_blend_selection", None)
    if eg_policy_use_blend_selection is not None:
        method_cfg["eg_policy_use_blend_selection"] = bool(eg_policy_use_blend_selection)
    eg_group_preset = getattr(args, "eg_group_preset", None)
    if eg_group_preset is not None:
        method_cfg["eg_group_preset"] = str(eg_group_preset)
    eg_group_policy = getattr(args, "eg_group_policy", None)
    if eg_group_policy is not None:
        method_cfg["eg_group_policy"] = str(eg_group_policy)
    eg_group_manifest_path = getattr(args, "eg_group_manifest_path", None)
    if eg_group_manifest_path is not None:
        method_cfg["eg_group_manifest_path"] = str(eg_group_manifest_path)
    eg_policy_min_gain = getattr(args, "eg_policy_min_gain", None)
    if eg_policy_min_gain is not None:
        method_cfg["eg_policy_min_gain"] = float(eg_policy_min_gain)
    eg_policy_calibration_max_samples = getattr(args, "eg_policy_calibration_max_samples", None)
    if eg_policy_calibration_max_samples is not None:
        method_cfg["eg_policy_calibration_max_samples"] = int(eg_policy_calibration_max_samples)
    eg_discovery_accept_gamma = getattr(args, "eg_discovery_accept_gamma", None)
    if eg_discovery_accept_gamma is not None:
        method_cfg["eg_discovery_accept_gamma"] = float(eg_discovery_accept_gamma)
    eg_discovery_min_consensus = getattr(args, "eg_discovery_min_consensus", None)
    if eg_discovery_min_consensus is not None:
        method_cfg["eg_discovery_min_consensus"] = float(eg_discovery_min_consensus)
    eg_discovery_min_stability = getattr(args, "eg_discovery_min_stability", None)
    if eg_discovery_min_stability is not None:
        method_cfg["eg_discovery_min_stability"] = float(eg_discovery_min_stability)
    eg_discovery_fallback = getattr(args, "eg_discovery_fallback", None)
    if eg_discovery_fallback is not None:
        method_cfg["eg_discovery_fallback"] = str(eg_discovery_fallback)
    eg_discovery_max_group_size = getattr(args, "eg_discovery_max_group_size", None)
    if eg_discovery_max_group_size is not None:
        method_cfg["eg_discovery_max_group_size"] = int(eg_discovery_max_group_size)
    eg_discovery_task_effect_weight = getattr(args, "eg_discovery_task_effect_weight", None)
    if eg_discovery_task_effect_weight is not None:
        method_cfg["eg_discovery_task_effect_weight"] = float(eg_discovery_task_effect_weight)
    eg_discovery_drift_weight = getattr(args, "eg_discovery_drift_weight", None)
    if eg_discovery_drift_weight is not None:
        method_cfg["eg_discovery_drift_weight"] = float(eg_discovery_drift_weight)
    eg_discovery_client_sample_ratio = getattr(args, "eg_discovery_client_sample_ratio", None)
    if eg_discovery_client_sample_ratio is not None:
        method_cfg["eg_discovery_client_sample_ratio"] = float(eg_discovery_client_sample_ratio)
    eg_discovery_row_sample_ratio = getattr(args, "eg_discovery_row_sample_ratio", None)
    if eg_discovery_row_sample_ratio is not None:
        method_cfg["eg_discovery_row_sample_ratio"] = float(eg_discovery_row_sample_ratio)
    eg_discovery_bootstrap_runs = getattr(args, "eg_discovery_bootstrap_runs", None)
    if eg_discovery_bootstrap_runs is not None:
        method_cfg["eg_discovery_bootstrap_runs"] = int(eg_discovery_bootstrap_runs)
    eg_interaction_weight = getattr(args, "eg_interaction_weight", None)
    if eg_interaction_weight is not None:
        method_cfg["eg_interaction_weight"] = float(eg_interaction_weight)


def _apply_dchb_method_cfg_overrides(method_cfg: dict[str, object], args: argparse.Namespace) -> None:
    method_cfg["num_packets"] = int(getattr(args, "dchb_num_packets", 16))
    method_cfg["val_fraction"] = float(getattr(args, "dchb_val_fraction", 0.10))
    method_cfg["gate_update_interval"] = int(getattr(args, "dchb_gate_update_interval", 3))
    method_cfg["gate_temperature"] = float(getattr(args, "dchb_gate_temperature", 1.0))
    method_cfg["gate_open_threshold"] = float(getattr(args, "dchb_open_threshold", 0.70))
    method_cfg["gate_close_threshold"] = float(getattr(args, "dchb_close_threshold", 0.30))
    method_cfg["gate_ema"] = float(getattr(args, "dchb_gate_ema", 0.20))
    method_cfg["tau_struct"] = float(getattr(args, "dchb_struct_threshold", 0.0))
    method_cfg["tau_val"] = float(getattr(args, "dchb_val_threshold", 0.0))
    method_cfg["utility_alpha"] = float(getattr(args, "dchb_utility_alpha", 1.0))
    method_cfg["utility_beta"] = float(getattr(args, "dchb_utility_beta", 0.5))
    method_cfg["utility_gamma"] = float(getattr(args, "dchb_utility_gamma", 0.5))
    method_cfg["utility_delta"] = float(getattr(args, "dchb_utility_delta", 0.1))
    method_cfg["utility_eta"] = float(getattr(args, "dchb_utility_eta", 0.05))
    method_cfg["bootstrap_max_samples"] = int(getattr(args, "dchb_bootstrap_max_samples", 20000))
    method_cfg["gate_eval_max_samples"] = int(getattr(args, "dchb_gate_eval_max_samples", 4096))
    method_cfg["runtime_seed"] = int(getattr(args, "dchb_runtime_seed", 13))


def _apply_lgdro_method_cfg_overrides(method_cfg: dict[str, object], args: argparse.Namespace) -> None:
    method_cfg["num_experts"] = int(getattr(args, "lgdro_num_experts", 3))
    method_cfg["router_temperature"] = float(getattr(args, "lgdro_router_temperature", 1.0))
    method_cfg["responsibility_temperature"] = float(getattr(args, "lgdro_responsibility_temperature", 0.75))
    method_cfg["dro_step_size"] = float(getattr(args, "lgdro_dro_step_size", 0.5))
    method_cfg["dro_temperature"] = float(getattr(args, "lgdro_dro_temperature", 1.0))
    method_cfg["router_lr"] = float(getattr(args, "lgdro_router_lr", 0.1))
    method_cfg["entropy_lambda"] = float(getattr(args, "lgdro_entropy_lambda", 0.02))
    method_cfg["balance_lambda"] = float(getattr(args, "lgdro_balance_lambda", 0.1))
    method_cfg["min_router_prob"] = float(getattr(args, "lgdro_min_router_prob", 1e-4))
    method_cfg["runtime_seed"] = int(getattr(args, "lgdro_runtime_seed", 13))


def _apply_metric_gate_method_cfg_overrides(method_cfg: dict[str, object], args: argparse.Namespace) -> None:
    method_cfg["interaction_dim"] = int(getattr(args, "mg_interaction_dim", 0))
    method_cfg["alpha_init"] = float(getattr(args, "mg_alpha_init", 0.5))
    method_cfg["alpha_lr"] = float(getattr(args, "mg_alpha_lr", 0.05))
    method_cfg["alpha_l2"] = float(getattr(args, "mg_alpha_l2", 0.01))
    method_cfg["alpha_max"] = float(getattr(args, "mg_alpha_max", 4.0))
    method_cfg["tail_gamma"] = float(getattr(args, "mg_tail_gamma", 0.1))
    method_cfg["tail_tau"] = float(getattr(args, "mg_tail_tau", 0.1))
    method_cfg["interaction_seed"] = int(getattr(args, "mg_interaction_seed", 13))
    method_cfg["update_mode"] = str(getattr(args, "mg_update_mode", "separate"))


def _apply_fd_masking_method_cfg_overrides(method_cfg: dict[str, object], args: argparse.Namespace) -> None:
    method_cfg["subspace_size"] = int(getattr(args, "fd_mask_subspace_size", 64))
    method_cfg["mask_seed"] = int(getattr(args, "fd_mask_seed", 13))


def _apply_fd_group_method_cfg_overrides(method_cfg: dict[str, object], args: argparse.Namespace) -> None:
    method_cfg["num_groups"] = int(getattr(args, "fd_group_count", 4))
    feature_topk = getattr(args, "fd_group_feature_topk", None)
    if feature_topk is not None:
        method_cfg["feature_topk"] = int(feature_topk)
    mlp_hidden_dim = getattr(args, "fd_group_mlp_hidden_dim", None)
    if mlp_hidden_dim is not None:
        method_cfg["mlp_hidden_dim"] = int(mlp_hidden_dim)
    method_cfg["mlp_epochs"] = int(getattr(args, "fd_group_mlp_epochs", 30))
    method_cfg["mlp_lr"] = float(getattr(args, "fd_group_mlp_lr", 0.01))
    method_cfg["mask_seed"] = int(getattr(args, "fd_group_seed", 13))


def _apply_fd_packed_method_cfg_overrides(method_cfg: dict[str, object], args: argparse.Namespace) -> None:
    method_cfg["num_groups"] = int(getattr(args, "fd_packed_num_groups", 4))
    method_cfg["feature_group_mode"] = str(getattr(args, "fd_packed_feature_group_mode", "random_partition"))
    feature_topk = getattr(args, "fd_packed_feature_topk", None)
    if feature_topk is not None:
        method_cfg["feature_topk"] = int(feature_topk)
    source_dim = getattr(args, "fd_packed_source_dim", None)
    if source_dim is not None:
        method_cfg["source_dim"] = int(source_dim)
    mlp_hidden_dim = getattr(args, "fd_packed_mlp_hidden_dim", None)
    if mlp_hidden_dim is not None:
        method_cfg["mlp_hidden_dim"] = int(mlp_hidden_dim)
    method_cfg["mlp_epochs"] = int(getattr(args, "fd_packed_mlp_epochs", 30))
    method_cfg["mlp_lr"] = float(getattr(args, "fd_packed_mlp_lr", 0.01))
    method_cfg["mask_seed"] = int(getattr(args, "fd_packed_seed", 13))


def _apply_fd_naive_group_method_cfg_overrides(method_cfg: dict[str, object], args: argparse.Namespace) -> None:
    method_cfg["num_groups"] = int(getattr(args, "fd_packed_num_groups", 4))
    method_cfg["feature_group_mode"] = str(getattr(args, "fd_packed_feature_group_mode", "random_partition"))
    feature_topk = getattr(args, "fd_packed_feature_topk", None)
    if feature_topk is not None:
        method_cfg["feature_topk"] = int(feature_topk)
    source_dim = getattr(args, "fd_packed_source_dim", None)
    if source_dim is not None:
        method_cfg["source_dim"] = int(source_dim)
    mlp_hidden_dim = getattr(args, "fd_packed_mlp_hidden_dim", None)
    if mlp_hidden_dim is not None:
        method_cfg["mlp_hidden_dim"] = int(mlp_hidden_dim)
    method_cfg["mlp_epochs"] = int(getattr(args, "fd_packed_mlp_epochs", 30))
    method_cfg["mlp_lr"] = float(getattr(args, "fd_packed_mlp_lr", 0.01))
    method_cfg["mask_seed"] = int(getattr(args, "fd_packed_seed", 13))


def _parse_float_csv(raw: str, *, arg_name: str) -> list[float]:
    values: list[float] = []
    for token in str(raw).split(","):
        stripped = token.strip()
        if not stripped:
            continue
        try:
            values.append(float(stripped))
        except ValueError as exc:
            raise ValueError(f"Invalid float in {arg_name}: {stripped!r}") from exc
    return values


def _apply_fd_residual_method_cfg_overrides(method_cfg: dict[str, object], args: argparse.Namespace) -> None:
    method_cfg["eta"] = float(getattr(args, "residual_eta", 0.5))
    method_cfg["residual_eta_mode"] = str(getattr(args, "residual_eta_mode", "fixed"))
    method_cfg["residual_eta_grid"] = _parse_float_csv(
        str(getattr(args, "residual_eta_grid", "0,0.25,0.5,0.75,1.0")),
        arg_name="--residual-eta-grid",
    )
    method_cfg["residual_eta_beta"] = float(getattr(args, "residual_eta_beta", 0.25))
    method_cfg["group_packing_mode"] = str(getattr(args, "residual_packed_mode", "hash"))
    method_cfg["num_groups"] = int(getattr(args, "residual_group_count", 4))
    method_cfg["feature_group_mode"] = str(getattr(args, "residual_feature_group_mode", "random_partition"))
    feature_topk = getattr(args, "residual_feature_topk", None)
    if feature_topk is not None:
        method_cfg["feature_topk"] = int(feature_topk)
    source_dim = getattr(args, "residual_source_dim", None)
    if source_dim is not None:
        method_cfg["source_dim"] = int(source_dim)
    mlp_hidden_dim = getattr(args, "residual_mlp_hidden_dim", None)
    if mlp_hidden_dim is not None:
        method_cfg["mlp_hidden_dim"] = int(mlp_hidden_dim)
    method_cfg["mlp_epochs"] = int(getattr(args, "residual_mlp_epochs", 30))
    method_cfg["mlp_lr"] = float(getattr(args, "residual_mlp_lr", 0.01))
    method_cfg["mask_seed"] = int(getattr(args, "residual_seed", 13))


def build_hd_method(cfg: dict, input_dim: int, num_classes: int, device: torch.device):
    method_cfg = cfg["method"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    method_name = method_cfg["name"]

    encoder_name = model_cfg.get("encoder", "cosine_projection")
    if method_name == "masking_anticollapse_hd":
        encoder = MaskedCosineProjectionEncoder(
            input_dim=input_dim,
            hd_dim=model_cfg["hd_dim"],
            subspace_size=int(method_cfg.get("subspace_size", 64)),
            binary=model_cfg.get("binary", False),
            random_phase=bool(model_cfg.get("cosine_random_phase", False)),
            mask_seed=int(method_cfg.get("mask_seed", 13)),
            device=device,
        )
    elif method_name == "mlp_group_learned_hd":
        encoder = GroupLearnedCosineProjectionEncoder(
            input_dim=input_dim,
            hd_dim=model_cfg["hd_dim"],
            num_groups=int(method_cfg.get("num_groups", 4)),
            feature_topk=(
                None
                if method_cfg.get("feature_topk") is None
                else int(method_cfg.get("feature_topk"))
            ),
            mlp_hidden_dim=(
                None
                if method_cfg.get("mlp_hidden_dim") is None
                else int(method_cfg.get("mlp_hidden_dim"))
            ),
            mlp_epochs=int(method_cfg.get("mlp_epochs", 30)),
            mlp_lr=float(method_cfg.get("mlp_lr", 0.01)),
            binary=model_cfg.get("binary", False),
            random_phase=bool(model_cfg.get("cosine_random_phase", False)),
            mask_seed=int(method_cfg.get("mask_seed", 13)),
            device=device,
        )
    elif method_name in {"block_packed_group_hd", "superposition_packed_group_hd", "hash_packed_group_hd"}:
        packing_mode = {
            "block_packed_group_hd": "block",
            "superposition_packed_group_hd": "superposition",
            "hash_packed_group_hd": "hash",
        }[method_name]
        encoder = PackedAdditiveCosineEncoder(
            input_dim=input_dim,
            hd_dim=model_cfg["hd_dim"],
            packing_mode=packing_mode,
            num_groups=int(method_cfg.get("num_groups", 4)),
            feature_group_mode=str(method_cfg.get("feature_group_mode", "random_partition")),
            feature_topk=(
                None
                if method_cfg.get("feature_topk") is None
                else int(method_cfg.get("feature_topk"))
            ),
            source_dim=(
                None
                if method_cfg.get("source_dim") is None
                else int(method_cfg.get("source_dim"))
            ),
            mlp_hidden_dim=(
                None
                if method_cfg.get("mlp_hidden_dim") is None
                else int(method_cfg.get("mlp_hidden_dim"))
            ),
            mlp_epochs=int(method_cfg.get("mlp_epochs", 30)),
            mlp_lr=float(method_cfg.get("mlp_lr", 0.01)),
            binary=model_cfg.get("binary", False),
            random_phase=bool(model_cfg.get("cosine_random_phase", False)),
            mask_seed=int(method_cfg.get("mask_seed", 13)),
            device=device,
        )
    elif method_name == "naive_group_ensemble":
        encoder = PackedAdditiveCosineEncoder(
            input_dim=input_dim,
            hd_dim=model_cfg["hd_dim"],
            packing_mode="superposition",
            num_groups=int(method_cfg.get("num_groups", 4)),
            feature_group_mode=str(method_cfg.get("feature_group_mode", "random_partition")),
            feature_topk=(
                None
                if method_cfg.get("feature_topk") is None
                else int(method_cfg.get("feature_topk"))
            ),
            source_dim=(
                None
                if method_cfg.get("source_dim") is None
                else int(method_cfg.get("source_dim"))
            ),
            mlp_hidden_dim=(
                None
                if method_cfg.get("mlp_hidden_dim") is None
                else int(method_cfg.get("mlp_hidden_dim"))
            ),
            mlp_epochs=int(method_cfg.get("mlp_epochs", 30)),
            mlp_lr=float(method_cfg.get("mlp_lr", 0.01)),
            binary=model_cfg.get("binary", False),
            random_phase=bool(model_cfg.get("cosine_random_phase", False)),
            mask_seed=int(method_cfg.get("mask_seed", 13)),
            device=device,
        )
    elif method_name == "residual_packed_group_hd":
        encoder = ResidualPackedCosineEncoder(
            input_dim=input_dim,
            hd_dim=model_cfg["hd_dim"],
            eta=float(method_cfg.get("eta", 0.5)),
            group_packing_mode=str(method_cfg.get("group_packing_mode", "hash")),
            num_groups=int(method_cfg.get("num_groups", 4)),
            feature_group_mode=str(method_cfg.get("feature_group_mode", "random_partition")),
            feature_topk=(
                None
                if method_cfg.get("feature_topk") is None
                else int(method_cfg.get("feature_topk"))
            ),
            source_dim=(
                None
                if method_cfg.get("source_dim") is None
                else int(method_cfg.get("source_dim"))
            ),
            mlp_hidden_dim=(
                None
                if method_cfg.get("mlp_hidden_dim") is None
                else int(method_cfg.get("mlp_hidden_dim"))
            ),
            mlp_epochs=int(method_cfg.get("mlp_epochs", 30)),
            mlp_lr=float(method_cfg.get("mlp_lr", 0.01)),
            binary=model_cfg.get("binary", False),
            random_phase=bool(model_cfg.get("cosine_random_phase", False)),
            mask_seed=int(method_cfg.get("mask_seed", 13)),
            device=device,
        )
    else:
        if encoder_name != "cosine_projection":
            raise ValueError(f"Unsupported encoder for now: {encoder_name}")
        encoder = CosineProjectionEncoder(
            input_dim=input_dim,
            hd_dim=model_cfg["hd_dim"],
            binary=model_cfg.get("binary", False),
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
    if is_commonbasis_method_name(method_name):
        method_class = COMMONBASIS_METHOD_CLASS_BY_NAME[method_name]
        method_kwargs = _build_commonbasis_method_kwargs(method_cfg, train_cfg)
        signature = inspect.signature(method_class)
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if not accepts_var_kwargs:
            allowed = set(signature.parameters.keys())
            method_kwargs = {key: value for key, value in method_kwargs.items() if key in allowed}
        method = method_class(
            encoder=encoder,
            num_classes=num_classes,
            **method_kwargs,
        )
        metric_key = "mean_personalized_accuracy"
    elif method_name == "local_hd":
        method = LocalHDMethod(
            encoder=encoder,
            updater=updater,
            num_classes=num_classes,
            metric=metric,
            debug=bool(method_cfg.get("debug", False)),
        )
        metric_key = "mean_local_test_accuracy"
    elif method_name == "fedhdc":
        method = FedHDCMethod(
            encoder=encoder,
            updater=updater,
            num_classes=num_classes,
            metric=metric,
            debug=bool(method_cfg.get("debug", False)),
        )
        metric_key = "global_test_accuracy"
    elif method_name == "hyperfeel":
        method = HyperFeelMethod(
            encoder=encoder,
            num_classes=num_classes,
            local_epochs=int(train_cfg["local_epochs"]),
            batch_size=int(train_cfg["batch_size"]),
            lr=float(train_cfg["lr"]),
            metric=metric,
            debug=bool(method_cfg.get("debug", False)),
        )
        metric_key = "mean_personalized_accuracy"
    elif method_name == "centralized_hd":
        method = CentralizedHDMethod(
            encoder=encoder,
            updater=updater,
            num_classes=num_classes,
            metric=metric,
            debug=bool(method_cfg.get("debug", False)),
        )
        metric_key = "mean_global_accuracy"
    elif method_name == "masking_anticollapse_hd":
        method = MaskingAntiCollapseHDMethod(
            encoder=encoder,
            updater=updater,
            num_classes=num_classes,
            metric=metric,
            debug=bool(method_cfg.get("debug", False)),
        )
        metric_key = "mean_global_accuracy"
    elif method_name == "mlp_group_learned_hd":
        method = MLPGroupLearnedHDMethod(
            encoder=encoder,
            updater=updater,
            num_classes=num_classes,
            metric=metric,
            debug=bool(method_cfg.get("debug", False)),
        )
        metric_key = "mean_global_accuracy"
    elif method_name in {"block_packed_group_hd", "superposition_packed_group_hd", "hash_packed_group_hd"}:
        method = PackedAdditiveHDMethod(
            encoder=encoder,
            updater=updater,
            num_classes=num_classes,
            metric=metric,
            debug=bool(method_cfg.get("debug", False)),
        )
        metric_key = "mean_global_accuracy"
    elif method_name == "naive_group_ensemble":
        method = NaiveGroupEnsembleHDMethod(
            encoder=encoder,
            updater=updater,
            num_classes=num_classes,
            metric=metric,
            debug=bool(method_cfg.get("debug", False)),
        )
        metric_key = "mean_global_accuracy"
    elif method_name == "residual_packed_group_hd":
        method = ResidualPackedHDMethod(
            encoder=encoder,
            updater=updater,
            num_classes=num_classes,
            metric=metric,
            residual_eta_mode=str(method_cfg.get("residual_eta_mode", "fixed")),
            residual_eta_grid=tuple(float(value) for value in method_cfg.get("residual_eta_grid", [0.0, 0.25, 0.5, 0.75, 1.0])),
            residual_eta_beta=float(method_cfg.get("residual_eta_beta", 0.25)),
            debug=bool(method_cfg.get("debug", False)),
        )
        metric_key = "mean_global_accuracy"
    elif method_name == "dchb_hd":
        method = DCHBMethod(
            encoder=encoder,
            updater=updater,
            num_classes=num_classes,
            num_packets=int(method_cfg.get("num_packets", 16)),
            val_fraction=float(method_cfg.get("val_fraction", 0.10)),
            gate_update_interval=int(method_cfg.get("gate_update_interval", 3)),
            gate_temperature=float(method_cfg.get("gate_temperature", 1.0)),
            gate_open_threshold=float(method_cfg.get("gate_open_threshold", 0.70)),
            gate_close_threshold=float(method_cfg.get("gate_close_threshold", 0.30)),
            gate_ema=float(method_cfg.get("gate_ema", 0.20)),
            tau_struct=float(method_cfg.get("tau_struct", 0.0)),
            tau_val=float(method_cfg.get("tau_val", 0.0)),
            utility_alpha=float(method_cfg.get("utility_alpha", 1.0)),
            utility_beta=float(method_cfg.get("utility_beta", 0.5)),
            utility_gamma=float(method_cfg.get("utility_gamma", 0.5)),
            utility_delta=float(method_cfg.get("utility_delta", 0.1)),
            utility_eta=float(method_cfg.get("utility_eta", 0.05)),
            bootstrap_max_samples=int(method_cfg.get("bootstrap_max_samples", 20000)),
            gate_eval_max_samples=int(method_cfg.get("gate_eval_max_samples", 4096)),
            runtime_seed=int(method_cfg.get("runtime_seed", 13)),
            centralized_mode=False,
            debug=bool(method_cfg.get("debug", False)),
        )
        metric_key = "mean_personalized_accuracy"
    elif method_name == "dchb_centralized_hd":
        method = DCHBMethod(
            encoder=encoder,
            updater=updater,
            num_classes=num_classes,
            num_packets=int(method_cfg.get("num_packets", 16)),
            val_fraction=float(method_cfg.get("val_fraction", 0.10)),
            gate_update_interval=int(method_cfg.get("gate_update_interval", 3)),
            gate_temperature=float(method_cfg.get("gate_temperature", 1.0)),
            gate_open_threshold=float(method_cfg.get("gate_open_threshold", 0.70)),
            gate_close_threshold=float(method_cfg.get("gate_close_threshold", 0.30)),
            gate_ema=float(method_cfg.get("gate_ema", 0.20)),
            tau_struct=float(method_cfg.get("tau_struct", 0.0)),
            tau_val=float(method_cfg.get("tau_val", 0.0)),
            utility_alpha=float(method_cfg.get("utility_alpha", 1.0)),
            utility_beta=float(method_cfg.get("utility_beta", 0.5)),
            utility_gamma=float(method_cfg.get("utility_gamma", 0.5)),
            utility_delta=float(method_cfg.get("utility_delta", 0.1)),
            utility_eta=float(method_cfg.get("utility_eta", 0.05)),
            bootstrap_max_samples=int(method_cfg.get("bootstrap_max_samples", 20000)),
            gate_eval_max_samples=int(method_cfg.get("gate_eval_max_samples", 4096)),
            runtime_seed=int(method_cfg.get("runtime_seed", 13)),
            centralized_mode=True,
            debug=bool(method_cfg.get("debug", False)),
        )
        metric_key = "mean_global_accuracy"
    elif method_name == "latent_group_dro_hd":
        method = LatentGroupDROHDMethod(
            encoder=encoder,
            updater=updater,
            num_classes=num_classes,
            num_experts=int(method_cfg.get("num_experts", 3)),
            router_temperature=float(method_cfg.get("router_temperature", 1.0)),
            responsibility_temperature=float(method_cfg.get("responsibility_temperature", 0.75)),
            dro_step_size=float(method_cfg.get("dro_step_size", 0.5)),
            dro_temperature=float(method_cfg.get("dro_temperature", 1.0)),
            router_lr=float(method_cfg.get("router_lr", 0.1)),
            entropy_lambda=float(method_cfg.get("entropy_lambda", 0.02)),
            balance_lambda=float(method_cfg.get("balance_lambda", 0.1)),
            min_router_prob=float(method_cfg.get("min_router_prob", 1e-4)),
            runtime_seed=int(method_cfg.get("runtime_seed", 13)),
            metric=metric,
            debug=bool(method_cfg.get("debug", False)),
        )
        metric_key = "mean_global_accuracy"
    elif method_name == "metric_gate_hd":
        method = MetricGateHDMethod(
            encoder=encoder,
            updater=updater,
            num_classes=num_classes,
            interaction_dim=int(method_cfg.get("interaction_dim", 0)),
            alpha_init=float(method_cfg.get("alpha_init", 0.5)),
            alpha_lr=float(method_cfg.get("alpha_lr", 0.05)),
            alpha_l2=float(method_cfg.get("alpha_l2", 0.01)),
            alpha_max=float(method_cfg.get("alpha_max", 4.0)),
            tail_gamma=float(method_cfg.get("tail_gamma", 0.1)),
            tail_tau=float(method_cfg.get("tail_tau", 0.1)),
            interaction_seed=int(method_cfg.get("interaction_seed", 13)),
            update_mode=str(method_cfg.get("update_mode", "separate")),
            metric=metric,
            debug=bool(method_cfg.get("debug", False)),
        )
        metric_key = "mean_global_accuracy"
    else:
        raise ValueError(f"Unsupported HD method: {method_name}")

    return method, metric_key


def build_nn_method(cfg: dict, input_dim: int, num_classes: int, device: torch.device):
    _ = input_dim
    method_cfg = cfg["method"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    method_name = method_cfg["name"]
    dataset_name = str(cfg["dataset"]["name"])

    hidden_dim = int(model_cfg.get("hidden_dim", train_cfg.get("mlp_hidden_dim", 256)))
    num_hidden_layers = int(model_cfg.get("num_hidden_layers", train_cfg.get("mlp_num_hidden_layers", 4)))
    branch_layers = int(model_cfg.get("branch_layers", train_cfg.get("dfl_branch_layers", num_hidden_layers)))
    dropout = float(model_cfg.get("dropout", 0.0))
    state_device = "cpu"
    optimizer_name = str(train_cfg.get("optimizer", "sgd"))

    if method_name == "fedavg_mlp":
        method = FedAvgMLPMethod(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            local_epochs=train_cfg["local_epochs"],
            batch_size=train_cfg["batch_size"],
            lr=float(train_cfg.get("fedavg_lr", train_cfg.get("lr", 0.01))),
            lr_decay=float(train_cfg.get("fedavg_lr_decay", 1.0)),
            momentum=float(train_cfg.get("fedavg_momentum", train_cfg.get("momentum", 0.0))),
            weight_decay=float(train_cfg.get("fedavg_weight_decay", train_cfg.get("weight_decay", 0.0))),
            optimizer_name=str(train_cfg.get("fedavg_optimizer", optimizer_name)),
            dropout=dropout,
            device=device,
            state_device=state_device,
            debug=method_cfg.get("debug", False),
        )
        metric_key = "mean_global_accuracy"
    elif method_name == "fedprox_mlp":
        method = FedProxMLPMethod(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            local_epochs=train_cfg["local_epochs"],
            batch_size=train_cfg["batch_size"],
            lr=float(train_cfg.get("fedprox_lr", train_cfg.get("lr", 0.01))),
            prox_mu=float(train_cfg.get("fedprox_mu", 0.01)),
            lr_decay=float(train_cfg.get("fedprox_lr_decay", 1.0)),
            momentum=float(train_cfg.get("fedprox_momentum", train_cfg.get("momentum", 0.0))),
            weight_decay=float(train_cfg.get("fedprox_weight_decay", train_cfg.get("weight_decay", 0.0))),
            optimizer_name=str(train_cfg.get("fedprox_optimizer", optimizer_name)),
            dropout=dropout,
            device=device,
            state_device=state_device,
            debug=method_cfg.get("debug", False),
        )
        metric_key = "mean_global_accuracy"
    elif method_name == "fedavg_cnn":
        if dataset_name != "femnist":
            raise ValueError(f"fedavg_cnn is only supported for femnist, got {dataset_name}")
        method = FedAvgCNNMethod(
            num_classes=num_classes,
            local_epochs=train_cfg["local_epochs"],
            batch_size=train_cfg["batch_size"],
            lr=float(train_cfg.get("fedavg_lr", train_cfg.get("lr", 0.01))),
            cnn_hidden_dim=hidden_dim,
            lr_decay=float(train_cfg.get("fedavg_lr_decay", 1.0)),
            momentum=float(train_cfg.get("fedavg_momentum", train_cfg.get("momentum", 0.0))),
            weight_decay=float(train_cfg.get("fedavg_weight_decay", train_cfg.get("weight_decay", 0.0))),
            optimizer_name=str(train_cfg.get("fedavg_optimizer", optimizer_name)),
            dropout=dropout,
            device=device,
            state_device=state_device,
            debug=method_cfg.get("debug", False),
        )
        metric_key = "mean_global_accuracy"
    elif method_name == "fedprox_cnn":
        if dataset_name != "femnist":
            raise ValueError(f"fedprox_cnn is only supported for femnist, got {dataset_name}")
        method = FedProxCNNMethod(
            num_classes=num_classes,
            local_epochs=train_cfg["local_epochs"],
            batch_size=train_cfg["batch_size"],
            lr=float(train_cfg.get("fedprox_lr", train_cfg.get("lr", 0.01))),
            prox_mu=float(train_cfg.get("fedprox_mu", 0.01)),
            cnn_hidden_dim=hidden_dim,
            lr_decay=float(train_cfg.get("fedprox_lr_decay", 1.0)),
            momentum=float(train_cfg.get("fedprox_momentum", train_cfg.get("momentum", 0.0))),
            weight_decay=float(train_cfg.get("fedprox_weight_decay", train_cfg.get("weight_decay", 0.0))),
            optimizer_name=str(train_cfg.get("fedprox_optimizer", optimizer_name)),
            dropout=dropout,
            device=device,
            state_device=state_device,
            debug=method_cfg.get("debug", False),
        )
        metric_key = "mean_global_accuracy"
    elif method_name == "ditto_mlp":
        method = DittoMLPMethod(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            local_epochs=train_cfg["local_epochs"],
            batch_size=train_cfg["batch_size"],
            global_lr=float(train_cfg.get("ditto_global_lr", train_cfg.get("lr", 0.01))),
            personal_lr=float(train_cfg.get("ditto_personal_lr", train_cfg.get("lr", 0.01))),
            lambda_reg=float(train_cfg.get("ditto_lambda", 1e-2)),
            global_lr_decay=float(train_cfg.get("ditto_global_lr_decay", 1.0)),
            personal_lr_decay=float(train_cfg.get("ditto_personal_lr_decay", 1.0)),
            momentum=float(train_cfg.get("ditto_momentum", train_cfg.get("momentum", 0.0))),
            weight_decay=float(train_cfg.get("ditto_weight_decay", train_cfg.get("weight_decay", 0.0))),
            optimizer_name=str(train_cfg.get("ditto_optimizer", optimizer_name)),
            dropout=dropout,
            device=device,
            state_device=state_device,
            debug=method_cfg.get("debug", False),
        )
        metric_key = "mean_personalized_accuracy"
    elif method_name == "ditto_cnn":
        if dataset_name != "femnist":
            raise ValueError(f"ditto_cnn is only supported for femnist, got {dataset_name}")
        method = DittoCNNMethod(
            num_classes=num_classes,
            local_epochs=train_cfg["local_epochs"],
            batch_size=train_cfg["batch_size"],
            global_lr=float(train_cfg.get("ditto_global_lr", train_cfg.get("lr", 0.01))),
            personal_lr=float(train_cfg.get("ditto_personal_lr", train_cfg.get("lr", 0.01))),
            lambda_reg=float(train_cfg.get("ditto_lambda", 1e-2)),
            cnn_hidden_dim=hidden_dim,
            global_lr_decay=float(train_cfg.get("ditto_global_lr_decay", 1.0)),
            personal_lr_decay=float(train_cfg.get("ditto_personal_lr_decay", 1.0)),
            momentum=float(train_cfg.get("ditto_momentum", train_cfg.get("momentum", 0.0))),
            weight_decay=float(train_cfg.get("ditto_weight_decay", train_cfg.get("weight_decay", 0.0))),
            optimizer_name=str(train_cfg.get("ditto_optimizer", optimizer_name)),
            dropout=dropout,
            device=device,
            state_device=state_device,
            debug=method_cfg.get("debug", False),
        )
        metric_key = "mean_personalized_accuracy"
    elif method_name == "pfedme_mlp":
        method = PFedMeMLPMethod(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            local_epochs=train_cfg["local_epochs"],
            batch_size=train_cfg["batch_size"],
            personal_lr=float(train_cfg.get("pfedme_personal_lr", train_cfg.get("lr", 0.01))),
            reference_lr=float(train_cfg.get("pfedme_reference_lr", train_cfg.get("pfedme_global_lr", 0.5))),
            lambda_reg=float(train_cfg.get("pfedme_lambda", 1e-2)),
            beta=float(train_cfg.get("pfedme_beta", 0.5)),
            personal_steps=int(train_cfg.get("pfedme_personal_steps", 1)),
            personal_lr_decay=float(train_cfg.get("pfedme_personal_lr_decay", 1.0)),
            momentum=float(train_cfg.get("pfedme_momentum", train_cfg.get("momentum", 0.0))),
            weight_decay=float(train_cfg.get("pfedme_weight_decay", train_cfg.get("weight_decay", 0.0))),
            optimizer_name=str(train_cfg.get("pfedme_optimizer", optimizer_name)),
            dropout=dropout,
            device=device,
            state_device=state_device,
            debug=method_cfg.get("debug", False),
        )
        metric_key = "mean_personalized_accuracy"
    elif method_name == "pfedme_cnn":
        if dataset_name != "femnist":
            raise ValueError(f"pfedme_cnn is only supported for femnist, got {dataset_name}")
        method = PFedMeCNNMethod(
            num_classes=num_classes,
            local_epochs=train_cfg["local_epochs"],
            batch_size=train_cfg["batch_size"],
            personal_lr=float(train_cfg.get("pfedme_personal_lr", train_cfg.get("lr", 0.01))),
            reference_lr=float(train_cfg.get("pfedme_reference_lr", train_cfg.get("pfedme_global_lr", 0.5))),
            lambda_reg=float(train_cfg.get("pfedme_lambda", 1e-2)),
            beta=float(train_cfg.get("pfedme_beta", 0.5)),
            cnn_hidden_dim=hidden_dim,
            personal_steps=int(train_cfg.get("pfedme_personal_steps", 1)),
            personal_lr_decay=float(train_cfg.get("pfedme_personal_lr_decay", 1.0)),
            momentum=float(train_cfg.get("pfedme_momentum", train_cfg.get("momentum", 0.0))),
            weight_decay=float(train_cfg.get("pfedme_weight_decay", train_cfg.get("weight_decay", 0.0))),
            optimizer_name=str(train_cfg.get("pfedme_optimizer", optimizer_name)),
            dropout=dropout,
            device=device,
            state_device=state_device,
            debug=method_cfg.get("debug", False),
        )
        metric_key = "mean_personalized_accuracy"
    elif method_name == "dfl_mlp":
        method = DFLMLPMethod(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            branch_layers=branch_layers,
            local_epochs=train_cfg["local_epochs"],
            batch_size=train_cfg["batch_size"],
            lr=float(train_cfg.get("dfl_lr", train_cfg.get("lr", 0.01))),
            align_weight=float(train_cfg.get("dfl_align_weight", 1.0)),
            disentangle_weight=float(train_cfg.get("dfl_disentangle_weight", 0.1)),
            lr_decay=float(train_cfg.get("dfl_lr_decay", 1.0)),
            momentum=float(train_cfg.get("dfl_momentum", train_cfg.get("momentum", 0.0))),
            weight_decay=float(train_cfg.get("dfl_weight_decay", train_cfg.get("weight_decay", 0.0))),
            optimizer_name=str(train_cfg.get("dfl_optimizer", optimizer_name)),
            dropout=dropout,
            device=device,
            state_device=state_device,
            debug=method_cfg.get("debug", False),
        )
        metric_key = "mean_personalized_accuracy"
    elif method_name == "dfl_cnn":
        if dataset_name != "femnist":
            raise ValueError(f"dfl_cnn is only supported for femnist, got {dataset_name}")
        method = DFLCNNMethod(
            num_classes=num_classes,
            local_epochs=train_cfg["local_epochs"],
            batch_size=train_cfg["batch_size"],
            lr=float(train_cfg.get("dfl_lr", train_cfg.get("lr", 0.01))),
            cnn_hidden_dim=hidden_dim,
            align_weight=float(train_cfg.get("dfl_align_weight", 1.0)),
            disentangle_weight=float(train_cfg.get("dfl_disentangle_weight", 0.1)),
            lr_decay=float(train_cfg.get("dfl_lr_decay", 1.0)),
            momentum=float(train_cfg.get("dfl_momentum", train_cfg.get("momentum", 0.0))),
            weight_decay=float(train_cfg.get("dfl_weight_decay", train_cfg.get("weight_decay", 0.0))),
            optimizer_name=str(train_cfg.get("dfl_optimizer", optimizer_name)),
            dropout=dropout,
            device=device,
            state_device=state_device,
            debug=method_cfg.get("debug", False),
        )
        metric_key = "mean_personalized_accuracy"
    else:
        raise ValueError(f"Unsupported NN method: {method_name}")

    return method, metric_key


def build_method_cfg(method_key: str, dataset_name: str, hd_dim: int) -> tuple[str, dict, dict]:
    if is_commonbasis_method_key(method_key):
        spec = COMMONBASIS_METHOD_SPECS[method_key]
        method_cfg: dict[str, object] = {"name": str(spec["name"])}
        defaults = spec.get("method_defaults")
        if isinstance(defaults, dict):
            method_cfg.update(copy.deepcopy(defaults))
        return "hd", method_cfg, build_hd_model_cfg(hd_dim)
    if method_key in OPTIONAL_HD_METHODS:
        return "hd", {"name": method_key}, build_hd_model_cfg(hd_dim)
    if method_key in NN_METHODS:
        return "nn", {"name": method_key}, build_nn_model_cfg(method_key, dataset_name)
    raise ValueError(f"Unsupported method key: {method_key}")


def build_config(dataset_cfg: dict, method_key: str, args: argparse.Namespace) -> tuple[str, dict]:
    family, method_cfg, model_cfg = build_method_cfg(method_key, dataset_cfg["name"], args.hd_dim)
    if family == "hd":
        model_cfg["cosine_random_phase"] = bool(getattr(args, "hd_cosine_random_phase", False))
    train_cfg = build_base_train(args, family=family)
    fusion_alpha = getattr(args, "subspace_fusion_alpha", None)
    fusion_grid = getattr(args, "subspace_fusion_grid", None)
    if fusion_grid is not None:
        alpha_grid = [float(value) for value in fusion_grid]
    else:
        alpha_grid = (
            [float(fusion_alpha)]
            if fusion_alpha is not None
            else [0.0, 0.25, 0.5, 0.75, 1.0]
        )
    if method_key in {"fedavg_cnn", "fedprox_cnn", "ditto_cnn", "pfedme_cnn", "dfl_cnn"}:
        train_cfg.update(
            {
                "optimizer": str(getattr(args, "cnn_optimizer", "sgd")),
                "lr": float(getattr(args, "cnn_lr", 0.06)),
                "momentum": float(getattr(args, "cnn_momentum", 0.0)),
                "weight_decay": float(getattr(args, "cnn_weight_decay", 0.0)),
            }
        )
    if method_key in {"fedprox_mlp", "fedprox_cnn"}:
        train_cfg["fedprox_mu"] = float(getattr(args, "fedprox_mu", 0.01))
    elif method_key in {"ditto_mlp", "ditto_cnn"}:
        train_cfg["ditto_lambda"] = 0.01
    elif method_key in {"pfedme_mlp", "pfedme_cnn"}:
        train_cfg.update(
            {
                "pfedme_lambda": 0.01,
                "pfedme_beta": 0.5,
                "pfedme_reference_lr": 0.5,
                "pfedme_personal_steps": 1,
            }
        )
    elif method_key in {"dfl_mlp", "dfl_cnn"}:
        train_cfg.update(
            {
                "dfl_align_weight": float(getattr(args, "dfl_align_weight", 1.0)),
                "dfl_disentangle_weight": float(getattr(args, "dfl_disentangle_weight", 0.1)),
            }
        )
    elif method_key in SUBSPACE_METHOD_KEYS:
        method_cfg.update(_base_subspace_method_cfg(args, alpha_grid))
        if is_commonbasis_method_key(method_key):
            _apply_commonbasis_method_cfg_overrides(method_cfg, args)
    elif method_key in {"dchb_hd", "dchb_centralized_hd"}:
        _apply_dchb_method_cfg_overrides(method_cfg, args)
    elif method_key == "latent_group_dro_hd":
        _apply_lgdro_method_cfg_overrides(method_cfg, args)
    elif method_key == "metric_gate_hd":
        _apply_metric_gate_method_cfg_overrides(method_cfg, args)
    elif method_key == "masking_anticollapse_hd":
        _apply_fd_masking_method_cfg_overrides(method_cfg, args)
    elif method_key == "mlp_group_learned_hd":
        _apply_fd_group_method_cfg_overrides(method_cfg, args)
    elif method_key in {"block_packed_group_hd", "superposition_packed_group_hd", "hash_packed_group_hd"}:
        _apply_fd_packed_method_cfg_overrides(method_cfg, args)
    elif method_key == "naive_group_ensemble":
        _apply_fd_naive_group_method_cfg_overrides(method_cfg, args)
    elif method_key == "residual_packed_group_hd":
        _apply_fd_residual_method_cfg_overrides(method_cfg, args)
    cfg = {
        "dataset": copy.deepcopy(dataset_cfg),
        "model": model_cfg,
        "method": method_cfg,
        "train": train_cfg,
        "runtime": {"device": args.device},
    }
    return family, cfg


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


def stratified_subsample_client_train_data(
    client: ClientData,
    max_samples: int,
    *,
    seed: int,
) -> ClientData:
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
                sum(
                    allocation[idx] < int(client.y_train.numel())
                    for idx, client in enumerate(clients)
                )
            ),
        }
    )
    return capped_clients, info


def maybe_apply_client_regime(
    clients: list[ClientData],
    *,
    regime: str,
) -> tuple[list[ClientData], dict[str, object]]:
    original_num_clients = int(len(clients))
    if original_num_clients <= 0:
        raise RuntimeError("No clients available for client-regime transformation.")

    if regime == "native":
        return clients, {
            "client_regime": "native",
            "original_num_clients": original_num_clients,
            "effective_num_clients": original_num_clients,
            "pooled_applied": False,
        }

    if regime != "pooled_all":
        raise ValueError(f"Unsupported client regime: {regime}")

    if original_num_clients == 1:
        return clients, {
            "client_regime": "pooled_all",
            "original_num_clients": original_num_clients,
            "effective_num_clients": 1,
            "pooled_applied": False,
        }

    pooled = ClientData(
        client_id="pooled_all",
        x_train=torch.cat([client.x_train for client in clients], dim=0),
        y_train=torch.cat([client.y_train for client in clients], dim=0),
        x_test=torch.cat([client.x_test for client in clients], dim=0),
        y_test=torch.cat([client.y_test for client in clients], dim=0),
    )
    return [pooled], {
        "client_regime": "pooled_all",
        "original_num_clients": original_num_clients,
        "effective_num_clients": 1,
        "pooled_applied": True,
    }


def empirical_prototype_matrix(
    encoder: CosineProjectionEncoder,
    client,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        x_hv = encoder.encode(client.x_train)
        y = client.y_train.to(x_hv.device).long()
        sums = torch.zeros(num_classes, x_hv.shape[1], device=x_hv.device, dtype=x_hv.dtype)
        counts = torch.zeros(num_classes, device=x_hv.device, dtype=x_hv.dtype)
        sums.index_add_(0, y, x_hv)
        counts.index_add_(0, y, torch.ones_like(y, dtype=x_hv.dtype))
        prototypes = sums / counts.clamp_min(1.0).unsqueeze(1)
        prototypes[counts <= 0] = 0.0
        return prototypes.to("cpu"), counts.to("cpu")


def covariance_from_prototypes(prototype_mats: list[torch.Tensor]) -> torch.Tensor:
    hd_dim = prototype_mats[0].shape[1]
    covariance = torch.zeros(hd_dim, hd_dim, dtype=torch.float32)
    for matrix in prototype_mats:
        covariance.add_(matrix.T @ matrix)
    return covariance


def top_basis_from_covariance(covariance: torch.Tensor, max_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = torch.flip(eigenvalues, dims=[0])
    eigenvectors = torch.flip(eigenvectors, dims=[1])
    resolved_rank = max(1, min(int(max_rank), eigenvectors.shape[1]))
    return eigenvalues[:resolved_rank], eigenvectors[:, :resolved_rank]


def per_client_explained_ratios(prototype_mats: list[torch.Tensor], basis: torch.Tensor) -> list[float]:
    ratios: list[float] = []
    for matrix in prototype_mats:
        total_energy = float((matrix * matrix).sum().item())
        if total_energy <= 1e-12:
            ratios.append(0.0)
            continue
        projected = matrix @ basis
        explained = float((projected * projected).sum().item())
        ratios.append(explained / total_energy)
    return ratios


def null_eta_records(
    prototype_mats: list[torch.Tensor],
    ranks: list[int],
    *,
    null_repeats: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    hd_dim = prototype_mats[0].shape[1]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    by_rank: dict[int, list[float]] = {int(rank): [] for rank in ranks}

    for _ in range(max(0, int(null_repeats))):
        covariance = torch.zeros(hd_dim, hd_dim, dtype=torch.float32)
        total_energy = 0.0
        for matrix in prototype_mats:
            permutation = torch.randperm(hd_dim, generator=generator)
            permuted = matrix[:, permutation]
            covariance.add_(permuted.T @ permuted)
            total_energy += float((permuted * permuted).sum().item())
        eigenvalues, _ = top_basis_from_covariance(covariance, max(ranks))
        cumulative = torch.cumsum(eigenvalues, dim=0)
        denom = max(total_energy, 1e-12)
        for rank in ranks:
            resolved_rank = min(int(rank), cumulative.shape[0])
            by_rank[int(rank)].append(float(cumulative[resolved_rank - 1].item() / denom))

    result: dict[str, dict[str, float]] = {}
    for rank in ranks:
        stats = mean_std(by_rank[int(rank)])
        result[str(rank)] = {
            "eta_g_mean": stats["mean"],
            "eta_g_std": stats["std"],
        }
    return result


def summarize_shared_subspace_records(records: list[dict], ranks: list[int]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for rank in ranks:
        eta_values = [float(record["rank_metrics"][str(rank)]["eta_g"]) for record in records]
        gap_values = [
            float(record["rank_metrics"][str(rank)]["eta_g"])
            - float(record["null_rank_metrics"][str(rank)]["eta_g_mean"])
            for record in records
        ]
        client_mean_values = [float(record["rank_metrics"][str(rank)]["per_client_explained_mean"]) for record in records]
        client_std_values = [float(record["rank_metrics"][str(rank)]["per_client_explained_std"]) for record in records]
        null_mean_values = [float(record["null_rank_metrics"][str(rank)]["eta_g_mean"]) for record in records]
        summary[str(rank)] = {
            "eta_g_mean": mean_std(eta_values)["mean"],
            "eta_g_std": mean_std(eta_values)["std"],
            "eta_gap_mean": mean_std(gap_values)["mean"],
            "eta_gap_std": mean_std(gap_values)["std"],
            "per_client_explained_mean": mean_std(client_mean_values)["mean"],
            "per_client_explained_std": mean_std(client_std_values)["mean"],
            "null_eta_g_mean": mean_std(null_mean_values)["mean"],
            "null_eta_g_std": mean_std(null_mean_values)["std"],
        }
    return summary


def render_shared_subspace_markdown(report: dict) -> str:
    lines = ["# Shared Subspace Existence", ""]
    lines.append(f"- device: `{report['device']}`")
    lines.append(f"- dataset seeds: `{report['dataset_seeds']}`")
    lines.append(f"- encoder seeds: `{report['encoder_seeds']}`")
    lines.append(f"- shared ranks: `{report['shared_ranks']}`")
    lines.append(f"- null repeats: `{report['null_repeats']}`")
    lines.append(f"- pilot: `{report['pilot']}`")
    if report["run_config"].get("large_dataset_train_threshold") is not None:
        lines.append(
            f"- large-dataset train cap: threshold `{report['run_config']['large_dataset_train_threshold']}`, "
            f"cap `{report['run_config']['large_dataset_train_cap']}`"
        )
    lines.append("")
    for dataset_name, dataset_report in report["datasets"].items():
        lines.append(f"## {dataset_name}")
        lines.append("")
        lines.append(
            f"- clients: `{dataset_report['num_clients']}`, classes: `{dataset_report['num_classes']}`, "
            f"input_dim: `{dataset_report['input_dim']}`, hd_dim: `{dataset_report['hd_dim']}`"
        )
        sampling_record = dataset_report.get("sampling_records", [{}])[0]
        if sampling_record.get("applied"):
            lines.append(
                f"- train sampling: `{sampling_record['total_train_samples_before']} -> "
                f"{sampling_record['total_train_samples_after']}` across "
                f"`{sampling_record['clients_modified']}` clients"
            )
        lines.append("")
        lines.append("| rank | eta_g mean | eta_g std | null eta mean | gap mean | client explained mean | client explained std |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for rank in report["shared_ranks"]:
            item = dataset_report["summary"][str(rank)]
            lines.append(
                f"| {rank} | {item['eta_g_mean']:.4f} | {item['eta_g_std']:.4f} | "
                f"{item['null_eta_g_mean']:.4f} | {item['eta_gap_mean']:.4f} | "
                f"{item['per_client_explained_mean']:.4f} | {item['per_client_explained_std']:.4f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def run_shared_subspace_analysis(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    dataset_specs = load_dataset_specs()
    out_dir = Path(args.analysis_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_out = out_dir / "shared_subspace_existence_latest.json"
    md_out = out_dir / "shared_subspace_existence_latest.md"

    report: dict = {
        "analysis": "shared_subspace",
        "device": str(device),
        "dataset_seeds": list(args.seeds),
        "encoder_seeds": list(args.encoder_seeds),
        "shared_ranks": [int(value) for value in args.shared_ranks],
        "null_repeats": int(args.null_repeats),
        "pilot": bool(args.pilot),
        "run_config": {
            "datasets": list(args.datasets),
            "device": args.device,
            "seeds": list(args.seeds),
            "encoder_seeds": list(args.encoder_seeds),
            "shared_ranks": [int(value) for value in args.shared_ranks],
            "null_repeats": int(args.null_repeats),
            "hd_dim": int(args.hd_dim),
            "pilot": bool(args.pilot),
            "pilot_max_clients": int(args.pilot_max_clients),
            "large_dataset_train_threshold": (
                None if args.large_dataset_train_threshold is None else int(args.large_dataset_train_threshold)
            ),
            "large_dataset_train_cap": (
                None if args.large_dataset_train_cap is None else int(args.large_dataset_train_cap)
            ),
        },
        "datasets": {},
    }

    for dataset_name in args.datasets:
        dataset_spec = copy.deepcopy(dataset_specs[dataset_name])
        if args.pilot:
            dataset_spec = apply_pilot_overrides(dataset_spec, args)

        dataset_records: list[dict] = []
        sampling_records: list[dict] = []
        num_clients = None
        num_classes = None
        input_dim = None
        resolved_ranks: list[int] | None = None

        for dataset_seed in args.seeds:
            seeded_dataset = apply_seed_to_dataset(dataset_spec, dataset_seed)
            set_seed(dataset_seed)
            print(f"[shared-subspace] dataset={dataset_name} dataset_seed={dataset_seed} phase=load")
            adapter = build_dataset_adapter({"dataset": seeded_dataset}, torch.device("cpu"))
            clients = adapter.load_clients()
            clients, sampling_info = maybe_cap_large_dataset_train_data(
                clients,
                threshold=args.large_dataset_train_threshold,
                total_cap=args.large_dataset_train_cap,
                seed=(int(dataset_seed) * 7919) + 17,
            )
            sampling_records.append({"dataset_seed": int(dataset_seed), **sampling_info})
            if sampling_info["applied"]:
                print(
                    f"[shared-subspace] dataset={dataset_name} dataset_seed={dataset_seed} "
                    f"phase=train_cap before={sampling_info['total_train_samples_before']} "
                    f"after={sampling_info['total_train_samples_after']}"
                )
            num_clients = len(clients)
            input_dim = int(clients[0].x_train.shape[1])
            num_classes = int(adapter.num_classes())
            resolved_ranks = sorted({max(1, min(int(rank), int(args.hd_dim))) for rank in args.shared_ranks})

            for encoder_seed in args.encoder_seeds:
                set_seed(encoder_seed)
                encoder = CosineProjectionEncoder(
                    input_dim=input_dim,
                    hd_dim=int(args.hd_dim),
                    binary=False,
                    device=device,
                )
                prototype_mats: list[torch.Tensor] = []
                observed_classes: list[int] = []
                total_energy = 0.0

                print(
                    f"[shared-subspace] dataset={dataset_name} dataset_seed={dataset_seed} "
                    f"encoder_seed={encoder_seed} phase=encode"
                )
                for client in clients:
                    prototypes, counts = empirical_prototype_matrix(encoder, client, num_classes)
                    prototype_mats.append(prototypes)
                    observed_classes.append(int((counts > 0).sum().item()))
                    total_energy += float((prototypes * prototypes).sum().item())

                covariance = covariance_from_prototypes(prototype_mats)
                eigenvalues, basis = top_basis_from_covariance(covariance, max(resolved_ranks))
                cumulative = torch.cumsum(eigenvalues, dim=0)

                rank_metrics: dict[str, dict[str, float]] = {}
                for rank in resolved_ranks:
                    resolved_rank = min(int(rank), basis.shape[1])
                    rank_basis = basis[:, :resolved_rank]
                    per_client = per_client_explained_ratios(prototype_mats, rank_basis)
                    per_client_mean = mean_std(per_client)
                    eta_g = float(cumulative[resolved_rank - 1].item() / max(total_energy, 1e-12))
                    rank_metrics[str(rank)] = {
                        "eta_g": eta_g,
                        "per_client_explained_mean": per_client_mean["mean"],
                        "per_client_explained_std": per_client_mean["std"],
                        "per_client_explained_min": float(min(per_client) if per_client else 0.0),
                        "per_client_explained_max": float(max(per_client) if per_client else 0.0),
                    }

                null_rank_metrics = null_eta_records(
                    prototype_mats,
                    resolved_ranks,
                    null_repeats=int(args.null_repeats),
                    seed=(int(dataset_seed) * 1009) + int(encoder_seed),
                )
                dataset_records.append(
                    {
                        "dataset_seed": int(dataset_seed),
                        "encoder_seed": int(encoder_seed),
                        "num_clients": int(len(clients)),
                        "num_classes": int(num_classes),
                        "input_dim": int(input_dim),
                        "hd_dim": int(args.hd_dim),
                        "mean_observed_classes": float(np.mean(observed_classes)),
                        "total_energy": float(total_energy),
                        "top_eigenvalues": [float(value) for value in eigenvalues.tolist()],
                        "rank_metrics": rank_metrics,
                        "null_rank_metrics": null_rank_metrics,
                    }
                )
                print(
                    f"[shared-subspace] dataset={dataset_name} dataset_seed={dataset_seed} "
                    f"encoder_seed={encoder_seed} status=done eta_r{resolved_ranks[-1]}="
                    f"{rank_metrics[str(resolved_ranks[-1])]['eta_g']:.4f}"
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        if num_clients is None or num_classes is None or input_dim is None or resolved_ranks is None:
            raise RuntimeError(f"Failed to collect shared-subspace records for dataset {dataset_name}")

        report["datasets"][dataset_name] = {
            "num_clients": int(num_clients),
            "num_classes": int(num_classes),
            "input_dim": int(input_dim),
            "hd_dim": int(args.hd_dim),
            "records": dataset_records,
            "sampling_records": sampling_records,
            "summary": summarize_shared_subspace_records(dataset_records, resolved_ranks),
            "dataset_spec": dataset_spec,
        }

    json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_out.write_text(render_shared_subspace_markdown(report), encoding="utf-8")
    print(f"saved_json={json_out}")
    print(f"saved_md={md_out}")


def render_markdown(report: dict) -> str:
    lines = ["# Seven Dataset Benchmark Latest", ""]
    lines.append(f"- device: `{report['device']}`")
    lines.append(f"- seeds: `{report['seeds']}`")
    lines.append(f"- pilot: `{report['pilot']}`")
    lines.append(f"- measure energy: `{report['run_config'].get('measure_energy', False)}`")
    if report["run_config"].get("large_dataset_train_threshold") is not None:
        lines.append(
            f"- large-dataset train cap: threshold `{report['run_config']['large_dataset_train_threshold']}`, "
            f"cap `{report['run_config']['large_dataset_train_cap']}`"
        )
    lines.append("")
    for dataset_name, dataset_report in report["datasets"].items():
        lines.append(f"## {dataset_name}")
        lines.append("")
        lines.append(
            f"- classes: `{dataset_report['num_classes']}`, chance accuracy: `{dataset_report['chance_accuracy']:.4f}`"
        )
        sampling_record = dataset_report.get("sampling_records", [{}])[0]
        if sampling_record.get("applied"):
            lines.append(
                f"- train sampling: `{sampling_record['total_train_samples_before']} -> "
                f"{sampling_record['total_train_samples_after']}` across "
                f"`{sampling_record['clients_modified']}` clients"
            )
        lines.append("")
        if report["run_config"].get("measure_energy", False):
            lines.append("| method | family | primary metric | mean | std | collapsed | runtime mean (s) | gpu energy mean (J) | gpu avg power mean (W) |")
            lines.append("| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |")
        else:
            lines.append("| method | family | primary metric | mean | std | collapsed | runtime mean (s) |")
            lines.append("| --- | --- | --- | ---: | ---: | --- | ---: |")
        summary_items = sorted(
            dataset_report["summary"].items(),
            key=lambda item: item[1]["primary_mean"],
            reverse=True,
        )
        for method_key, item in summary_items:
            if report["run_config"].get("measure_energy", False):
                gpu_energy = "" if item.get("gpu_energy_j_mean") is None else f"{item['gpu_energy_j_mean']:.2f}"
                gpu_power = "" if item.get("gpu_avg_power_w_mean") is None else f"{item['gpu_avg_power_w_mean']:.2f}"
                lines.append(
                    f"| {method_key} | {item['family']} | {item['metric_key']} | "
                    f"{item['primary_mean']:.4f} | {item['primary_std']:.4f} | "
                    f"{'yes' if item['collapsed'] else 'no'} | {item['runtime_seconds_mean']:.2f} | "
                    f"{gpu_energy} | {gpu_power} |"
                )
            else:
                lines.append(
                    f"| {method_key} | {item['family']} | {item['metric_key']} | "
                    f"{item['primary_mean']:.4f} | {item['primary_std']:.4f} | "
                    f"{'yes' if item['collapsed'] else 'no'} | {item['runtime_seconds_mean']:.2f} |"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def run_benchmark(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    dataset_specs = load_dataset_specs()
    json_out = Path(args.json_out)
    md_out = Path(args.md_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "device": str(device),
        "seeds": args.seeds,
        "pilot": bool(args.pilot),
        "run_config": {
            "datasets": list(args.datasets),
            "methods": list(args.methods),
            "device": args.device,
            "seeds": list(args.seeds),
            "rounds": int(args.rounds),
            "local_epochs": int(args.local_epochs),
            "batch_size": int(args.batch_size),
            "hd_dim": int(args.hd_dim),
            "hd_lr": float(args.hd_lr),
            "hd_cosine_random_phase": bool(args.hd_cosine_random_phase),
            "nn_lr": float(args.nn_lr),
            "fedprox_mu": float(args.fedprox_mu),
            "pilot_rounds": int(args.pilot_rounds),
            "pilot_local_epochs": int(args.pilot_local_epochs),
            "pilot_max_clients": int(args.pilot_max_clients),
            "collapse_factor": float(args.collapse_factor),
            "measure_energy": bool(args.measure_energy),
            "power_sample_interval_ms": float(args.power_sample_interval_ms),
            "large_dataset_train_threshold": (
                None if args.large_dataset_train_threshold is None else int(args.large_dataset_train_threshold)
            ),
            "large_dataset_train_cap": (
                None if args.large_dataset_train_cap is None else int(args.large_dataset_train_cap)
            ),
            "client_regime": str(args.client_regime),
        },
        "datasets": {},
    }

    for dataset_name in args.datasets:
        dataset_spec = copy.deepcopy(dataset_specs[dataset_name])
        if args.pilot:
            dataset_spec = apply_pilot_overrides(dataset_spec, args)

        rows: list[dict] = []
        sampling_records: list[dict] = []
        num_classes = None

        for seed in args.seeds:
            seeded_dataset = apply_seed_to_dataset(dataset_spec, seed)
            family_cache: dict[str, dict] = {}

            for method_key in args.methods:
                family = "nn" if method_key in NN_METHODS else "hd"
                effective_dataset = apply_family_dataset_overrides(seeded_dataset, family=family)
                cfg_family, cfg = build_config(effective_dataset, method_key, args)
                if cfg_family not in family_cache:
                    print(f"[benchmark] dataset={dataset_name} seed={seed} family={cfg_family} phase=load")
                    adapter = build_dataset_adapter({"dataset": effective_dataset}, torch.device("cpu"))
                    clients = adapter.load_clients()
                    clients, sampling_info = maybe_cap_large_dataset_train_data(
                        clients,
                        threshold=args.large_dataset_train_threshold,
                        total_cap=args.large_dataset_train_cap,
                        seed=(int(seed) * 7919) + 29,
                    )
                    clients, regime_info = maybe_apply_client_regime(
                        clients,
                        regime=str(args.client_regime),
                    )
                    family_cache[cfg_family] = {
                        "clients": clients,
                        "input_dim": clients[0].x_train.shape[1],
                        "num_classes": adapter.num_classes(),
                        "chance_accuracy": 1.0 / float(adapter.num_classes()),
                        "sampling_info": sampling_info,
                        "regime_info": regime_info,
                    }
                    sampling_records.append({"seed": int(seed), **sampling_info, **regime_info})
                    if regime_info["pooled_applied"]:
                        print(
                            f"[benchmark] dataset={dataset_name} seed={seed} family={cfg_family} phase=client_regime "
                            f"mode={args.client_regime} clients={regime_info['original_num_clients']}->"
                            f"{regime_info['effective_num_clients']}"
                        )
                    if sampling_info["applied"]:
                        print(
                            f"[benchmark] dataset={dataset_name} seed={seed} family={cfg_family} phase=train_cap "
                            f"before={sampling_info['total_train_samples_before']} "
                            f"after={sampling_info['total_train_samples_after']}"
                        )
                cached = family_cache[cfg_family]
                clients = cached["clients"]
                input_dim = int(cached["input_dim"])
                num_classes = int(cached["num_classes"])
                chance_accuracy = float(cached["chance_accuracy"])
                set_seed(seed)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                monitor = RunEnergyMonitor(
                    device=device,
                    enabled=bool(args.measure_energy),
                    sample_interval_sec=float(args.power_sample_interval_ms) / 1000.0,
                )
                start = time.perf_counter()
                with monitor:
                    if cfg_family == "hd":
                        method, metric_key = build_hd_method(cfg, input_dim, num_classes, device)
                        result = FederatedRunner(
                            method=method,
                            rounds=cfg["train"]["rounds"],
                            client_participation=float(cfg["train"].get("client_participation", 1.0)),
                            seed=seed,
                        ).run(clients)
                    else:
                        method, metric_key = build_nn_method(cfg, input_dim, num_classes, device)
                        result = NNFederatedRunner(
                            method=method,
                            rounds=cfg["train"]["rounds"],
                            client_participation=float(cfg["train"].get("client_participation", 1.0)),
                            seed=seed,
                        ).run(clients)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                runtime_seconds = time.perf_counter() - start
                energy_metrics = monitor.stop(elapsed_seconds=runtime_seconds)
                final_metrics = result["history"][-1]
                primary_value = float(final_metrics.get(metric_key, 0.0))
                rows.append(
                    {
                        "dataset": dataset_name,
                        "seed": seed,
                        "method": method_key,
                        "family": cfg_family,
                        "metric_key": metric_key,
                        "chance_accuracy": chance_accuracy,
                        "config": cfg,
                        "final_metrics": final_metrics,
                        "primary_value": primary_value,
                        "runtime_seconds": runtime_seconds,
                        "energy_metrics": energy_metrics,
                    }
                )
                gpu_energy_text = ""
                if energy_metrics.get("gpu_energy_j") is not None:
                    gpu_energy_text = f" gpu_energy_j={float(energy_metrics['gpu_energy_j']):.2f}"
                print(
                    f"[benchmark] dataset={dataset_name} seed={seed} method={method_key} "
                    f"primary_{metric_key}={primary_value:.4f} runtime_seconds={runtime_seconds:.2f}{gpu_energy_text}"
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        if num_classes is None:
            raise RuntimeError(f"Failed to load dataset {dataset_name}")

        summary: dict[str, dict] = {}
        for method_key in args.methods:
            method_rows = [row for row in rows if row["method"] == method_key]
            if not method_rows:
                continue
            primary = mean_std([row["primary_value"] for row in method_rows])
            runtime_stats = mean_std([row["runtime_seconds"] for row in method_rows])
            gpu_energy_values = [
                float(row["energy_metrics"]["gpu_energy_j"])
                for row in method_rows
                if row.get("energy_metrics", {}).get("gpu_energy_j") is not None
            ]
            gpu_avg_power_values = [
                float(row["energy_metrics"]["gpu_avg_power_w"])
                for row in method_rows
                if row.get("energy_metrics", {}).get("gpu_avg_power_w") is not None
            ]
            cpu_energy_values = [
                float(row["energy_metrics"]["cpu_energy_j"])
                for row in method_rows
                if row.get("energy_metrics", {}).get("cpu_energy_j") is not None
            ]
            chance_accuracy = float(method_rows[0]["chance_accuracy"])
            collapsed = primary["mean"] <= float(args.collapse_factor) * chance_accuracy
            summary[method_key] = {
                "family": method_rows[0]["family"],
                "metric_key": method_rows[0]["metric_key"],
                "primary_mean": primary["mean"],
                "primary_std": primary["std"],
                "runtime_seconds_mean": runtime_stats["mean"],
                "gpu_energy_j_mean": mean_std(gpu_energy_values)["mean"] if gpu_energy_values else None,
                "gpu_energy_j_std": mean_std(gpu_energy_values)["std"] if gpu_energy_values else None,
                "gpu_avg_power_w_mean": mean_std(gpu_avg_power_values)["mean"] if gpu_avg_power_values else None,
                "gpu_avg_power_w_std": mean_std(gpu_avg_power_values)["std"] if gpu_avg_power_values else None,
                "cpu_energy_j_mean": mean_std(cpu_energy_values)["mean"] if cpu_energy_values else None,
                "cpu_energy_j_std": mean_std(cpu_energy_values)["std"] if cpu_energy_values else None,
                "chance_accuracy": chance_accuracy,
                "collapsed": collapsed,
            }
        report["datasets"][dataset_name] = {
            "num_classes": num_classes,
            "chance_accuracy": 1.0 / float(num_classes),
            "rows": rows,
            "sampling_records": sampling_records,
            "summary": summary,
            "dataset_spec": dataset_spec,
        }

    json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_out.write_text(render_markdown(report), encoding="utf-8")
    print(f"saved_json={json_out}")
    print(f"saved_md={md_out}")


def main() -> None:
    args = parse_args()
    if args.analysis == "shared_subspace":
        run_shared_subspace_analysis(args)
        return
    run_benchmark(args)


if __name__ == "__main__":
    main()
