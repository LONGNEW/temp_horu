"""Legacy ISOLET federated partition used by the 26CASES HoRU code path."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .federated import FederatedDataset, write_cache
from .legacy_contract import build_dirichlet_clients, build_explicit_clients, compact_explicit_split_labels, resolve_normalization_mode


def _read(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [value.strip() for value in line.strip().split(",")]
        if len(parts) < 2:
            continue
        rows.append([float(value) for value in parts[:-1]])
        labels.append(int(float(parts[-1])) - 1)
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def prepare_data(data_root: str | Path, source_root: str | Path, seed: int = 42, alpha: float = 5.0, preserve_original_split: bool = False) -> FederatedDataset:
    source = Path(source_root)
    x_train_raw, y_train_raw = _read(source / "isolet1+2+3+4.data")
    x_test_raw, y_test_raw = _read(source / "isolet5.data")
    normalization = resolve_normalization_mode("l2", default="l2")
    if preserve_original_split:
        grouped = compact_explicit_split_labels([
            (x_train_raw, y_train_raw, x_test_raw, y_test_raw),
        ])
        clients = build_explicit_clients(
            grouped,
            ["client_0"],
            min_client_samples=10,
            limit_clients=None,
            normalization=normalization,
            max_train_samples_per_client=None,
            max_test_samples_per_client=None,
            seed=seed,
        )
    else:
        x = np.concatenate([x_train_raw, x_test_raw], axis=0)
        y = np.concatenate([y_train_raw, y_test_raw], axis=0)
        clients = build_dirichlet_clients(
            x,
            y,
            num_clients=8,
            alpha=alpha,
            test_size=0.3,
            min_client_samples=10,
            limit_clients=None,
            normalization=normalization,
            max_train_samples_per_client=None,
            max_test_samples_per_client=None,
            seed=seed,
        )
    if not clients:
        raise RuntimeError(f"no valid ISOLET clients found under {source}")
    manifest = {
        "source": str(source),
        "license": "UCI ISOLET",
        "parser": "isolet_legacy_v1",
        "clients": len(clients),
        "features": 617,
        "classes": 26,
        "num_clients": 8,
        "alpha": alpha,
        "test_size": 0.3,
        "preserve_original_split": preserve_original_split,
        "normalization": "l2",
        "seed": seed,
        "provenance": "LEGACY_26CASES_ISOLET",
    }
    return write_cache(FederatedDataset("isolet", clients, 617, 26, manifest), data_root)
