"""Legacy LEAF FEMNIST loader used by the 26CASES HoRU code path."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .federated import FederatedDataset, write_cache
from .legacy_contract import build_explicit_clients, compact_explicit_split_labels, resolve_normalization_mode, resolve_split_root, select_records


def _extract_xy(entry: dict) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(entry["x"], dtype=np.float32)
    y = np.asarray(entry["y"], dtype=np.int64)
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    return x, y


def _load_users(split_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    users: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for path in sorted(split_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload.get("user_data", {})
        for user in [str(value) for value in payload.get("users", [])]:
            if user in entries:
                users[user] = _extract_xy(entries[user])
    return users


def prepare_data(data_root: str | Path, source_root: str | Path, selection_seed: int = 42, limit_clients: int = 200) -> FederatedDataset:
    root = resolve_split_root(source_root)
    train_users = _load_users(root / "train")
    test_users = _load_users(root / "test")
    common = [(user, None) for user in sorted(set(train_users) & set(test_users))]
    selected_users = [user for user, _ in select_records(common, limit_clients, seed=selection_seed)]
    grouped = []
    client_ids = []
    for user in selected_users:
        x_train, y_train = train_users[user]
        x_test, y_test = test_users[user]
        grouped.append((x_train / 255.0, y_train, x_test / 255.0, y_test))
        client_ids.append(user)
    grouped = compact_explicit_split_labels(grouped)
    clients = build_explicit_clients(
        grouped,
        client_ids,
        min_client_samples=20,
        limit_clients=None,
        normalization=resolve_normalization_mode("l2", default="l2"),
        max_train_samples_per_client=256,
        max_test_samples_per_client=None,
        seed=selection_seed,
    )
    if not clients:
        raise RuntimeError(f"no valid FEMNIST clients found under {root}")
    manifest = {
        "source": str(root),
        "license": "LEAF/FEMNIST",
        "parser": "leaf_femnist_legacy_v1",
        "partition": "leaf_user",
        "clients": len(clients),
        "features": 784,
        "classes": 62,
        "limit_clients": limit_clients,
        "selection_seed": selection_seed,
        "preserve_original_split": True,
        "min_client_samples": 20,
        "max_train_samples_per_client": 256,
        "normalization": "l2",
        "provenance": "LEGACY_26CASES_FEMNIST",
    }
    return write_cache(FederatedDataset("femnist", clients, 784, 62, manifest), data_root)
