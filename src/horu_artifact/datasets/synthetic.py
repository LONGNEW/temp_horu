"""Legacy LEAF synthetic loader used by the 26CASES HoRU code path."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .federated import FederatedDataset, write_cache
from .legacy_contract import build_explicit_clients, compact_explicit_split_labels, resolve_normalization_mode, resolve_split_root


def _extract_xy(entry: dict) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(entry["x"], dtype=np.float32)
    y = np.asarray(entry["y"], dtype=np.int64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    return x, y


def _load_split_users(split_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    users: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for path in sorted(split_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for user in [str(value) for value in payload.get("users", [])]:
            entry = payload.get("user_data", {}).get(user)
            if entry is not None:
                users[user] = _extract_xy(entry)
    return users


def prepare_data(data_root: str | Path, source_root: str | Path, seed: int = 42, limit_clients: int = 30) -> FederatedDataset:
    root = resolve_split_root(source_root)
    train_users = _load_split_users(root / "train")
    test_users = _load_split_users(root / "test")
    grouped = []
    client_ids = []
    for user in sorted(set(train_users) & set(test_users)):
        x_train, y_train = train_users[user]
        x_test, y_test = test_users[user]
        grouped.append((x_train, y_train, x_test, y_test))
        client_ids.append(user)
    grouped = compact_explicit_split_labels(grouped)
    clients = build_explicit_clients(
        grouped,
        client_ids,
        min_client_samples=10,
        limit_clients=limit_clients,
        normalization=resolve_normalization_mode("none", default="none"),
        max_train_samples_per_client=None,
        max_test_samples_per_client=None,
        seed=seed,
    )
    if not clients:
        raise RuntimeError(f"no valid LEAF Synthetic clients found under {root}")
    manifest = {
        "source": str(root),
        "license": "LEAF Synthetic",
        "parser": "leaf_synthetic_legacy_v1",
        "partition": "leaf_user",
        "clients": len(clients),
        "features": next(iter(clients.values())).train_x.shape[1],
        "classes": int(max(max(client.train_y.max().item(), client.test_y.max().item()) for client in clients.values())) + 1,
        "limit_clients": limit_clients,
        "min_client_samples": 10,
        "normalization": "none",
        "seed": seed,
        "provenance": "LEGACY_26CASES_SYNTHETIC",
    }
    return write_cache(FederatedDataset("synthetic", clients, manifest["features"], manifest["classes"], manifest), data_root)
