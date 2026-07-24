"""Legacy NinaPro DB1 subject loader used by the 26CASES HoRU code path."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .federated import FederatedDataset, write_cache
from .legacy_contract import build_grouped_clients, compact_grouped_labels, resolve_normalization_mode


def prepare_data(data_root: str | Path, source_root: str | Path, seed: int = 42) -> FederatedDataset:
    root = Path(source_root)
    exercise_offsets = {1: 0, 2: 12, 3: 29}
    grouped = []
    client_ids = []
    subject_names = sorted({path.name.split("_")[0] for path in root.glob("*.mat")})
    for subject in subject_names:
        x_parts = []
        y_parts = []
        for exercise in (1, 2, 3):
            path = root / f"{subject}_A1_E{exercise}.mat"
            if not path.exists():
                continue
            mat = loadmat(path)
            emg = np.asarray(mat["emg"], dtype=np.float32)
            glove = np.asarray(mat["glove"], dtype=np.float32)
            labels = np.asarray(mat["restimulus"], dtype=np.int64).reshape(-1)
            keep = labels != 0
            if not np.any(keep):
                continue
            x_parts.append(np.concatenate([emg[keep], glove[keep]], axis=1))
            y_parts.append((labels[keep] + exercise_offsets[exercise] - 1).astype(np.int64))
        if x_parts:
            grouped.append((np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)))
            client_ids.append(subject)
    clients = build_grouped_clients(
        compact_grouped_labels(grouped),
        client_ids,
        test_size=0.3,
        min_client_samples=20,
        limit_clients=27,
        normalization=resolve_normalization_mode("standardize", default="standardize"),
        max_train_samples_per_client=5000,
        max_test_samples_per_client=1000,
        seed=seed,
    )
    if not clients:
        raise RuntimeError(f"no valid NinaPro DB1 clients found under {root}")
    manifest = {
        "source": str(root),
        "license": "NinaPro DB1",
        "parser": "ninapro_db1_legacy_v1",
        "partition": "subject",
        "clients": len(clients),
        "features": 32,
        "classes": 52,
        "test_size": 0.3,
        "min_client_samples": 20,
        "limit_clients": 27,
        "max_train_samples_per_client": 5000,
        "max_test_samples_per_client": 1000,
        "normalization": "standardize",
        "seed": seed,
        "provenance": "LEGACY_26CASES_NINAPRO",
    }
    return write_cache(FederatedDataset("ninapro", clients, 32, 52, manifest), data_root)
