"""Legacy WISDM raw phone-accelerometer loader used by the 26CASES HoRU code path."""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

from .federated import FederatedDataset, write_cache
from .legacy_contract import build_grouped_clients, compact_grouped_labels, resolve_normalization_mode


def _archive_path(root: str | Path) -> Path:
    root_path = Path(root)
    if root_path.is_file():
        return root_path
    archive = root_path / "wisdm-dataset.zip"
    if archive.exists():
        return archive
    fallback = root_path / "wisdm_dataset.zip"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"WISDM archive not found under {root_path}")


def _iter_user_series(archive: Path, modality: str = "phone_accel") -> list[tuple[str, np.ndarray, list[str]]]:
    device_name, sensor_name = modality.split("_", 1)
    prefix = f"wisdm-dataset/raw/{device_name}/{sensor_name}/"
    rows: list[tuple[str, np.ndarray, list[str]]] = []
    with zipfile.ZipFile(archive) as bundle:
        names = sorted(name for name in bundle.namelist() if name.startswith(prefix) and name.endswith(".txt"))
        for name in names:
            user_id = Path(name).name.split("_")[1]
            user_rows = []
            user_labels = []
            with bundle.open(name) as handle:
                for raw_line in handle:
                    line = raw_line.decode("utf-8", "ignore").strip().rstrip(";")
                    if not line:
                        continue
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) < 6:
                        continue
                    try:
                        xyz = [float(parts[3]), float(parts[4]), float(parts[5])]
                    except ValueError:
                        continue
                    user_rows.append(xyz)
                    user_labels.append(parts[1])
            if user_rows:
                rows.append((user_id, np.asarray(user_rows, dtype=np.float32), user_labels))
    return rows


def prepare_data(data_root: str | Path, archive: str | Path, seed: int = 42, client_ids: list[int] | None = None, recover_missing_from_raw: bool = False) -> FederatedDataset:
    del client_ids, recover_missing_from_raw
    archive_path = _archive_path(archive)
    label_map: dict[str, int] = {}
    grouped = []
    ids = []
    for user_id, x_user, labels in _iter_user_series(archive_path):
        y_user = []
        for label in labels:
            if label not in label_map:
                label_map[label] = len(label_map)
            y_user.append(label_map[label])
        grouped.append((x_user, np.asarray(y_user, dtype=np.int64)))
        ids.append(f"user_{user_id}")
    clients = build_grouped_clients(
        compact_grouped_labels(grouped),
        ids,
        test_size=0.3,
        min_client_samples=20,
        limit_clients=51,
        normalization=resolve_normalization_mode("standardize", default="standardize"),
        max_train_samples_per_client=5000,
        max_test_samples_per_client=1000,
        seed=seed,
    )
    if not clients:
        raise RuntimeError(f"no valid WISDM clients found under {archive_path}")
    manifest = {
        "source": str(archive_path),
        "license": "WISDM",
        "parser": "wisdm_raw_xyz_legacy_v1",
        "partition": "user",
        "clients": len(clients),
        "features": 3,
        "classes": len(label_map),
        "wisdm_modality": "phone_accel",
        "test_size": 0.3,
        "min_client_samples": 20,
        "limit_clients": 51,
        "max_train_samples_per_client": 5000,
        "max_test_samples_per_client": 1000,
        "normalization": "standardize",
        "seed": seed,
        "provenance": "LEGACY_26CASES_WISDM",
    }
    return write_cache(FederatedDataset("wisdm", clients, 3, len(label_map), manifest), data_root)
