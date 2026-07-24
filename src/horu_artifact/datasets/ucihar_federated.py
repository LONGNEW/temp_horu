"""Legacy UCI-HAR subject federation used by the 26CASES HoRU code path."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .federated import FederatedDataset, write_cache
from .legacy_contract import build_explicit_clients, build_grouped_clients, compact_explicit_split_labels, compact_grouped_labels, resolve_normalization_mode
from .ucihar import load_cache


def _load_matrix(root: Path, *parts: str) -> np.ndarray:
    return np.loadtxt(root.joinpath(*parts), dtype=np.float32)


def prepare_data(data_root: str | Path, seed: int = 42, source_root: str | Path | None = None, preserve_original_split: bool = True) -> FederatedDataset:
    normalization = resolve_normalization_mode("l2", default="l2")
    if source_root is None:
        raw = load_cache(data_root)
        features = raw.features.numpy()
        labels = raw.labels.numpy()
        subjects = raw.subjects.numpy()
        grouped = compact_grouped_labels([
            (features[subjects == subject], labels[subjects == subject])
            for subject in sorted(set(subjects.tolist()))
        ])
        client_ids = [f"subject_{int(subject)}" for subject in sorted(set(subjects.tolist()))]
        clients = build_grouped_clients(
            grouped,
            client_ids,
            test_size=0.3,
            min_client_samples=10,
            limit_clients=30,
            normalization=normalization,
            max_train_samples_per_client=None,
            max_test_samples_per_client=None,
            seed=seed,
        )
        manifest_source = "downloaded_cache"
    else:
        root = Path(source_root)
        x_train = _load_matrix(root, "train", "X_train.txt")
        y_train = _load_matrix(root, "train", "y_train.txt").astype(np.int64) - 1
        s_train = _load_matrix(root, "train", "subject_train.txt").astype(np.int64)
        x_test = _load_matrix(root, "test", "X_test.txt")
        y_test = _load_matrix(root, "test", "y_test.txt").astype(np.int64) - 1
        s_test = _load_matrix(root, "test", "subject_test.txt").astype(np.int64)
        manifest_source = str(root)
        explicit_grouped = []
        explicit_ids = []
        merged_grouped = []
        merged_ids = []
        for subject in sorted(set(s_train.tolist()) | set(s_test.tolist())):
            train_mask = s_train == subject
            test_mask = s_test == subject
            client_id = f"subject_{int(subject)}"
            if preserve_original_split and np.any(train_mask) and np.any(test_mask):
                explicit_grouped.append((x_train[train_mask], y_train[train_mask], x_test[test_mask], y_test[test_mask]))
                explicit_ids.append(client_id)
            else:
                parts_x = []
                parts_y = []
                if np.any(train_mask):
                    parts_x.append(x_train[train_mask])
                    parts_y.append(y_train[train_mask])
                if np.any(test_mask):
                    parts_x.append(x_test[test_mask])
                    parts_y.append(y_test[test_mask])
                merged_grouped.append((np.concatenate(parts_x, axis=0), np.concatenate(parts_y, axis=0)))
                merged_ids.append(client_id)
        clients = build_explicit_clients(
            compact_explicit_split_labels(explicit_grouped),
            explicit_ids,
            min_client_samples=10,
            limit_clients=None,
            normalization=normalization,
            max_train_samples_per_client=None,
            max_test_samples_per_client=None,
            seed=seed,
        )
        clients.update(build_grouped_clients(
            compact_grouped_labels(merged_grouped),
            merged_ids,
            test_size=0.3,
            min_client_samples=10,
            limit_clients=None,
            normalization=normalization,
            max_train_samples_per_client=None,
            max_test_samples_per_client=None,
            seed=seed,
        ))
        clients = dict(list(sorted(clients.items()))[:30])
    manifest = {
        "source": manifest_source,
        "license": "UCI-HAR",
        "parser": "ucihar_legacy_v1",
        "clients": len(clients),
        "features": 561,
        "classes": 6,
        "partition": "subject",
        "test_size": 0.3,
        "min_client_samples": 10,
        "limit_clients": 30,
        "normalization": "l2",
        "preserve_original_split": preserve_original_split,
        "seed": seed,
        "provenance": "LEGACY_26CASES_UCIHAR",
    }
    return write_cache(FederatedDataset("ucihar", clients, 561, 6, manifest), data_root)
