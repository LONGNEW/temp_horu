from __future__ import annotations

import json
import pickle
import sys
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat


@dataclass(slots=True)
class ClientData:
    client_id: str
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor


class ClientDatasetAdapter(ABC):
    """Normalizes dataset-specific loading into a per-client view."""

    @abstractmethod
    def load_clients(self) -> list[ClientData]:
        raise NotImplementedError

    @abstractmethod
    def num_classes(self) -> int:
        raise NotImplementedError


def l2_normalize_rows_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms < eps] = 1.0
    return x / norms


def l2_normalize_rows_torch(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    norms = torch.linalg.norm(x, dim=1, keepdim=True)
    norms = torch.where(norms < eps, torch.ones_like(norms), norms)
    return x / norms


def standardize_train_test_np(x_train: np.ndarray, x_test: np.ndarray, std_floor: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.maximum(std, std_floor)
    return (x_train - mean) / std, (x_test - mean) / std


def _resolve_normalization_mode(mode: str | None, *, default: str) -> str:
    resolved = default if mode is None else str(mode).lower()
    if resolved not in {"l2", "standardize", "none"}:
        raise ValueError(f"Unsupported normalization mode: {mode}")
    return resolved


def _apply_normalization(
    x_train: np.ndarray,
    x_test: np.ndarray,
    *,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "none":
        return x_train, x_test
    if mode == "l2":
        return l2_normalize_rows_np(x_train), l2_normalize_rows_np(x_test)
    return standardize_train_test_np(x_train, x_test)


def _capped_subset(
    x: np.ndarray,
    y: np.ndarray,
    max_samples: int | None,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None:
        return x, y
    max_samples = int(max_samples)
    if max_samples <= 0 or len(y) <= max_samples:
        return x, y

    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) == 0:
        return x, y

    if max_samples >= len(classes):
        proportions = counts.astype(np.float64) / float(counts.sum())
        allocation = np.floor(proportions * max_samples).astype(int)
        allocation = np.maximum(allocation, 1)
        allocation = np.minimum(allocation, counts)

        while allocation.sum() > max_samples:
            reducible = np.where(allocation > 1)[0]
            if reducible.size == 0:
                break
            target = reducible[np.argmax(allocation[reducible])]
            allocation[target] -= 1

        while allocation.sum() < max_samples:
            remaining = counts - allocation
            growable = np.where(remaining > 0)[0]
            if growable.size == 0:
                break
            target = growable[np.argmax(remaining[growable])]
            allocation[target] += 1

        picked_indices: list[np.ndarray] = []
        for cls, take in zip(classes, allocation):
            cls_idx = np.where(y == cls)[0]
            rng.shuffle(cls_idx)
            picked_indices.append(cls_idx[:take])
        chosen = np.sort(np.concatenate(picked_indices, axis=0))
        return x[chosen], y[chosen]

    chosen = np.sort(rng.choice(len(y), size=max_samples, replace=False))
    return x[chosen], y[chosen]


def _split_train_test(
    x: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(y) < 2:
        raise ValueError("Need at least two samples to split client data.")

    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    can_stratify = len(classes) >= 2 and counts.min() >= 2 and len(y) >= max(6, len(classes) * 2)

    if can_stratify:
        train_parts: list[np.ndarray] = []
        test_parts: list[np.ndarray] = []
        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            rng.shuffle(cls_idx)
            num_test = int(round(len(cls_idx) * float(test_size)))
            num_test = max(1, min(len(cls_idx) - 1, num_test))
            test_parts.append(cls_idx[:num_test])
            train_parts.append(cls_idx[num_test:])
        train_idx = np.concatenate(train_parts, axis=0)
        test_idx = np.concatenate(test_parts, axis=0)
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
    else:
        perm = rng.permutation(len(y))
        split_idx = max(1, min(len(y) - 1, int(round((1.0 - float(test_size)) * len(y)))))
        train_idx = perm[:split_idx]
        test_idx = perm[split_idx:]

    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def _compact_grouped_labels(grouped: list[tuple[np.ndarray, np.ndarray]]) -> list[tuple[np.ndarray, np.ndarray]]:
    if not grouped:
        return []
    all_y = np.concatenate([y for _, y in grouped], axis=0)
    classes = np.unique(all_y)
    mapping = {int(cls): idx for idx, cls in enumerate(classes.tolist())}
    compacted: list[tuple[np.ndarray, np.ndarray]] = []
    for x_group, y_group in grouped:
        compacted.append(
            (
                x_group,
                np.asarray([mapping[int(label)] for label in y_group], dtype=np.int64),
            )
        )
    return compacted


def _compact_explicit_split_labels(
    grouped: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if not grouped:
        return []
    all_y = np.concatenate(
        [np.concatenate([y_train, y_test], axis=0) for _, y_train, _, y_test in grouped],
        axis=0,
    )
    classes = np.unique(all_y)
    mapping = {int(cls): idx for idx, cls in enumerate(classes.tolist())}
    compacted: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for x_train, y_train, x_test, y_test in grouped:
        compacted.append(
            (
                x_train,
                np.asarray([mapping[int(label)] for label in y_train], dtype=np.int64),
                x_test,
                np.asarray([mapping[int(label)] for label in y_test], dtype=np.int64),
            )
        )
    return compacted


def _replace_nan_with_column_means(x: np.ndarray) -> np.ndarray:
    if not np.isnan(x).any():
        return x
    x = x.copy()
    col_means = np.nanmean(x, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_rows, nan_cols = np.where(np.isnan(x))
    x[nan_rows, nan_cols] = col_means[nan_cols]
    return x


def _window_flattened(
    x: np.ndarray,
    y: np.ndarray,
    *,
    window_size: int,
    stride: int,
    strict_label: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {x.shape}")
    if y.ndim != 1:
        raise ValueError(f"Expected 1D labels, got shape {y.shape}")
    if len(x) != len(y):
        raise ValueError(f"Feature/label length mismatch: {len(x)} vs {len(y)}")
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    if len(y) < window_size:
        return np.empty((0, x.shape[1] * window_size), dtype=np.float32), np.empty((0,), dtype=np.int64)

    x_windows: list[np.ndarray] = []
    y_windows: list[int] = []
    for start in range(0, len(y) - window_size + 1, stride):
        stop = start + window_size
        label_slice = y[start:stop]
        if strict_label and int(np.unique(label_slice).size) != 1:
            continue
        label = int(np.bincount(label_slice.astype(np.int64)).argmax())
        x_windows.append(x[start:stop].reshape(-1))
        y_windows.append(label)
    if not x_windows:
        return np.empty((0, x.shape[1] * window_size), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(x_windows, axis=0).astype(np.float32), np.asarray(y_windows, dtype=np.int64)


def _chronological_split(
    x: np.ndarray,
    y: np.ndarray,
    *,
    train_fraction: float,
    val_fraction: float,
    include_val_in_train: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(y) == 0:
        return (
            np.empty((0, x.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0, x.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    train_fraction = float(train_fraction)
    val_fraction = float(val_fraction)
    if train_fraction <= 0.0 or train_fraction >= 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    if val_fraction < 0.0 or (train_fraction + val_fraction) >= 1.0:
        raise ValueError(f"Invalid train/val fractions: {train_fraction}, {val_fraction}")

    n = len(y)
    train_end = int(round(n * train_fraction))
    val_end = int(round(n * (train_fraction + val_fraction)))
    train_end = max(1, min(n - 1, train_end))
    val_end = max(train_end, min(n - 1, val_end))
    test_start = val_end

    if include_val_in_train:
        x_train = x[:val_end]
        y_train = y[:val_end]
    else:
        x_train = x[:train_end]
        y_train = y[:train_end]
    x_test = x[test_start:]
    y_test = y[test_start:]

    if len(y_test) == 0 and n >= 2:
        x_train = x[:-1]
        y_train = y[:-1]
        x_test = x[-1:]
        y_test = y[-1:]
    return x_train, y_train, x_test, y_test


def _per_label_chronological_split(
    x: np.ndarray,
    y: np.ndarray,
    *,
    train_fraction: float,
    val_fraction: float,
    include_val_in_train: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(y) == 0:
        return _chronological_split(
            x,
            y,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            include_val_in_train=include_val_in_train,
        )

    train_parts: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    val_labels: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    test_labels: list[np.ndarray] = []

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        if cls_idx.size == 0:
            continue
        cls_x = x[cls_idx]
        cls_y = y[cls_idx]
        x_train_c, y_train_c, x_test_c, y_test_c = _chronological_split(
            cls_x,
            cls_y,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            include_val_in_train=False,
        )

        n_cls = cls_idx.size
        train_end = int(round(n_cls * float(train_fraction)))
        val_end = int(round(n_cls * (float(train_fraction) + float(val_fraction))))
        train_end = max(1, min(n_cls - 1, train_end))
        val_end = max(train_end, min(n_cls - 1, val_end))
        val_count = max(0, val_end - train_end)

        train_parts.append(x_train_c)
        train_labels.append(y_train_c)
        if val_count > 0:
            val_parts.append(cls_x[train_end:val_end])
            val_labels.append(cls_y[train_end:val_end])
        if len(y_test_c) > 0:
            test_parts.append(x_test_c)
            test_labels.append(y_test_c)

    if not train_parts or not test_parts:
        return _chronological_split(
            x,
            y,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            include_val_in_train=include_val_in_train,
        )

    x_train = np.concatenate(train_parts, axis=0)
    y_train = np.concatenate(train_labels, axis=0)
    if include_val_in_train and val_parts:
        x_train = np.concatenate([x_train, *val_parts], axis=0)
        y_train = np.concatenate([y_train, *val_labels], axis=0)
    x_test = np.concatenate(test_parts, axis=0)
    y_test = np.concatenate(test_labels, axis=0)

    return x_train, y_train, x_test, y_test


def _finalize_client_data(
    *,
    client_id: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    normalization: str,
    max_train_samples_per_client: int | None,
    max_test_samples_per_client: int | None,
    seed: int,
) -> ClientData | None:
    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    x_test = np.asarray(x_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int64)

    x_train, y_train = _capped_subset(x_train, y_train, max_train_samples_per_client, seed=seed)
    x_test, y_test = _capped_subset(x_test, y_test, max_test_samples_per_client, seed=seed + 1000003)
    if len(y_train) == 0 or len(y_test) == 0:
        return None

    x_train, x_test = _apply_normalization(x_train, x_test, mode=normalization)
    return ClientData(
        client_id=client_id,
        x_train=torch.tensor(x_train, dtype=torch.float32, device=device),
        y_train=torch.tensor(y_train, dtype=torch.long, device=device),
        x_test=torch.tensor(x_test, dtype=torch.float32, device=device),
        y_test=torch.tensor(y_test, dtype=torch.long, device=device),
    )


def _dirichlet_client_indices(y: np.ndarray, *, num_clients: int, alpha: float, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    num_classes = int(y.max()) + 1
    class_indices = [np.where(y == cls)[0] for cls in range(num_classes)]
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for indices in class_indices:
        if len(indices) == 0:
            continue
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        proportions = rng.dirichlet(np.full(num_clients, float(alpha), dtype=np.float64))
        cuts = (np.cumsum(proportions) * len(shuffled)).astype(int)[:-1]
        splits = np.split(shuffled, cuts)
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    return [np.asarray(sorted(indices), dtype=np.int64) for indices in client_indices]


def _build_dirichlet_clients(
    *,
    x: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    alpha: float,
    test_size: float,
    min_client_samples: int,
    limit_clients: int | None,
    normalization: str,
    max_train_samples_per_client: int | None,
    max_test_samples_per_client: int | None,
    seed: int,
    device: torch.device,
) -> list[ClientData]:
    clients: list[ClientData] = []
    for client_id, indices in enumerate(_dirichlet_client_indices(y, num_clients=num_clients, alpha=alpha, seed=seed)):
        if len(indices) < min_client_samples:
            continue
        x_client = x[indices]
        y_client = y[indices]
        if len(np.unique(y_client)) < 2:
            continue
        x_train, y_train, x_test, y_test = _split_train_test(
            x_client,
            y_client,
            test_size=test_size,
            seed=seed + client_id,
        )
        client = _finalize_client_data(
            client_id=f"client_{client_id}",
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            device=device,
            normalization=normalization,
            max_train_samples_per_client=max_train_samples_per_client,
            max_test_samples_per_client=max_test_samples_per_client,
            seed=seed + client_id,
        )
        if client is not None:
            clients.append(client)

    if limit_clients is not None and limit_clients < len(clients):
        # Select subjects from the complete valid pool.  A prefix here would
        # make the selected federation depend on source-shard ordering rather
        # than the configured experimental seed.
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(clients), size=limit_clients, replace=False))
        clients = [clients[int(index)] for index in selected]
    return clients


def _build_dirichlet_explicit_split_clients(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_clients: int,
    alpha: float,
    min_client_samples: int,
    limit_clients: int | None,
    normalization: str,
    max_train_samples_per_client: int | None,
    max_test_samples_per_client: int | None,
    seed: int,
    device: torch.device,
) -> list[ClientData]:
    rng = np.random.default_rng(seed)
    num_classes = int(max(y_train.max(initial=0), y_test.max(initial=0))) + 1
    train_client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    test_client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for cls in range(num_classes):
        class_train_idx = np.where(y_train == cls)[0]
        class_test_idx = np.where(y_test == cls)[0]
        if class_train_idx.size == 0 and class_test_idx.size == 0:
            continue

        proportions = rng.dirichlet(np.full(num_clients, float(alpha), dtype=np.float64))

        if class_train_idx.size > 0:
            shuffled_train = class_train_idx.copy()
            rng.shuffle(shuffled_train)
            train_cuts = (np.cumsum(proportions) * len(shuffled_train)).astype(int)[:-1]
            for client_id, split in enumerate(np.split(shuffled_train, train_cuts)):
                train_client_indices[client_id].extend(split.tolist())

        if class_test_idx.size > 0:
            shuffled_test = class_test_idx.copy()
            rng.shuffle(shuffled_test)
            test_cuts = (np.cumsum(proportions) * len(shuffled_test)).astype(int)[:-1]
            for client_id, split in enumerate(np.split(shuffled_test, test_cuts)):
                test_client_indices[client_id].extend(split.tolist())

    clients: list[ClientData] = []
    for client_id in range(num_clients):
        train_idx = np.asarray(sorted(train_client_indices[client_id]), dtype=np.int64)
        test_idx = np.asarray(sorted(test_client_indices[client_id]), dtype=np.int64)
        if train_idx.size < min_client_samples or test_idx.size == 0:
            continue

        y_train_client = y_train[train_idx]
        if len(np.unique(y_train_client)) < 2:
            continue

        client = _finalize_client_data(
            client_id=f"client_{client_id}",
            x_train=x_train[train_idx],
            y_train=y_train_client,
            x_test=x_test[test_idx],
            y_test=y_test[test_idx],
            device=device,
            normalization=normalization,
            max_train_samples_per_client=max_train_samples_per_client,
            max_test_samples_per_client=max_test_samples_per_client,
            seed=seed + client_id,
        )
        if client is not None:
            clients.append(client)

    if limit_clients is not None and limit_clients < len(clients):
        # Select subjects from the complete valid pool.  A prefix here would
        # make the selected federation depend on source-shard ordering rather
        # than the configured experimental seed.
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(clients), size=limit_clients, replace=False))
        clients = [clients[int(index)] for index in selected]
    return clients


def _build_grouped_clients(
    *,
    grouped: list[tuple[np.ndarray, np.ndarray]],
    client_ids: list[str],
    test_size: float,
    min_client_samples: int,
    limit_clients: int | None,
    normalization: str,
    max_train_samples_per_client: int | None,
    max_test_samples_per_client: int | None,
    seed: int,
    device: torch.device,
) -> list[ClientData]:
    clients: list[ClientData] = []
    for idx, (client_id, (x_group, y_group)) in enumerate(zip(client_ids, grouped)):
        if len(y_group) < min_client_samples or len(np.unique(y_group)) < 2:
            continue
        x_train, y_train, x_test, y_test = _split_train_test(
            np.asarray(x_group, dtype=np.float32),
            np.asarray(y_group, dtype=np.int64),
            test_size=test_size,
            seed=seed + idx,
        )
        client = _finalize_client_data(
            client_id=client_id,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            device=device,
            normalization=normalization,
            max_train_samples_per_client=max_train_samples_per_client,
            max_test_samples_per_client=max_test_samples_per_client,
            seed=seed + idx,
        )
        if client is not None:
            clients.append(client)

    if limit_clients is not None and limit_clients < len(clients):
        # Grouped clients are subjects for PAMAP2 and similar sensor datasets.
        # Draw from the full valid set under the configured experimental seed.
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(clients), size=limit_clients, replace=False))
        clients = [clients[int(index)] for index in selected]
    return clients


def _build_explicit_clients(
    *,
    grouped: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    client_ids: list[str],
    min_client_samples: int,
    limit_clients: int | None,
    require_multi_class: bool = True,
    normalization: str,
    max_train_samples_per_client: int | None,
    max_test_samples_per_client: int | None,
    seed: int,
    device: torch.device,
) -> list[ClientData]:
    clients: list[ClientData] = []
    for idx, (client_id, (x_train, y_train, x_test, y_test)) in enumerate(zip(client_ids, grouped)):
        if len(y_train) < min_client_samples or len(y_test) == 0:
            continue
        if require_multi_class and len(np.unique(y_train)) < 2:
            continue
        client = _finalize_client_data(
            client_id=client_id,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            device=device,
            normalization=normalization,
            max_train_samples_per_client=max_train_samples_per_client,
            max_test_samples_per_client=max_test_samples_per_client,
            seed=seed + idx,
        )
        if client is not None:
            clients.append(client)

    if limit_clients is not None and limit_clients < len(clients):
        # Explicit client ids are subjects/writers.  Seeded selection prevents
        # a source-file prefix from becoming an implicit experimental setting.
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(clients), size=limit_clients, replace=False))
        clients = [clients[int(index)] for index in selected]
    return clients


class ISOLETAdapter(ClientDatasetAdapter):
    """Loads ISOLET and partitions it into synthetic federated clients with Dirichlet splits."""

    def __init__(
        self,
        root: str | Path = "../fisher_FL_hd_unzipped/data/raw/isolet",
        *,
        num_clients: int = 8,
        alpha: float = 5.0,
        test_size: float = 0.3,
        min_client_samples: int = 10,
        limit_clients: int | None = None,
        normalization: str | None = None,
        max_train_samples_per_client: int | None = None,
        max_test_samples_per_client: int | None = None,
        preserve_original_split: bool = False,
        seed: int = 13,
        device: torch.device | str = "cpu",
    ) -> None:
        self.root = Path(root)
        self.num_clients = num_clients
        self.alpha = alpha
        self.test_size = test_size
        self.min_client_samples = min_client_samples
        self.limit_clients = limit_clients
        self.normalization = _resolve_normalization_mode(normalization, default="l2")
        self.max_train_samples_per_client = max_train_samples_per_client
        self.max_test_samples_per_client = max_test_samples_per_client
        self.preserve_original_split = preserve_original_split
        self.seed = seed
        self.device = torch.device(device)
        self._num_classes: int | None = None

    def _parse_isolet_file(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        rows: list[list[float]] = []
        labels: list[int] = []
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                parts = [part.strip() for part in line.strip().split(",")]
                if len(parts) < 2:
                    continue
                rows.append([float(value) for value in parts[:-1]])
                labels.append(int(float(parts[-1])) - 1)
        return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int64)

    def load_clients(self) -> list[ClientData]:
        x_train_raw, y_train_raw = self._parse_isolet_file(self.root / "isolet1+2+3+4.data")
        x_test_raw, y_test_raw = self._parse_isolet_file(self.root / "isolet5.data")
        if self.preserve_original_split:
            clients = _build_dirichlet_explicit_split_clients(
                x_train=x_train_raw,
                y_train=y_train_raw,
                x_test=x_test_raw,
                y_test=y_test_raw,
                num_clients=self.num_clients,
                alpha=self.alpha,
                min_client_samples=self.min_client_samples,
                limit_clients=self.limit_clients,
                normalization=self.normalization,
                max_train_samples_per_client=self.max_train_samples_per_client,
                max_test_samples_per_client=self.max_test_samples_per_client,
                seed=self.seed,
                device=self.device,
            )
        else:
            x = np.concatenate([x_train_raw, x_test_raw], axis=0)
            y = np.concatenate([y_train_raw, y_test_raw], axis=0)
            clients = _build_dirichlet_clients(
                x=x,
                y=y,
                num_clients=self.num_clients,
                alpha=self.alpha,
                test_size=self.test_size,
                min_client_samples=self.min_client_samples,
                limit_clients=self.limit_clients,
                normalization=self.normalization,
                max_train_samples_per_client=self.max_train_samples_per_client,
                max_test_samples_per_client=self.max_test_samples_per_client,
                seed=self.seed,
                device=self.device,
            )
        if not clients:
            raise RuntimeError(f"No valid ISOLET clients found under {self.root}")
        self._num_classes = int(max(y_train_raw.max(initial=0), y_test_raw.max(initial=0))) + 1
        return clients

    def num_classes(self) -> int:
        if self._num_classes is None:
            raise RuntimeError("Call load_clients() before requesting num_classes().")
        return self._num_classes


class CIFAR10Adapter(ClientDatasetAdapter):
    """Loads CIFAR-10 pickled batches and partitions them into Dirichlet clients."""

    def __init__(
        self,
        root: str | Path = "../fisher_FL_hd_unzipped/data/tiers/standard_pfl/cifar10/cifar-10-batches-py",
        *,
        num_clients: int = 37,
        alpha: float = 0.5,
        test_size: float = 0.3,
        min_client_samples: int = 10,
        limit_clients: int | None = None,
        normalization: str | None = None,
        max_train_samples_per_client: int | None = None,
        max_test_samples_per_client: int | None = None,
        seed: int = 13,
        device: torch.device | str = "cpu",
    ) -> None:
        self.root = Path(root)
        self.num_clients = num_clients
        self.alpha = alpha
        self.test_size = test_size
        self.min_client_samples = min_client_samples
        self.limit_clients = limit_clients
        self.normalization = _resolve_normalization_mode(normalization, default="none")
        self.max_train_samples_per_client = max_train_samples_per_client
        self.max_test_samples_per_client = max_test_samples_per_client
        self.seed = seed
        self.device = torch.device(device)
        self._num_classes: int | None = None

    @staticmethod
    def _load_pickle(path: Path) -> dict:
        with path.open("rb") as handle:
            return pickle.load(handle, encoding="bytes")

    def load_clients(self) -> list[ClientData]:
        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        for batch_id in range(1, 6):
            batch = self._load_pickle(self.root / f"data_batch_{batch_id}")
            x_parts.append(np.asarray(batch[b"data"], dtype=np.float32))
            y_parts.append(np.asarray(batch[b"labels"], dtype=np.int64))
        test_batch = self._load_pickle(self.root / "test_batch")
        x_parts.append(np.asarray(test_batch[b"data"], dtype=np.float32))
        y_parts.append(np.asarray(test_batch[b"labels"], dtype=np.int64))

        x = np.concatenate(x_parts, axis=0) / 255.0
        y = np.concatenate(y_parts, axis=0)
        clients = _build_dirichlet_clients(
            x=x,
            y=y,
            num_clients=self.num_clients,
            alpha=self.alpha,
            test_size=self.test_size,
            min_client_samples=self.min_client_samples,
            limit_clients=self.limit_clients,
            normalization=self.normalization,
            max_train_samples_per_client=self.max_train_samples_per_client,
            max_test_samples_per_client=self.max_test_samples_per_client,
            seed=self.seed,
            device=self.device,
        )
        if not clients:
            raise RuntimeError(f"No valid CIFAR-10 clients found under {self.root}")
        self._num_classes = 10
        return clients

    def num_classes(self) -> int:
        if self._num_classes is None:
            raise RuntimeError("Call load_clients() before requesting num_classes().")
        return self._num_classes


class CIFAR100Adapter(ClientDatasetAdapter):
    """Loads CIFAR-100 pickled batches with coarse or fine labels."""

    def __init__(
        self,
        root: str | Path = "../fisher_FL_hd_unzipped/data/tiers/standard_pfl/cifar100/cifar-100-python",
        *,
        label_type: str = "coarse",
        num_clients: int = 33,
        alpha: float = 1.0,
        test_size: float = 0.3,
        min_client_samples: int = 10,
        limit_clients: int | None = None,
        normalization: str | None = None,
        max_train_samples_per_client: int | None = None,
        max_test_samples_per_client: int | None = None,
        seed: int = 13,
        device: torch.device | str = "cpu",
    ) -> None:
        self.root = Path(root)
        self.label_type = str(label_type).lower()
        self.num_clients = num_clients
        self.alpha = alpha
        self.test_size = test_size
        self.min_client_samples = min_client_samples
        self.limit_clients = limit_clients
        self.normalization = _resolve_normalization_mode(normalization, default="none")
        self.max_train_samples_per_client = max_train_samples_per_client
        self.max_test_samples_per_client = max_test_samples_per_client
        self.seed = seed
        self.device = torch.device(device)
        self._num_classes: int | None = None

    @staticmethod
    def _load_pickle(path: Path) -> dict:
        with path.open("rb") as handle:
            return pickle.load(handle, encoding="bytes")

    def load_clients(self) -> list[ClientData]:
        label_key = b"coarse_labels" if self.label_type == "coarse" else b"fine_labels"
        train_batch = self._load_pickle(self.root / "train")
        test_batch = self._load_pickle(self.root / "test")
        x = np.concatenate(
            [
                np.asarray(train_batch[b"data"], dtype=np.float32),
                np.asarray(test_batch[b"data"], dtype=np.float32),
            ],
            axis=0,
        ) / 255.0
        y = np.concatenate(
            [
                np.asarray(train_batch[label_key], dtype=np.int64),
                np.asarray(test_batch[label_key], dtype=np.int64),
            ],
            axis=0,
        )
        clients = _build_dirichlet_clients(
            x=x,
            y=y,
            num_clients=self.num_clients,
            alpha=self.alpha,
            test_size=self.test_size,
            min_client_samples=self.min_client_samples,
            limit_clients=self.limit_clients,
            normalization=self.normalization,
            max_train_samples_per_client=self.max_train_samples_per_client,
            max_test_samples_per_client=self.max_test_samples_per_client,
            seed=self.seed,
            device=self.device,
        )
        if not clients:
            raise RuntimeError(f"No valid CIFAR-100 clients found under {self.root}")
        self._num_classes = int(y.max()) + 1
        return clients

    def num_classes(self) -> int:
        if self._num_classes is None:
            raise RuntimeError("Call load_clients() before requesting num_classes().")
        return self._num_classes


class EMNISTAdapter(ClientDatasetAdapter):
    """Loads EMNIST and partitions it into synthetic federated clients with Dirichlet splits."""

    def __init__(
        self,
        root: str | Path = "../fisher_FL_hd_unzipped/data/tiers/standard_pfl/emnist",
        *,
        split: str = "balanced",
        num_clients: int = 100,
        alpha: float = 0.5,
        test_size: float = 0.3,
        min_client_samples: int = 10,
        limit_clients: int | None = None,
        normalization: str | None = None,
        max_train_samples_per_client: int | None = None,
        max_test_samples_per_client: int | None = None,
        seed: int = 13,
        download: bool = True,
        device: torch.device | str = "cpu",
    ) -> None:
        self.root = Path(root)
        self.split = str(split)
        self.num_clients = num_clients
        self.alpha = alpha
        self.test_size = test_size
        self.min_client_samples = min_client_samples
        self.limit_clients = limit_clients
        self.normalization = _resolve_normalization_mode(normalization, default="none")
        self.max_train_samples_per_client = max_train_samples_per_client
        self.max_test_samples_per_client = max_test_samples_per_client
        self.seed = seed
        self.download = bool(download)
        self.device = torch.device(device)
        self._num_classes: int | None = None

    def load_clients(self) -> list[ClientData]:
        try:
            from torchvision.datasets import EMNIST
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError("torchvision is required to load EMNIST.") from exc

        self.root.mkdir(parents=True, exist_ok=True)
        try:
            train_set = EMNIST(root=str(self.root), split=self.split, train=True, download=self.download)
            test_set = EMNIST(root=str(self.root), split=self.split, train=False, download=self.download)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load EMNIST. If running offline, pre-download files under "
                f"{self.root} and rerun with download=False."
            ) from exc

        x_train = np.asarray(train_set.data, dtype=np.float32).reshape(len(train_set), -1) / 255.0
        y_train = np.asarray(train_set.targets, dtype=np.int64)
        x_test = np.asarray(test_set.data, dtype=np.float32).reshape(len(test_set), -1) / 255.0
        y_test = np.asarray(test_set.targets, dtype=np.int64)

        x = np.concatenate([x_train, x_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        clients = _build_dirichlet_clients(
            x=x,
            y=y,
            num_clients=self.num_clients,
            alpha=self.alpha,
            test_size=self.test_size,
            min_client_samples=self.min_client_samples,
            limit_clients=self.limit_clients,
            normalization=self.normalization,
            max_train_samples_per_client=self.max_train_samples_per_client,
            max_test_samples_per_client=self.max_test_samples_per_client,
            seed=self.seed,
            device=self.device,
        )
        if not clients:
            raise RuntimeError(f"No valid EMNIST clients found under {self.root}")
        self._num_classes = int(y.max()) + 1
        return clients

    def num_classes(self) -> int:
        if self._num_classes is None:
            raise RuntimeError("Call load_clients() before requesting num_classes().")
        return self._num_classes


class FlambyTcgaBrcaAdapter(ClientDatasetAdapter):
    """Loads FLamby Fed-TCGA-BRCA and converts (event, time) to binary event labels."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        num_clients: int = 100,
        alpha: float = 0.5,
        test_size: float = 0.3,
        min_client_samples: int = 10,
        limit_clients: int | None = 100,
        preserve_native_clients: bool = True,
        normalization: str | None = None,
        max_train_samples_per_client: int | None = None,
        max_test_samples_per_client: int | None = None,
        seed: int = 13,
        auto_accept_license: bool = False,
        device: torch.device | str = "cpu",
    ) -> None:
        self.root = None if root is None else Path(root)
        self.num_clients = num_clients
        self.alpha = alpha
        self.test_size = test_size
        self.min_client_samples = min_client_samples
        self.limit_clients = limit_clients
        self.preserve_native_clients = bool(preserve_native_clients)
        self.normalization = _resolve_normalization_mode(normalization, default="standardize")
        self.max_train_samples_per_client = max_train_samples_per_client
        self.max_test_samples_per_client = max_test_samples_per_client
        self.seed = seed
        self.auto_accept_license = bool(auto_accept_license)
        self.device = torch.device(device)
        self._num_classes: int | None = None

    @staticmethod
    def _resolve_flamby_import():
        try:
            from flamby.datasets.fed_tcga_brca import FedTcgaBrca  # type: ignore

            return FedTcgaBrca
        except ModuleNotFoundError:
            project_root = Path(__file__).resolve().parent.parent
            local_flamby = project_root / "external" / "FLamby"
            if not local_flamby.exists():
                raise
            sys.path.insert(0, str(local_flamby))
            from flamby.datasets.fed_tcga_brca import FedTcgaBrca  # type: ignore

            return FedTcgaBrca

    def _ensure_license(self, fed_cls) -> None:
        dataset_file = Path(sys.modules[fed_cls.__module__].__file__).resolve()
        license_file = dataset_file.parent / "dataset_creation_scripts" / "license_agreement_fed_tcga_brca"
        if license_file.exists():
            return
        if self.auto_accept_license:
            license_file.parent.mkdir(parents=True, exist_ok=True)
            license_file.touch(exist_ok=True)
            return
        raise RuntimeError(
            "FLamby Fed-TCGA-BRCA requires accepting data terms first. "
            "Create the FLamby license marker file after reading terms or set auto_accept_license=True."
        )

    @staticmethod
    def _dataset_to_xy(dataset) -> tuple[np.ndarray, np.ndarray]:
        if len(dataset) == 0:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
        x_parts: list[np.ndarray] = []
        y_parts: list[int] = []
        for idx in range(len(dataset)):
            x_tensor, y_tensor = dataset[idx]
            x_parts.append(np.asarray(x_tensor, dtype=np.float32))
            # y[0] is event indicator (0/1), y[1] is survival time.
            y_parts.append(int(float(y_tensor[0]) > 0.5))
        return np.stack(x_parts, axis=0), np.asarray(y_parts, dtype=np.int64)

    def load_clients(self) -> list[ClientData]:
        fed_cls = self._resolve_flamby_import()
        self._ensure_license(fed_cls)

        if self.preserve_native_clients:
            explicit_grouped: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
            explicit_ids: list[str] = []
            for center in range(6):
                train_center = fed_cls(center=center, train=True, pooled=False)
                test_center = fed_cls(center=center, train=False, pooled=False)
                x_train, y_train = self._dataset_to_xy(train_center)
                x_test, y_test = self._dataset_to_xy(test_center)
                explicit_grouped.append((x_train, y_train, x_test, y_test))
                explicit_ids.append(f"center_{center}")

            clients = _build_explicit_clients(
                grouped=_compact_explicit_split_labels(explicit_grouped),
                client_ids=explicit_ids,
                min_client_samples=self.min_client_samples,
                limit_clients=self.limit_clients,
                require_multi_class=False,
                normalization=self.normalization,
                max_train_samples_per_client=self.max_train_samples_per_client,
                max_test_samples_per_client=self.max_test_samples_per_client,
                seed=self.seed,
                device=self.device,
            )
        else:
            train_pooled = fed_cls(train=True, pooled=True)
            test_pooled = fed_cls(train=False, pooled=True)
            x_train, y_train = self._dataset_to_xy(train_pooled)
            x_test, y_test = self._dataset_to_xy(test_pooled)

            x = np.concatenate([x_train, x_test], axis=0)
            y = np.concatenate([y_train, y_test], axis=0)
            clients = _build_dirichlet_clients(
                x=x,
                y=y,
                num_clients=self.num_clients,
                alpha=self.alpha,
                test_size=self.test_size,
                min_client_samples=self.min_client_samples,
                limit_clients=self.limit_clients,
                normalization=self.normalization,
                max_train_samples_per_client=self.max_train_samples_per_client,
                max_test_samples_per_client=self.max_test_samples_per_client,
                seed=self.seed,
                device=self.device,
            )
        if not clients:
            raise RuntimeError("No valid FLamby TCGA-BRCA clients were created.")
        self._num_classes = 2
        return clients

    def num_classes(self) -> int:
        if self._num_classes is None:
            raise RuntimeError("Call load_clients() before requesting num_classes().")
        return self._num_classes


class FlambyHeartDiseaseAdapter(ClientDatasetAdapter):
    """Loads FLamby Fed-Heart-Disease and maps labels to binary disease targets."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        num_clients: int = 100,
        alpha: float = 0.5,
        test_size: float = 0.3,
        min_client_samples: int = 10,
        limit_clients: int | None = 100,
        preserve_native_clients: bool = True,
        normalization: str | None = None,
        max_train_samples_per_client: int | None = None,
        max_test_samples_per_client: int | None = None,
        seed: int = 13,
        auto_accept_license: bool = False,
        device: torch.device | str = "cpu",
    ) -> None:
        self.root = None if root is None else Path(root)
        self.num_clients = num_clients
        self.alpha = alpha
        self.test_size = test_size
        self.min_client_samples = min_client_samples
        self.limit_clients = limit_clients
        self.preserve_native_clients = bool(preserve_native_clients)
        self.normalization = _resolve_normalization_mode(normalization, default="standardize")
        self.max_train_samples_per_client = max_train_samples_per_client
        self.max_test_samples_per_client = max_test_samples_per_client
        self.seed = seed
        self.auto_accept_license = bool(auto_accept_license)
        self.device = torch.device(device)
        self._num_classes: int | None = None

    @staticmethod
    def _resolve_flamby_import():
        try:
            from flamby.datasets.fed_heart_disease import FedHeartDisease  # type: ignore

            return FedHeartDisease
        except ModuleNotFoundError:
            project_root = Path(__file__).resolve().parent.parent
            local_flamby = project_root / "external" / "FLamby"
            if not local_flamby.exists():
                raise
            sys.path.insert(0, str(local_flamby))
            from flamby.datasets.fed_heart_disease import FedHeartDisease  # type: ignore

            return FedHeartDisease

    def _ensure_license(self, fed_cls) -> None:
        dataset_file = Path(sys.modules[fed_cls.__module__].__file__).resolve()
        license_file = dataset_file.parent / "dataset_creation_scripts" / "license_agreement_fed_heart_disease"
        if license_file.exists():
            return
        if self.auto_accept_license:
            license_file.parent.mkdir(parents=True, exist_ok=True)
            license_file.touch(exist_ok=True)
            return
        raise RuntimeError(
            "FLamby Fed-Heart-Disease requires accepting data terms first. "
            "Create the FLamby license marker file after reading terms or set auto_accept_license=True."
        )

    @staticmethod
    def _dataset_to_xy(dataset) -> tuple[np.ndarray, np.ndarray]:
        if len(dataset) == 0:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
        x_parts: list[np.ndarray] = []
        y_parts: list[int] = []
        for idx in range(len(dataset)):
            x_tensor, y_tensor = dataset[idx]
            x_parts.append(np.asarray(x_tensor, dtype=np.float32).reshape(-1))
            y_arr = np.asarray(y_tensor, dtype=np.float32).reshape(-1)
            y_parts.append(int(float(y_arr[0]) > 0.5))
        return np.stack(x_parts, axis=0), np.asarray(y_parts, dtype=np.int64)

    def load_clients(self) -> list[ClientData]:
        fed_cls = self._resolve_flamby_import()
        self._ensure_license(fed_cls)
        # FLamby HeartDisease still references np.NaN in some versions.
        if not hasattr(np, "NaN"):
            np.NaN = np.nan  # type: ignore[attr-defined]

        kwargs = {}
        if self.root is not None:
            kwargs["data_path"] = str(self.root)

        if self.preserve_native_clients:
            explicit_grouped: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
            explicit_ids: list[str] = []
            for center in range(4):
                train_center = fed_cls(center=center, train=True, pooled=False, **kwargs)
                test_center = fed_cls(center=center, train=False, pooled=False, **kwargs)
                x_train, y_train = self._dataset_to_xy(train_center)
                x_test, y_test = self._dataset_to_xy(test_center)
                explicit_grouped.append((x_train, y_train, x_test, y_test))
                explicit_ids.append(f"center_{center}")

            clients = _build_explicit_clients(
                grouped=_compact_explicit_split_labels(explicit_grouped),
                client_ids=explicit_ids,
                min_client_samples=self.min_client_samples,
                limit_clients=self.limit_clients,
                require_multi_class=False,
                normalization=self.normalization,
                max_train_samples_per_client=self.max_train_samples_per_client,
                max_test_samples_per_client=self.max_test_samples_per_client,
                seed=self.seed,
                device=self.device,
            )
        else:
            train_pooled = fed_cls(train=True, pooled=True, **kwargs)
            test_pooled = fed_cls(train=False, pooled=True, **kwargs)
            x_train, y_train = self._dataset_to_xy(train_pooled)
            x_test, y_test = self._dataset_to_xy(test_pooled)

            x = np.concatenate([x_train, x_test], axis=0)
            y = np.concatenate([y_train, y_test], axis=0)
            clients = _build_dirichlet_clients(
                x=x,
                y=y,
                num_clients=self.num_clients,
                alpha=self.alpha,
                test_size=self.test_size,
                min_client_samples=self.min_client_samples,
                limit_clients=self.limit_clients,
                normalization=self.normalization,
                max_train_samples_per_client=self.max_train_samples_per_client,
                max_test_samples_per_client=self.max_test_samples_per_client,
                seed=self.seed,
                device=self.device,
            )
        if not clients:
            raise RuntimeError("No valid FLamby Heart-Disease clients were created.")
        self._num_classes = 2
        return clients

    def num_classes(self) -> int:
        if self._num_classes is None:
            raise RuntimeError("Call load_clients() before requesting num_classes().")
        return self._num_classes


class FEMNISTAdapter(ClientDatasetAdapter):
    """Loads FEMNIST with LEAF user splits preserved per client."""

    def __init__(
        self,
        root: str | Path = "../fisher_FL_hd_unzipped/data/tiers/standard_pfl/femnist",
        *,
        limit_clients: int | None = 200,
        cache_limit_clients: int | None = None,
        cache_dir: str | Path | None = "cache/femnist",
        selection_seed: int | None = None,
        min_client_samples: int = 20,
        normalization: str | None = None,
        max_train_samples_per_client: int | None = None,
        max_test_samples_per_client: int | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.root = Path(root)
        self.limit_clients = limit_clients
        self.cache_limit_clients = cache_limit_clients
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.selection_seed = selection_seed
        self.min_client_samples = min_client_samples
        self.normalization = _resolve_normalization_mode(normalization, default="l2")
        self.max_train_samples_per_client = max_train_samples_per_client
        self.max_test_samples_per_client = max_test_samples_per_client
        self.device = torch.device(device)
        self._num_classes: int | None = None

    @staticmethod
    def _extract_xy(entry) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(entry, dict):
            if "x" in entry and "y" in entry:
                x, y = entry["x"], entry["y"]
            elif "inputs" in entry and "targets" in entry:
                x, y = entry["inputs"], entry["targets"]
            else:
                raise ValueError(f"Unsupported FEMNIST entry keys: {list(entry.keys())[:10]}")
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            x, y = entry
        else:
            raise ValueError(f"Unsupported FEMNIST entry type: {type(entry)}")
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        return x, y

    @staticmethod
    def _lookup_user(blob, user: str):
        if isinstance(blob, dict):
            if user in blob:
                return blob[user]
            if "user_data" in blob and user in blob["user_data"]:
                return blob["user_data"][user]
        raise KeyError(f"Could not find FEMNIST user '{user}' in loaded blob.")

    def _cache_path(self, pool_limit: int | None) -> Path | None:
        if self.cache_dir is None or pool_limit is None:
            return None
        cache_name = (
            f"femnist_pool_limit{pool_limit}"
            f"_selection{self.selection_seed}"
            f"_min{self.min_client_samples}"
            f"_norm{self.normalization}"
            f"_traincap{self.max_train_samples_per_client}"
            f"_testcap{self.max_test_samples_per_client}.pt"
        )
        return self.cache_dir / cache_name

    def _record_from_arrays(
        self,
        *,
        user: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        seed: int,
    ) -> dict[str, torch.Tensor | str]:
        client = _finalize_client_data(
            client_id=user,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            device=torch.device("cpu"),
            normalization=self.normalization,
            max_train_samples_per_client=self.max_train_samples_per_client,
            max_test_samples_per_client=self.max_test_samples_per_client,
            seed=seed,
        )
        if client is None:
            raise RuntimeError(f"Failed to materialize FEMNIST client {user}")
        return {
            "client_id": user,
            "x_train": client.x_train.cpu(),
            "y_train": client.y_train.cpu(),
            "x_test": client.x_test.cpu(),
            "y_test": client.y_test.cpu(),
        }

    def _extract_client_pool_from_raw(self, pool_limit: int | None) -> list[dict[str, torch.Tensor | str]]:
        leaf_train = self.root / "train"
        leaf_test = self.root / "test"
        if leaf_train.is_dir() and leaf_test.is_dir():
            # First enumerate all writer ids, then choose from that complete
            # pool.  The former implementation stopped at the first N ids in
            # lexical shard order, so `selection_seed` could not make the
            # requested subject sample random over the LEAF input.
            all_users: list[str] = []
            for path in sorted(leaf_train.glob("*.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                entries = payload.get("user_data", {})
                all_users.extend(str(user) for user in payload.get("users", []) if str(user) in entries)
            if pool_limit is not None and pool_limit < len(all_users):
                if self.selection_seed is None:
                    selected_users_order = all_users[:pool_limit]
                else:
                    rng = np.random.default_rng(self.selection_seed)
                    selected_indices = np.sort(rng.choice(len(all_users), size=pool_limit, replace=False))
                    selected_users_order = [all_users[int(index)] for index in selected_indices]
            else:
                selected_users_order = all_users
            selected_users = set(selected_users_order)

            train_users: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            test_users: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for path in sorted(leaf_train.glob("*.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                entries = payload.get("user_data", {})
                for user in payload.get("users", []):
                    user = str(user)
                    if user not in selected_users or user not in entries:
                        continue
                    train_users[user] = self._extract_xy(entries[user])
            for path in sorted(leaf_test.glob("*.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                entries = payload.get("user_data", {})
                for user in payload.get("users", []):
                    user = str(user)
                    if user in selected_users and user in entries:
                        test_users[user] = self._extract_xy(entries[user])
                if len(test_users) >= len(selected_users):
                    break
            clients: list[dict[str, torch.Tensor | str]] = []
            for user_idx, user in enumerate(selected_users_order):
                if user not in test_users:
                    continue
                x_train, y_train = train_users[user]
                x_test, y_test = test_users[user]
                if len(y_train) < self.min_client_samples or len(y_test) == 0:
                    continue
                clients.append(
                    self._record_from_arrays(
                        user=user,
                        x_train=x_train,
                        y_train=y_train,
                        x_test=x_test,
                        y_test=y_test,
                        seed=user_idx,
                    )
                )
                if pool_limit is not None and len(clients) >= pool_limit:
                    break
            if not clients:
                raise RuntimeError(f"No valid LEAF FEMNIST clients found under {self.root}")
            return clients
        user_blob = torch.load(self.root / "femnist_user_keys.pt", map_location="cpu", weights_only=False)
        train_blob = torch.load(self.root / "femnist_train.pt", map_location="cpu", weights_only=False)
        test_blob = torch.load(self.root / "femnist_test.pt", map_location="cpu", weights_only=False)
        users = [str(user) for user in user_blob["users"]]

        clients: list[dict[str, torch.Tensor | str]] = []
        if isinstance(train_blob, tuple) and len(train_blob) == 3 and isinstance(test_blob, tuple) and len(test_blob) == 3:
            x_train_all = np.asarray(train_blob[0], dtype=np.float32)
            y_train_all = np.asarray(train_blob[1], dtype=np.int64)
            u_train_all = np.asarray([str(user) for user in train_blob[2]])
            x_test_all = np.asarray(test_blob[0], dtype=np.float32)
            y_test_all = np.asarray(test_blob[1], dtype=np.int64)
            u_test_all = np.asarray([str(user) for user in test_blob[2]])

            for user_idx, user in enumerate(users):
                train_mask = u_train_all == user
                test_mask = u_test_all == user
                if not np.any(train_mask) or not np.any(test_mask):
                    continue
                x_train = x_train_all[train_mask]
                y_train = y_train_all[train_mask]
                x_test = x_test_all[test_mask]
                y_test = y_test_all[test_mask]
                if x_train.ndim > 2:
                    x_train = x_train.reshape(x_train.shape[0], -1)
                if x_test.ndim > 2:
                    x_test = x_test.reshape(x_test.shape[0], -1)
                if len(y_train) < self.min_client_samples or len(y_test) == 0:
                    continue
                clients.append(
                    self._record_from_arrays(
                        user=user,
                        x_train=x_train,
                        y_train=y_train,
                        x_test=x_test,
                        y_test=y_test,
                        seed=user_idx,
                    )
                )
                if pool_limit is not None and len(clients) >= pool_limit:
                    break
        else:
            for user_idx, user in enumerate(users):
                x_train, y_train = self._extract_xy(self._lookup_user(train_blob, user))
                x_test, y_test = self._extract_xy(self._lookup_user(test_blob, user))
                if len(y_train) < self.min_client_samples or len(y_test) == 0:
                    continue
                clients.append(
                    self._record_from_arrays(
                        user=user,
                        x_train=x_train,
                        y_train=y_train,
                        x_test=x_test,
                        y_test=y_test,
                        seed=user_idx,
                    )
                )
                if pool_limit is not None and len(clients) >= pool_limit:
                    break

        if not clients:
            raise RuntimeError(f"No valid FEMNIST clients found under {self.root}")
        return clients

    def _load_client_pool(self) -> list[dict[str, torch.Tensor | str]]:
        pool_limit = self.cache_limit_clients if self.cache_limit_clients is not None else self.limit_clients
        cache_path = self._cache_path(pool_limit)
        if cache_path is not None and cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            return payload["clients"]

        clients = self._extract_client_pool_from_raw(pool_limit)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"clients": clients}, cache_path)
        return clients

    def _select_client_pool(self, client_pool: list[dict[str, torch.Tensor | str]]) -> list[dict[str, torch.Tensor | str]]:
        if self.limit_clients is None or self.limit_clients >= len(client_pool):
            return client_pool
        if self.selection_seed is None:
            return client_pool[: self.limit_clients]
        rng = np.random.default_rng(self.selection_seed)
        selected_indices = np.sort(rng.choice(len(client_pool), size=self.limit_clients, replace=False))
        return [client_pool[int(index)] for index in selected_indices]

    def load_clients(self) -> list[ClientData]:
        clients = [
            ClientData(
                client_id=str(record["client_id"]),
                x_train=record["x_train"].to(self.device),
                y_train=record["y_train"].to(self.device),
                x_test=record["x_test"].to(self.device),
                y_test=record["y_test"].to(self.device),
            )
            for record in self._select_client_pool(self._load_client_pool())
        ]
        if not clients:
            raise RuntimeError(f"No valid FEMNIST clients found under {self.root}")
        self._num_classes = (
            int(max(max(client.y_train.max().item(), client.y_test.max().item()) for client in clients)) + 1
        )
        return clients

    def num_classes(self) -> int:
        if self._num_classes is None:
            raise RuntimeError("Call load_clients() before requesting num_classes().")
        return self._num_classes


class SyntheticAdapter(ClientDatasetAdapter):
    """Loads LEAF Synthetic explicit train/test user splits from JSON shards."""

    def __init__(
        self,
        root: str | Path = "data/leaf_synthetic/data",
        *,
        limit_clients: int | None = None,
        min_client_samples: int = 10,
        normalization: str | None = None,
        max_train_samples_per_client: int | None = None,
        max_test_samples_per_client: int | None = None,
        seed: int = 13,
        device: torch.device | str = "cpu",
    ) -> None:
        self.root = Path(root)
        self.limit_clients = limit_clients
        self.min_client_samples = min_client_samples
        self.normalization = _resolve_normalization_mode(normalization, default="none")
        self.max_train_samples_per_client = max_train_samples_per_client
        self.max_test_samples_per_client = max_test_samples_per_client
        self.seed = int(seed)
        self.device = torch.device(device)
        self._num_classes: int | None = None

    def _data_root(self) -> Path:
        if (self.root / "train").is_dir() and (self.root / "test").is_dir():
            return self.root
        if (self.root / "data" / "train").is_dir() and (self.root / "data" / "test").is_dir():
            return self.root / "data"
        raise RuntimeError(
            f"LEAF Synthetic train/test directories not found under {self.root}"
        )

    @staticmethod
    def _extract_xy(entry) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(entry, dict) or "x" not in entry or "y" not in entry:
            raise ValueError(f"Unsupported LEAF Synthetic entry: {type(entry)}")
        x = np.asarray(entry["x"], dtype=np.float32)
        y = np.asarray(entry["y"], dtype=np.int64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        elif x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        return x, y

    @staticmethod
    def _load_split_users(split_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        user_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for path in sorted(split_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            users = [str(user) for user in payload.get("users", [])]
            user_data = payload.get("user_data", {})
            for user in users:
                if user not in user_data:
                    continue
                user_map[user] = SyntheticAdapter._extract_xy(user_data[user])
        return user_map

    def load_clients(self) -> list[ClientData]:
        data_root = self._data_root()
        train_users = self._load_split_users(data_root / "train")
        test_users = self._load_split_users(data_root / "test")

        grouped: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        client_ids: list[str] = []
        for user in sorted(set(train_users) & set(test_users)):
            x_train, y_train = train_users[user]
            x_test, y_test = test_users[user]
            grouped.append((x_train, y_train, x_test, y_test))
            client_ids.append(user)

        clients = _build_explicit_clients(
            grouped=_compact_explicit_split_labels(grouped),
            client_ids=client_ids,
            min_client_samples=self.min_client_samples,
            limit_clients=self.limit_clients,
            normalization=self.normalization,
            max_train_samples_per_client=self.max_train_samples_per_client,
            max_test_samples_per_client=self.max_test_samples_per_client,
            seed=self.seed,
            device=self.device,
        )
        if not clients:
            raise RuntimeError(f"No valid LEAF Synthetic clients found under {data_root}")
        self._num_classes = (
            int(max(max(client.y_train.max().item(), client.y_test.max().item()) for client in clients)) + 1
        )
        return clients

    def num_classes(self) -> int:
        if self._num_classes is None:
            raise RuntimeError("Call load_clients() before requesting num_classes().")
        return self._num_classes


class WISDMAdapter(ClientDatasetAdapter):
    """Loads WISDM raw user logs directly from the archive into client splits."""

    def __init__(
        self,
        root: str | Path = "../fisher_FL_hd_unzipped/data/tiers/on_device_hdc/wisdm",
        *,
        wisdm_modality: str = "phone_accel",
        test_size: float = 0.3,
        min_client_samples: int = 20,
        limit_clients: int | None = 51,
        normalization: str | None = None,
        max_train_samples_per_client: int | None = None,
        max_test_samples_per_client: int | None = None,
        seed: int = 13,
        device: torch.device | str = "cpu",
    ) -> None:
        self.root = Path(root)
        self.wisdm_modality = str(wisdm_modality).lower()
        self.test_size = test_size
        self.min_client_samples = min_client_samples
        self.limit_clients = limit_clients
        self.normalization = _resolve_normalization_mode(normalization, default="standardize")
        self.max_train_samples_per_client = max_train_samples_per_client
        self.max_test_samples_per_client = max_test_samples_per_client
        self.seed = int(seed)
        self.device = torch.device(device)
        self._num_classes: int | None = None

    def _archive_path(self) -> Path:
        if self.root.is_file():
            return self.root
        archive = self.root / "wisdm-dataset.zip"
        if archive.exists():
            return archive
        fallback = self.root / "wisdm_dataset.zip"
        if fallback.exists():
            return fallback
        raise RuntimeError(f"WISDM archive not found under {self.root}")

    def _iter_user_series(self) -> list[tuple[str, np.ndarray, list[str]]]:
        archive = self._archive_path()
        try:
            device_name, sensor_name = self.wisdm_modality.split("_", 1)
        except ValueError as exc:
            raise ValueError(f"Expected wisdm_modality like 'phone_accel', got: {self.wisdm_modality}") from exc
        prefix = f"wisdm-dataset/raw/{device_name}/{sensor_name}/"

        rows: list[tuple[str, np.ndarray, list[str]]] = []
        with zipfile.ZipFile(archive) as zf:
            names = sorted(name for name in zf.namelist() if name.startswith(prefix) and name.endswith(".txt"))
            if not names:
                raise RuntimeError(f"No WISDM raw files found for modality {self.wisdm_modality} in {archive}")
            for name in names:
                user_id = Path(name).name.split("_")[1]
                user_rows: list[list[float]] = []
                user_labels: list[str] = []
                with zf.open(name) as handle:
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

    def load_clients(self) -> list[ClientData]:
        label_map: dict[str, int] = {}
        grouped: list[tuple[np.ndarray, np.ndarray]] = []
        client_ids: list[str] = []

        for user_id, x_user, labels in self._iter_user_series():
            y_user: list[int] = []
            for label in labels:
                if label not in label_map:
                    label_map[label] = len(label_map)
                y_user.append(label_map[label])
            grouped.append((x_user, np.asarray(y_user, dtype=np.int64)))
            client_ids.append(f"user_{user_id}")

        clients = _build_grouped_clients(
            grouped=_compact_grouped_labels(grouped),
            client_ids=client_ids,
            test_size=self.test_size,
            min_client_samples=self.min_client_samples,
            limit_clients=self.limit_clients,
            normalization=self.normalization,
            max_train_samples_per_client=self.max_train_samples_per_client,
            max_test_samples_per_client=self.max_test_samples_per_client,
            seed=self.seed,
            device=self.device,
        )
        if not clients:
            raise RuntimeError(f"No valid WISDM clients found under {self.root}")
        self._num_classes = int(max(max(client.y_train.max().item(), client.y_test.max().item()) for client in clients)) + 1
        return clients

    def num_classes(self) -> int:
        if self._num_classes is None:
            raise RuntimeError("Call load_clients() before requesting num_classes().")
        return self._num_classes


class UCIHARAdapter(ClientDatasetAdapter):
    """Loads the subject-partitioned UCI HAR dataset into client splits."""

    def __init__(
        self,
        root: str | Path = "../fisher_FL_hd_unzipped/data/tiers/on_device_hdc/uci_har/UCI HAR Dataset",
        *,
        test_size: float = 0.3,
        min_client_samples: int = 10,
        limit_clients: int | None = 30,
        normalization: str | None = None,
        max_train_samples_per_client: int | None = None,
        max_test_samples_per_client: int | None = None,
        preserve_original_split: bool = False,
        device: torch.device | str = "cpu",
    ) -> None:
        self.root = Path(root)
        self.test_size = test_size
        self.min_client_samples = min_client_samples
        self.limit_clients = limit_clients
        self.normalization = _resolve_normalization_mode(normalization, default="l2")
        self.max_train_samples_per_client = max_train_samples_per_client
        self.max_test_samples_per_client = max_test_samples_per_client
        self.preserve_original_split = preserve_original_split
        self.device = torch.device(device)
        self._num_classes: int | None = None

    def _load_matrix(self, *parts: str) -> np.ndarray:
        return np.loadtxt(self.root.joinpath(*parts), dtype=np.float32)

    def load_clients(self) -> list[ClientData]:
        x_train = self._load_matrix("train", "X_train.txt")
        y_train = self._load_matrix("train", "y_train.txt").astype(np.int64) - 1
        s_train = self._load_matrix("train", "subject_train.txt").astype(np.int64)
        x_test = self._load_matrix("test", "X_test.txt")
        y_test = self._load_matrix("test", "y_test.txt").astype(np.int64) - 1
        s_test = self._load_matrix("test", "subject_test.txt").astype(np.int64)

        explicit_grouped: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        explicit_ids: list[str] = []
        merged_grouped: list[tuple[np.ndarray, np.ndarray]] = []
        merged_ids: list[str] = []

        for subject in sorted(set(s_train.tolist()) | set(s_test.tolist())):
            train_mask = s_train == subject
            test_mask = s_test == subject
            if not np.any(train_mask) and not np.any(test_mask):
                continue
            client_id = f"subject_{int(subject)}"
            if self.preserve_original_split and np.any(train_mask) and np.any(test_mask):
                explicit_grouped.append((x_train[train_mask], y_train[train_mask], x_test[test_mask], y_test[test_mask]))
                explicit_ids.append(client_id)
                continue

            x_parts: list[np.ndarray] = []
            y_parts: list[np.ndarray] = []
            if np.any(train_mask):
                x_parts.append(x_train[train_mask])
                y_parts.append(y_train[train_mask])
            if np.any(test_mask):
                x_parts.append(x_test[test_mask])
                y_parts.append(y_test[test_mask])
            merged_grouped.append((np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)))
            merged_ids.append(client_id)

        clients = _build_explicit_clients(
            grouped=_compact_explicit_split_labels(explicit_grouped),
            client_ids=explicit_ids,
            min_client_samples=self.min_client_samples,
            limit_clients=None,
            normalization=self.normalization,
            max_train_samples_per_client=self.max_train_samples_per_client,
            max_test_samples_per_client=self.max_test_samples_per_client,
            seed=13,
            device=self.device,
        )
        clients.extend(
            _build_grouped_clients(
                grouped=_compact_grouped_labels(merged_grouped),
                client_ids=merged_ids,
                test_size=self.test_size,
                min_client_samples=self.min_client_samples,
                limit_clients=None,
                normalization=self.normalization,
                max_train_samples_per_client=self.max_train_samples_per_client,
                max_test_samples_per_client=self.max_test_samples_per_client,
                seed=13,
                device=self.device,
            )
        )

        if self.limit_clients is not None:
            clients = clients[: self.limit_clients]
        if not clients:
            raise RuntimeError(f"No valid UCI HAR clients found under {self.root}")

        self._num_classes = int(max(max(client.y_train.max().item(), client.y_test.max().item()) for client in clients)) + 1
        return clients

    def num_classes(self) -> int:
        if self._num_classes is None:
            raise RuntimeError("Call load_clients() before requesting num_classes().")
        return self._num_classes


class MHEALTHAdapter(ClientDatasetAdapter):
    """Loads MHEALTH per-subject logs with sliding-window segmentation."""

    def __init__(
        self,
        root: str | Path = "../fisher_FL_hd_unzipped/data/tiers/on_device_hdc/mhealth",
        *,
        split_mode: str = "per_activity_chrono",
        window_size: int = 128,
        window_stride: int = 64,
        strict_windows: bool = False,
        drop_ecg: bool = True,
        train_fraction: float = 0.7,
        val_fraction: float = 0.1,
        include_val_in_train: bool = False,
        min_client_samples: int = 20,
        limit_clients: int | None = 10,
        normalization: str | None = None,
        max_train_samples_per_client: int | None = None,
        max_test_samples_per_client: int | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.root = Path(root)
        mode = str(split_mode).lower()
        if mode not in {"global_chrono", "per_activity_chrono"}:
            raise ValueError(f"Unsupported MHEALTH split_mode: {split_mode}")
        self.split_mode = mode
        self.window_size = int(window_size)
        self.window_stride = int(window_stride)
        self.strict_windows = bool(strict_windows)
        self.drop_ecg = bool(drop_ecg)
        self.train_fraction = float(train_fraction)
        self.val_fraction = float(val_fraction)
        self.include_val_in_train = bool(include_val_in_train)
        self.min_client_samples = int(min_client_samples)
        self.limit_clients = limit_clients
        self.normalization = _resolve_normalization_mode(normalization, default="standardize")
        self.max_train_samples_per_client = max_train_samples_per_client
        self.max_test_samples_per_client = max_test_samples_per_client
        self.device = torch.device(device)
        self._num_classes: int | None = None

    def _dataset_root(self) -> Path:
        if (self.root / "MHEALTHDATASET").is_dir():
            return self.root / "MHEALTHDATASET"
        return self.root

    def load_clients(self) -> list[ClientData]:
        dataset_root = self._dataset_root()
        grouped: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        client_ids: list[str] = []

        for path in sorted(dataset_root.glob("mHealth_subject*.log")):
            arr = np.loadtxt(path, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] < 24:
                continue
            y_rows = arr[:, 23].astype(np.int64)
            keep = y_rows != 0
            if not np.any(keep):
                continue
            y_rows = y_rows[keep] - 1
            x_rows = arr[keep, :23]
            if self.drop_ecg:
                x_rows = np.delete(x_rows, [3, 4], axis=1)

            x_windows, y_windows = _window_flattened(
                x_rows,
                y_rows,
                window_size=self.window_size,
                stride=self.window_stride,
                strict_label=self.strict_windows,
            )
            if len(y_windows) == 0:
                continue
            if self.split_mode == "per_activity_chrono":
                x_train, y_train, x_test, y_test = _per_label_chronological_split(
                    x_windows,
                    y_windows,
                    train_fraction=self.train_fraction,
                    val_fraction=self.val_fraction,
                    include_val_in_train=self.include_val_in_train,
                )
            else:
                x_train, y_train, x_test, y_test = _chronological_split(
                    x_windows,
                    y_windows,
                    train_fraction=self.train_fraction,
                    val_fraction=self.val_fraction,
                    include_val_in_train=self.include_val_in_train,
                )
            if len(y_train) == 0 or len(y_test) == 0:
                continue

            subject_id = path.stem.split("subject")[-1]
            client_ids.append(f"subject_{subject_id}")
            grouped.append((x_train, y_train, x_test, y_test))

        clients = _build_explicit_clients(
            grouped=_compact_explicit_split_labels(grouped),
            client_ids=client_ids,
            min_client_samples=self.min_client_samples,
            limit_clients=self.limit_clients,
            normalization=self.normalization,
            max_train_samples_per_client=self.max_train_samples_per_client,
            max_test_samples_per_client=self.max_test_samples_per_client,
            seed=13,
            device=self.device,
        )
        if not clients:
            raise RuntimeError(f"No valid MHEALTH clients found under {dataset_root}")
        self._num_classes = int(max(max(client.y_train.max().item(), client.y_test.max().item()) for client in clients)) + 1
        return clients

    def num_classes(self) -> int:
        if self._num_classes is None:
            raise RuntimeError("Call load_clients() before requesting num_classes().")
        return self._num_classes


class USCHADAdapter(ClientDatasetAdapter):
    """Loads USC-HAD with subject clients and trial-based train/test split."""

    def __init__(
        self,
        root: str | Path = "../fisher_FL_hd_unzipped/data/tiers/on_device_hdc/usc_had",
        *,
        window_size: int = 128,
        window_stride: int = 64,
        train_trials: tuple[int, ...] = (1, 2, 3),
        val_trials: tuple[int, ...] = (4,),
        test_trials: tuple[int, ...] = (5,),
        include_val_in_train: bool = False,
        strict_windows: bool = False,
        min_client_samples: int = 20,
        limit_clients: int | None = 14,
        normalization: str | None = None,
        max_train_samples_per_client: int | None = None,
        max_test_samples_per_client: int | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.root = Path(root)
        self.window_size = int(window_size)
        self.window_stride = int(window_stride)
        self.train_trials = tuple(int(value) for value in train_trials)
        self.val_trials = tuple(int(value) for value in val_trials)
        self.test_trials = tuple(int(value) for value in test_trials)
        self.include_val_in_train = bool(include_val_in_train)
        self.strict_windows = bool(strict_windows)
        self.min_client_samples = int(min_client_samples)
        self.limit_clients = limit_clients
        self.normalization = _resolve_normalization_mode(normalization, default="standardize")
        self.max_train_samples_per_client = max_train_samples_per_client
        self.max_test_samples_per_client = max_test_samples_per_client
        self.device = torch.device(device)
        self._num_classes: int | None = None

    def _dataset_root(self) -> Path:
        if (self.root / "USC-HAD").is_dir():
            return self.root / "USC-HAD"
        return self.root

    @staticmethod
    def _parse_activity_trial(path: Path) -> tuple[int, int] | None:
        stem = path.stem.lower()
        if not stem.startswith("a") or "t" not in stem:
            return None
        sep = stem.find("t")
        if sep <= 1:
            return None
        try:
            activity_id = int(stem[1:sep])
            trial_id = int(stem[sep + 1 :])
        except ValueError:
            return None
        return activity_id, trial_id

    def load_clients(self) -> list[ClientData]:
        dataset_root = self._dataset_root()
        grouped: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        client_ids: list[str] = []

        for subject_dir in sorted(path for path in dataset_root.glob("Subject*") if path.is_dir()):
            train_parts: list[np.ndarray] = []
            train_labels: list[np.ndarray] = []
            val_parts: list[np.ndarray] = []
            val_labels: list[np.ndarray] = []
            test_parts: list[np.ndarray] = []
            test_labels: list[np.ndarray] = []

            for trial_path in sorted(subject_dir.glob("a*t*.mat")):
                parsed = self._parse_activity_trial(trial_path)
                if parsed is None:
                    continue
                activity_id, trial_id = parsed
                mat = loadmat(trial_path)
                if "sensor_readings" not in mat:
                    continue
                sensor = np.asarray(mat["sensor_readings"], dtype=np.float32)
                if sensor.ndim != 2 or sensor.shape[1] < 3:
                    continue
                x_rows = sensor
                y_rows = np.full((x_rows.shape[0],), int(activity_id) - 1, dtype=np.int64)
                x_windows, y_windows = _window_flattened(
                    x_rows,
                    y_rows,
                    window_size=self.window_size,
                    stride=self.window_stride,
                    strict_label=self.strict_windows,
                )
                if len(y_windows) == 0:
                    continue
                if trial_id in self.train_trials:
                    train_parts.append(x_windows)
                    train_labels.append(y_windows)
                elif trial_id in self.val_trials:
                    val_parts.append(x_windows)
                    val_labels.append(y_windows)
                elif trial_id in self.test_trials:
                    test_parts.append(x_windows)
                    test_labels.append(y_windows)

            if not train_parts or not test_parts:
                continue
            x_train = np.concatenate(train_parts, axis=0)
            y_train = np.concatenate(train_labels, axis=0)
            if self.include_val_in_train and val_parts:
                x_train = np.concatenate([x_train, *val_parts], axis=0)
                y_train = np.concatenate([y_train, *val_labels], axis=0)
            x_test = np.concatenate(test_parts, axis=0)
            y_test = np.concatenate(test_labels, axis=0)

            grouped.append((x_train, y_train, x_test, y_test))
            client_ids.append(subject_dir.name.lower())

        clients = _build_explicit_clients(
            grouped=_compact_explicit_split_labels(grouped),
            client_ids=client_ids,
            min_client_samples=self.min_client_samples,
            limit_clients=self.limit_clients,
            normalization=self.normalization,
            max_train_samples_per_client=self.max_train_samples_per_client,
            max_test_samples_per_client=self.max_test_samples_per_client,
            seed=13,
            device=self.device,
        )
        if not clients:
            raise RuntimeError(f"No valid USC-HAD clients found under {dataset_root}")
        self._num_classes = int(max(max(client.y_train.max().item(), client.y_test.max().item()) for client in clients)) + 1
        return clients

    def num_classes(self) -> int:
        if self._num_classes is None:
            raise RuntimeError("Call load_clients() before requesting num_classes().")
        return self._num_classes


class HHARAdapter(ClientDatasetAdapter):
    """Loads HHAR Activity-recognition streams with accel/gyro alignment per user/device."""

    def __init__(
        self,
        root: str | Path = "../fisher_FL_hd_unzipped/data/tiers/on_device_hdc/hhar",
        *,
        hhar_source: str = "watch",
        client_mode: str = "user",
        window_size: int = 128,
        window_stride: int = 64,
        strict_windows: bool = False,
        train_fraction: float = 0.7,
        val_fraction: float = 0.1,
        include_val_in_train: bool = False,
        resample_hz: float | None = 50.0,
        min_client_samples: int = 20,
        limit_clients: int | None = 9,
        normalization: str | None = None,
        max_train_samples_per_client: int | None = None,
        max_test_samples_per_client: int | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.root = Path(root)
        source = str(hhar_source).lower()
        if source not in {"phone", "watch"}:
            raise ValueError(f"Unsupported hhar_source: {hhar_source}")
        self.hhar_source = source
        mode = str(client_mode).lower()
        if mode not in {"user", "user_device"}:
            raise ValueError(f"Unsupported client_mode: {client_mode}")
        self.client_mode = mode
        self.window_size = int(window_size)
        self.window_stride = int(window_stride)
        self.strict_windows = bool(strict_windows)
        self.train_fraction = float(train_fraction)
        self.val_fraction = float(val_fraction)
        self.include_val_in_train = bool(include_val_in_train)
        self.resample_hz = None if resample_hz is None else float(resample_hz)
        self.min_client_samples = int(min_client_samples)
        self.limit_clients = limit_clients
        self.normalization = _resolve_normalization_mode(normalization, default="standardize")
        self.max_train_samples_per_client = max_train_samples_per_client
        self.max_test_samples_per_client = max_test_samples_per_client
        self.device = torch.device(device)
        self._num_classes: int | None = None

    def _activity_root(self) -> Path:
        if (self.root / "Activity recognition exp").is_dir():
            return self.root / "Activity recognition exp"
        return self.root

    @staticmethod
    def _read_sensor_df(path: Path):
        try:
            import pandas as pd
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("pandas is required to parse HHAR CSV files.") from exc

        df = pd.read_csv(
            path,
            usecols=["Arrival_Time", "x", "y", "z", "User", "Device", "gt"],
            low_memory=False,
        )
        df = df.dropna(subset=["Arrival_Time", "x", "y", "z", "User", "Device", "gt"])
        df["gt"] = df["gt"].astype(str)
        df = df[df["gt"].str.lower() != "null"]
        return df

    @staticmethod
    def _pair_streams_from_df(df, *, include_labels: bool):
        streams: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}
        for (user, device), group in df.groupby(["User", "Device"], sort=True):
            times = group["Arrival_Time"].to_numpy(dtype=np.float64)
            values = group[["x", "y", "z"]].to_numpy(dtype=np.float32)
            labels = group["gt"].astype(str).to_numpy() if include_labels else None
            order = np.argsort(times, kind="stable")
            times = times[order]
            values = values[order]
            if labels is not None:
                labels = labels[order]
            streams[(str(user), str(device))] = (times, values, labels)
        return streams

    def _resample_acc_stream(
        self,
        times: np.ndarray,
        values: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.resample_hz is None or self.resample_hz <= 0.0 or len(times) <= 2:
            return times, values, labels
        diffs = np.diff(times)
        diffs = diffs[diffs > 0]
        if diffs.size == 0:
            return times, values, labels
        median_dt_ms = float(np.median(diffs))
        target_dt_ms = 1000.0 / float(self.resample_hz)
        step = max(1, int(round(target_dt_ms / max(median_dt_ms, 1e-6))))
        if step <= 1:
            return times, values, labels
        return times[::step], values[::step], labels[::step]

    def load_clients(self) -> list[ClientData]:
        activity_root = self._activity_root()
        prefix = "Watch" if self.hhar_source == "watch" else "Phones"
        acc_path = activity_root / f"{prefix}_accelerometer.csv"
        gyro_path = activity_root / f"{prefix}_gyroscope.csv"
        if not acc_path.exists() or not gyro_path.exists():
            raise RuntimeError(
                f"HHAR source files not found under {activity_root}: "
                f"{acc_path.name}, {gyro_path.name}"
            )

        acc_df = self._read_sensor_df(acc_path)
        gyro_df = self._read_sensor_df(gyro_path)
        acc_streams = self._pair_streams_from_df(acc_df, include_labels=True)
        gyro_streams = self._pair_streams_from_df(gyro_df, include_labels=False)

        label_map: dict[str, int] = {}
        grouped_by_client: dict[str, dict[str, list[np.ndarray]]] = {}

        for user_device in sorted(set(acc_streams) & set(gyro_streams)):
            acc_t, acc_xyz, acc_labels = acc_streams[user_device]
            gyro_t, gyro_xyz, _ = gyro_streams[user_device]
            if acc_labels is None:
                continue
            if len(acc_t) < self.window_size or len(gyro_t) < 2:
                continue

            uniq_t, uniq_idx = np.unique(gyro_t, return_index=True)
            gyro_t = uniq_t
            gyro_xyz = gyro_xyz[uniq_idx]
            if len(gyro_t) < 2:
                continue

            acc_t, acc_xyz, acc_labels = self._resample_acc_stream(acc_t, acc_xyz, acc_labels)
            if len(acc_t) < self.window_size:
                continue

            gyro_interp = np.column_stack(
                [
                    np.interp(acc_t, gyro_t, gyro_xyz[:, axis]).astype(np.float32)
                    for axis in range(3)
                ]
            )
            x_rows = np.concatenate([acc_xyz.astype(np.float32), gyro_interp], axis=1)
            y_rows = np.asarray([label_map.setdefault(str(label), len(label_map)) for label in acc_labels], dtype=np.int64)

            x_windows, y_windows = _window_flattened(
                x_rows,
                y_rows,
                window_size=self.window_size,
                stride=self.window_stride,
                strict_label=self.strict_windows,
            )
            if len(y_windows) == 0:
                continue

            x_train, y_train, x_test, y_test = _chronological_split(
                x_windows,
                y_windows,
                train_fraction=self.train_fraction,
                val_fraction=self.val_fraction,
                include_val_in_train=self.include_val_in_train,
            )
            if len(y_train) == 0 or len(y_test) == 0:
                continue

            user, device = user_device
            if self.client_mode == "user":
                client_id = f"user_{user}"
            else:
                client_id = f"user_{user}__device_{device}"
            bucket = grouped_by_client.setdefault(
                client_id,
                {"x_train": [], "y_train": [], "x_test": [], "y_test": []},
            )
            bucket["x_train"].append(x_train)
            bucket["y_train"].append(y_train)
            bucket["x_test"].append(x_test)
            bucket["y_test"].append(y_test)

        grouped: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        client_ids: list[str] = []
        for client_id in sorted(grouped_by_client):
            bucket = grouped_by_client[client_id]
            if not bucket["x_train"] or not bucket["x_test"]:
                continue
            grouped.append(
                (
                    np.concatenate(bucket["x_train"], axis=0),
                    np.concatenate(bucket["y_train"], axis=0),
                    np.concatenate(bucket["x_test"], axis=0),
                    np.concatenate(bucket["y_test"], axis=0),
                )
            )
            client_ids.append(client_id)

        clients = _build_explicit_clients(
            grouped=_compact_explicit_split_labels(grouped),
            client_ids=client_ids,
            min_client_samples=self.min_client_samples,
            limit_clients=self.limit_clients,
            normalization=self.normalization,
            max_train_samples_per_client=self.max_train_samples_per_client,
            max_test_samples_per_client=self.max_test_samples_per_client,
            seed=13,
            device=self.device,
        )
        if not clients:
            raise RuntimeError(f"No valid HHAR clients found under {activity_root}")
        self._num_classes = int(max(max(client.y_train.max().item(), client.y_test.max().item()) for client in clients)) + 1
        return clients

    def num_classes(self) -> int:
        if self._num_classes is None:
            raise RuntimeError("Call load_clients() before requesting num_classes().")
        return self._num_classes


class PAMAP2Adapter(ClientDatasetAdapter):
    """Loads PAMAP2 per-subject sensor streams into client splits."""

    def __init__(
        self,
        root: str | Path = "../fisher_FL_hd_unzipped/data/tiers/on_device_hdc/pamap2/PAMAP2_Dataset/Protocol",
        *,
        test_size: float = 0.3,
        min_client_samples: int = 20,
        limit_clients: int | None = 8,
        normalization: str | None = None,
        max_train_samples_per_client: int | None = None,
        max_test_samples_per_client: int | None = None,
        seed: int = 13,
        device: torch.device | str = "cpu",
    ) -> None:
        self.root = Path(root)
        self.test_size = test_size
        self.min_client_samples = min_client_samples
        self.limit_clients = limit_clients
        self.normalization = _resolve_normalization_mode(normalization, default="standardize")
        self.max_train_samples_per_client = max_train_samples_per_client
        self.max_test_samples_per_client = max_test_samples_per_client
        self.seed = int(seed)
        self.device = torch.device(device)
        self._num_classes: int | None = None

    def load_clients(self) -> list[ClientData]:
        if not self.root.is_dir():
            raise RuntimeError(f"PAMAP2 protocol files not found under {self.root}")

        grouped: list[tuple[np.ndarray, np.ndarray]] = []
        client_ids: list[str] = []
        for path in sorted(self.root.glob("*.dat")):
            arr = np.loadtxt(path, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] < 3:
                continue
            y = arr[:, 1].astype(np.int64)
            keep = y != 0
            if not np.any(keep):
                continue
            x = _replace_nan_with_column_means(arr[keep, 2:])
            y = y[keep]
            grouped.append((x, y))
            client_ids.append(path.stem)

        clients = _build_grouped_clients(
            grouped=_compact_grouped_labels(grouped),
            client_ids=client_ids,
            test_size=self.test_size,
            min_client_samples=self.min_client_samples,
            limit_clients=self.limit_clients,
            normalization=self.normalization,
            max_train_samples_per_client=self.max_train_samples_per_client,
            max_test_samples_per_client=self.max_test_samples_per_client,
            seed=self.seed,
            device=self.device,
        )
        if not clients:
            raise RuntimeError(f"No valid PAMAP2 clients found under {self.root}")
        self._num_classes = int(max(max(client.y_train.max().item(), client.y_test.max().item()) for client in clients)) + 1
        return clients

    def num_classes(self) -> int:
        if self._num_classes is None:
            raise RuntimeError("Call load_clients() before requesting num_classes().")
        return self._num_classes


class NinaProDB1Adapter(ClientDatasetAdapter):
    """Loads NinaPro DB1 subject data using the requested modality."""

    def __init__(
        self,
        root: str | Path = "../fisher_FL_hd_unzipped/data/tiers/on_device_hdc/ninapro_db1",
        *,
        ninapro_modality: str = "emg_glove",
        test_size: float = 0.3,
        min_client_samples: int = 20,
        limit_clients: int | None = 27,
        normalization: str | None = None,
        max_train_samples_per_client: int | None = None,
        max_test_samples_per_client: int | None = None,
        seed: int = 13,
        device: torch.device | str = "cpu",
    ) -> None:
        self.root = Path(root)
        self.ninapro_modality = str(ninapro_modality).lower()
        self.test_size = test_size
        self.min_client_samples = min_client_samples
        self.limit_clients = limit_clients
        self.normalization = _resolve_normalization_mode(normalization, default="standardize")
        self.max_train_samples_per_client = max_train_samples_per_client
        self.max_test_samples_per_client = max_test_samples_per_client
        self.seed = int(seed)
        self.device = torch.device(device)
        self._num_classes: int | None = None

    def load_clients(self) -> list[ClientData]:
        if not self.root.is_dir():
            raise RuntimeError(f"NinaPro DB1 files not found under {self.root}")

        exercise_offsets = {1: 0, 2: 12, 3: 29}
        grouped: list[tuple[np.ndarray, np.ndarray]] = []
        client_ids: list[str] = []
        subject_names = sorted({path.name.split("_")[0] for path in self.root.glob("*.mat")})

        for subject in subject_names:
            x_parts: list[np.ndarray] = []
            y_parts: list[np.ndarray] = []
            for exercise in (1, 2, 3):
                path = self.root / f"{subject}_A1_E{exercise}.mat"
                if not path.exists():
                    continue
                mat = loadmat(path)
                emg = np.asarray(mat["emg"], dtype=np.float32)
                glove = np.asarray(mat["glove"], dtype=np.float32)
                labels = np.asarray(mat["restimulus"], dtype=np.int64).reshape(-1)
                keep = labels != 0
                if not np.any(keep):
                    continue
                if self.ninapro_modality == "emg":
                    x = emg[keep]
                elif self.ninapro_modality == "glove":
                    x = glove[keep]
                elif self.ninapro_modality in {"emg_glove", "all"}:
                    x = np.concatenate([emg[keep], glove[keep]], axis=1)
                else:
                    raise ValueError(f"Unsupported ninapro_modality: {self.ninapro_modality}")
                y = labels[keep] + exercise_offsets[exercise] - 1
                x_parts.append(x)
                y_parts.append(y.astype(np.int64))

            if not x_parts:
                continue
            grouped.append((np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)))
            client_ids.append(subject)

        clients = _build_grouped_clients(
            grouped=_compact_grouped_labels(grouped),
            client_ids=client_ids,
            test_size=self.test_size,
            min_client_samples=self.min_client_samples,
            limit_clients=self.limit_clients,
            normalization=self.normalization,
            max_train_samples_per_client=self.max_train_samples_per_client,
            max_test_samples_per_client=self.max_test_samples_per_client,
            seed=self.seed,
            device=self.device,
        )
        if not clients:
            raise RuntimeError(f"No valid NinaPro DB1 clients found under {self.root}")
        self._num_classes = int(max(max(client.y_train.max().item(), client.y_test.max().item()) for client in clients)) + 1
        return clients

    def num_classes(self) -> int:
        if self._num_classes is None:
            raise RuntimeError("Call load_clients() before requesting num_classes().")
        return self._num_classes
