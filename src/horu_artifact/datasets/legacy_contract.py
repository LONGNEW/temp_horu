from __future__ import annotations

from pathlib import Path

import numpy as np

from .federated import ClientData


def resolve_normalization_mode(mode: str | None, *, default: str) -> str:
    resolved = default if mode is None else str(mode).lower()
    if resolved not in {"l2", "standardize", "none"}:
        raise ValueError(f"unsupported normalization mode: {mode}")
    return resolved


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms < eps] = 1.0
    return x / norms


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray, std_floor: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = np.maximum(x_train.std(axis=0, keepdims=True), std_floor)
    return (x_train - mean) / std, (x_test - mean) / std


def apply_normalization(x_train: np.ndarray, x_test: np.ndarray, *, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "none":
        return x_train, x_test
    if mode == "l2":
        return l2_normalize_rows(x_train), l2_normalize_rows(x_test)
    return standardize_train_test(x_train, x_test)


def capped_subset(x: np.ndarray, y: np.ndarray, max_samples: int | None, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None:
        return x, y
    max_samples = int(max_samples)
    if max_samples <= 0 or len(y) <= max_samples:
        return x, y
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    if max_samples >= len(classes):
        proportions = counts.astype(np.float64) / float(counts.sum())
        allocation = np.floor(proportions * max_samples).astype(int)
        allocation = np.maximum(allocation, 1)
        allocation = np.minimum(allocation, counts)
        while allocation.sum() > max_samples:
            reducible = np.where(allocation > 1)[0]
            if reducible.size == 0:
                break
            allocation[reducible[np.argmax(allocation[reducible])]] -= 1
        while allocation.sum() < max_samples:
            remaining = counts - allocation
            growable = np.where(remaining > 0)[0]
            if growable.size == 0:
                break
            allocation[growable[np.argmax(remaining[growable])]] += 1
        picked = []
        for cls, take in zip(classes, allocation):
            cls_idx = np.where(y == cls)[0]
            rng.shuffle(cls_idx)
            picked.append(cls_idx[:take])
        chosen = np.sort(np.concatenate(picked, axis=0))
        return x[chosen], y[chosen]
    chosen = np.sort(rng.choice(len(y), size=max_samples, replace=False))
    return x[chosen], y[chosen]


def split_train_test(x: np.ndarray, y: np.ndarray, *, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(y) < 2:
        raise ValueError("need at least two samples to split client data")
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    can_stratify = len(classes) >= 2 and counts.min() >= 2 and len(y) >= max(6, len(classes) * 2)
    if can_stratify:
        train_parts: list[np.ndarray] = []
        test_parts: list[np.ndarray] = []
        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            rng.shuffle(cls_idx)
            num_test = max(1, min(len(cls_idx) - 1, int(round(len(cls_idx) * float(test_size)))))
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


def compact_explicit_split_labels(grouped: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if not grouped:
        return []
    all_y = np.concatenate([np.concatenate([y_train, y_test], axis=0) for _, y_train, _, y_test in grouped], axis=0)
    classes = np.unique(all_y)
    mapping = {int(cls): idx for idx, cls in enumerate(classes.tolist())}
    compacted = []
    for x_train, y_train, x_test, y_test in grouped:
        compacted.append((
            x_train,
            np.asarray([mapping[int(label)] for label in y_train], dtype=np.int64),
            x_test,
            np.asarray([mapping[int(label)] for label in y_test], dtype=np.int64),
        ))
    return compacted


def compact_grouped_labels(grouped: list[tuple[np.ndarray, np.ndarray]]) -> list[tuple[np.ndarray, np.ndarray]]:
    if not grouped:
        return []
    all_y = np.concatenate([y for _, y in grouped], axis=0)
    classes = np.unique(all_y)
    mapping = {int(cls): idx for idx, cls in enumerate(classes.tolist())}
    return [(x_group, np.asarray([mapping[int(label)] for label in y_group], dtype=np.int64)) for x_group, y_group in grouped]


def _make_client(client_id: str, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, *, normalization: str, max_train_samples_per_client: int | None, max_test_samples_per_client: int | None, seed: int, base_id: int) -> ClientData | None:
    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    x_test = np.asarray(x_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int64)
    x_train, y_train = capped_subset(x_train, y_train, max_train_samples_per_client, seed=seed)
    x_test, y_test = capped_subset(x_test, y_test, max_test_samples_per_client, seed=seed + 1_000_003)
    if len(y_train) == 0 or len(y_test) == 0:
        return None
    x_train, x_test = apply_normalization(x_train, x_test, mode=normalization)
    train_ids = np.arange(base_id, base_id + len(y_train), dtype=np.int64)
    test_ids = np.arange(base_id + 5_000_000, base_id + 5_000_000 + len(y_test), dtype=np.int64)
    import torch
    return ClientData(
        train_x=torch.tensor(x_train, dtype=torch.float32),
        train_y=torch.tensor(y_train, dtype=torch.long),
        test_x=torch.tensor(x_test, dtype=torch.float32),
        test_y=torch.tensor(y_test, dtype=torch.long),
        train_ids=torch.tensor(train_ids, dtype=torch.long),
        test_ids=torch.tensor(test_ids, dtype=torch.long),
    )


def select_records(records: list[tuple[str, object]], limit: int | None, *, seed: int) -> list[tuple[str, object]]:
    if limit is None or limit >= len(records):
        return records
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(records), size=limit, replace=False))
    return [records[int(index)] for index in selected]


def build_explicit_clients(grouped: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], client_ids: list[str], *, min_client_samples: int, limit_clients: int | None, require_multi_class: bool = True, normalization: str, max_train_samples_per_client: int | None, max_test_samples_per_client: int | None, seed: int) -> dict[str, ClientData]:
    prepared: list[tuple[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], int]] = []
    for idx, (client_id, values) in enumerate(zip(client_ids, grouped)):
        x_train, y_train, x_test, y_test = values
        if len(y_train) < min_client_samples or len(y_test) == 0:
            continue
        if require_multi_class and len(np.unique(y_train)) < 2:
            continue
        prepared.append((client_id, values, idx))
    selected = select_records([(client_id, (values, idx)) for client_id, values, idx in prepared], limit_clients, seed=seed)
    clients: dict[str, ClientData] = {}
    for position, (client_id, (values, idx)) in enumerate(selected):
        client = _make_client(
            client_id,
            values[0],
            values[1],
            values[2],
            values[3],
            normalization=normalization,
            max_train_samples_per_client=max_train_samples_per_client,
            max_test_samples_per_client=max_test_samples_per_client,
            seed=seed + idx,
            base_id=position * 10_000_000,
        )
        if client is not None:
            clients[client_id] = client
    return dict(sorted(clients.items()))


def build_grouped_clients(grouped: list[tuple[np.ndarray, np.ndarray]], client_ids: list[str], *, test_size: float, min_client_samples: int, limit_clients: int | None, normalization: str, max_train_samples_per_client: int | None, max_test_samples_per_client: int | None, seed: int) -> dict[str, ClientData]:
    prepared: list[tuple[str, tuple[np.ndarray, np.ndarray], int]] = []
    for idx, (client_id, (x_group, y_group)) in enumerate(zip(client_ids, grouped)):
        if len(y_group) < min_client_samples or len(np.unique(y_group)) < 2:
            continue
        prepared.append((client_id, (x_group, y_group), idx))
    selected = select_records([(client_id, (values, idx)) for client_id, values, idx in prepared], limit_clients, seed=seed)
    clients: dict[str, ClientData] = {}
    for position, (client_id, (values, idx)) in enumerate(selected):
        x_train, y_train, x_test, y_test = split_train_test(np.asarray(values[0], dtype=np.float32), np.asarray(values[1], dtype=np.int64), test_size=test_size, seed=seed + idx)
        client = _make_client(
            client_id,
            x_train,
            y_train,
            x_test,
            y_test,
            normalization=normalization,
            max_train_samples_per_client=max_train_samples_per_client,
            max_test_samples_per_client=max_test_samples_per_client,
            seed=seed + idx,
            base_id=position * 10_000_000,
        )
        if client is not None:
            clients[client_id] = client
    return dict(sorted(clients.items()))


def dirichlet_client_indices(y: np.ndarray, *, num_clients: int, alpha: float, seed: int) -> list[np.ndarray]:
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
        for client_id, split in enumerate(np.split(shuffled, cuts)):
            client_indices[client_id].extend(split.tolist())
    return [np.asarray(sorted(indices), dtype=np.int64) for indices in client_indices]


def build_dirichlet_clients(x: np.ndarray, y: np.ndarray, *, num_clients: int, alpha: float, test_size: float, min_client_samples: int, limit_clients: int | None, normalization: str, max_train_samples_per_client: int | None, max_test_samples_per_client: int | None, seed: int) -> dict[str, ClientData]:
    prepared: list[tuple[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], int]] = []
    for client_id, indices in enumerate(dirichlet_client_indices(y, num_clients=num_clients, alpha=alpha, seed=seed)):
        if len(indices) < min_client_samples:
            continue
        x_client = x[indices]
        y_client = y[indices]
        if len(np.unique(y_client)) < 2:
            continue
        prepared.append((f"client_{client_id}", split_train_test(x_client, y_client, test_size=test_size, seed=seed + client_id), client_id))
    selected = select_records([(client_id, (values, idx)) for client_id, values, idx in prepared], limit_clients, seed=seed)
    clients: dict[str, ClientData] = {}
    for position, (client_id, (values, idx)) in enumerate(selected):
        client = _make_client(
            client_id,
            values[0],
            values[1],
            values[2],
            values[3],
            normalization=normalization,
            max_train_samples_per_client=max_train_samples_per_client,
            max_test_samples_per_client=max_test_samples_per_client,
            seed=seed + idx,
            base_id=position * 10_000_000,
        )
        if client is not None:
            clients[client_id] = client
    return dict(sorted(clients.items()))


def resolve_split_root(root: str | Path) -> Path:
    root_path = Path(root)
    if (root_path / "train").is_dir() and (root_path / "test").is_dir():
        return root_path
    if (root_path / "data" / "train").is_dir() and (root_path / "data" / "test").is_dir():
        return root_path / "data"
    raise FileNotFoundError(f"train/test split directories not found under {root_path}")
