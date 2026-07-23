"""LEAF FEMNIST natural-client cache builder."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch
from .federated import ClientData, FederatedDataset, write_cache


def _shards(directory: Path) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for writer, data in payload.get("user_data", {}).items():
            if writer in output: raise ValueError(f"duplicate FEMNIST writer {writer}")
            output[writer] = data
    if not output: raise FileNotFoundError(f"no LEAF FEMNIST JSON shards in {directory}")
    return output


def _tensors(data: dict, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.tensor(data["x"], dtype=torch.float32).reshape(-1, 784) / 255.0
    y = torch.tensor(data["y"], dtype=torch.long)
    if x.shape[0] != y.numel(): raise ValueError("FEMNIST x/y count mismatch")
    return torch.nn.functional.normalize(x, p=2, dim=1), y, ids


def prepare_data(data_root: str | Path, source_root: str | Path) -> FederatedDataset:
    root = Path(source_root)
    train, test = _shards(root / "train"), _shards(root / "test")
    writers = sorted(set(train) & set(test))[:200]
    if len(writers) != 200: raise ValueError("FEMNIST requires 200 writers shared by LEAF train/test")
    clients = {}
    for index, writer in enumerate(writers):
        tx, ty, _ = _tensors(train[writer], torch.empty(0, dtype=torch.long))
        vx, vy, _ = _tensors(test[writer], torch.empty(0, dtype=torch.long))
        tids = torch.arange(index * 10_000_000, index * 10_000_000 + ty.numel(), dtype=torch.long)
        vids = torch.arange(index * 10_000_000 + 5_000_000, index * 10_000_000 + 5_000_000 + vy.numel(), dtype=torch.long)
        clients[writer] = ClientData(tx, ty, vx, vy, tids, vids)
    sha = hashlib.sha256()
    for path in sorted((root / "train").glob("*.json")) + sorted((root / "test").glob("*.json")): sha.update(path.read_bytes())
    manifest = {"source": str(root), "license": "LEAF/FEMNIST", "leaf_json_sha256": sha.hexdigest(), "parser": "leaf_femnist_v1",
                "clients": 200, "features": 784, "classes": 62, "writer_selection": "sorted_common_train_test_first_200",
                "normalization": "pixel_div_255_then_samplewise_l2", "split": "LEAF_writer_supplied_niid_split",
                "provenance": "USER_SPECIFIED_LEAF_NATURAL_SPLIT"}
    return write_cache(FederatedDataset("femnist", clients, 784, 62, manifest), data_root)
