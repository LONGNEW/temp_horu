"""Bridge GitHub canonical adapters into the immutable local cache contract.

The adapters are vendored read-only under the shared data root at the exact
GitHub commit recorded in the experiment config.  This module intentionally
does not reimplement their preprocessing or split logic.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import torch

from .federated import ClientData, FederatedDataset, write_cache


def _load_remote_adapters(adapter_file: Path):
    spec = importlib.util.spec_from_file_location("horu_github_canonical_data", adapter_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import GitHub adapter source: {adapter_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_ids(client_id: str, count: int, split: str) -> torch.Tensor:
    """Produce deterministic cache ids without changing adapter tensors."""
    prefix = int(hashlib.sha256(f"{client_id}:{split}".encode()).hexdigest()[:8], 16)
    return torch.arange(count, dtype=torch.long) + prefix * 1_000_000


def prepare_github_canonical(config_path: str | Path, data_root: str | Path) -> dict[str, Any]:
    """Run the pinned GitHub adapters and persist their exact client tensors."""
    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    canonical = cfg["canonical_preprocessing"]
    adapter_file = Path(canonical["adapter_file"])
    if not adapter_file.is_file():
        raise FileNotFoundError(f"GitHub adapter source is absent: {adapter_file}")
    remote = _load_remote_adapters(adapter_file)
    prepared: dict[str, Any] = {}
    for name, spec in canonical["datasets"].items():
        adapter_class = getattr(remote, spec["adapter"])
        kwargs = dict(spec["kwargs"])
        adapter = adapter_class(device="cpu", **kwargs)
        remote_clients = adapter.load_clients()
        clients: dict[str, ClientData] = {}
        for source_client in sorted(remote_clients, key=lambda row: str(row.client_id)):
            cid = str(source_client.client_id)
            clients[cid] = ClientData(
                source_client.x_train.detach().cpu().contiguous(),
                source_client.y_train.detach().cpu().long().contiguous(),
                source_client.x_test.detach().cpu().contiguous(),
                source_client.y_test.detach().cpu().long().contiguous(),
                _stable_ids(cid, source_client.y_train.numel(), "train"),
                _stable_ids(cid, source_client.y_test.numel(), "test"),
            )
        source_paths = {key: str(value) for key, value in spec["source_paths"].items()}
        source_hashes = {
            key: _file_sha256(Path(value)) for key, value in source_paths.items() if Path(value).is_file()
        }
        manifest = {
            "preprocessing": "github_canonical_adapter",
            "github_repository": canonical["repository"],
            "github_commit": canonical["commit"],
            "adapter_file": str(adapter_file),
            "adapter_file_sha256": _file_sha256(adapter_file),
            "adapter_class": spec["adapter"],
            "adapter_kwargs": kwargs,
            "source_paths": source_paths,
            "source_file_sha256": source_hashes,
            "sample_ids": "deterministic bridge ids; not used by GitHub preprocessing",
        }
        dataset = FederatedDataset(name, clients, int(remote_clients[0].x_train.shape[1]), adapter.num_classes(), manifest)
        prepared[name] = write_cache(dataset, data_root).statistics()
    return prepared
