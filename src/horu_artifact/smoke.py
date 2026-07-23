"""Independent local-client nonlinear HDC smoke workflow."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from pathlib import Path

import torch
import yaml

from .config import SmokeConfig
from .datasets.ucihar import DOI, load_cache, split_subjects
from .hdc.encoder import NonlinearEncoder, make_projection
from .hdc.prototype import PrototypeMemory
from .runtime import resolve_device


def run_smoke(config: SmokeConfig, data_root: str | Path, output: str | Path, device_override: str | None = None, overwrite: bool = False) -> dict:
    """Run independent local prototype learning and persist reproducible artifacts."""
    output_path = Path(output)
    if output_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory {output_path} is non-empty; pass --overwrite")
    output_path.mkdir(parents=True, exist_ok=True)
    requested = device_override or config.device
    device = resolve_device(requested)
    data = load_cache(data_root)  # Deliberately never downloads.
    splits = split_subjects(data, config.subject_ids, config.test_ratio, config.seed)
    projection_cpu = make_projection(data.features.shape[1], config.hd_dim, config.seed)
    projection_hash = hashlib.sha256(projection_cpu.numpy().tobytes()).hexdigest()
    encoder = NonlinearEncoder(projection_cpu.to(device))
    records, updates, initial_scores, final_scores = [], 0, [], []
    for client_id in config.subject_ids:
        indices = splits[client_id]
        train_x, train_y = data.features[indices["train"]].to(device), data.labels[indices["train"]].to(device)
        test_x, test_y = data.features[indices["test"]].to(device), data.labels[indices["test"]].to(device)
        train_h, test_h = encoder.encode(train_x), encoder.encode(test_x)
        memory = PrototypeMemory.initialize(train_h, train_y, 6, config.normalize_prototypes)
        initial = _accuracy(memory, test_h, test_y, config.similarity)
        for epoch in range(config.local_epochs):
            generator = torch.Generator(device="cpu").manual_seed(config.seed + client_id * 1009 + epoch)
            order = torch.randperm(train_h.shape[0], generator=generator, device="cpu").to(device)
            for start in range(0, order.numel(), config.batch_size):
                selection = order[start:start + config.batch_size]
                if config.update_mode == "samplewise":
                    updates += memory.update(train_h[selection], train_y[selection], config.learning_rate, config.similarity, config.normalize_update_hypervectors)
                else:
                    updates += memory.update_hdzoo_batch(train_h[selection], train_y[selection], config.learning_rate, config.similarity, config.normalize_update_hypervectors)
        final = _accuracy(memory, test_h, test_y, config.similarity)
        initial_scores.append(initial); final_scores.append(final)
        records.append({"client_id": client_id, "train_samples": int(train_y.numel()), "test_samples": int(test_y.numel()), "initial_accuracy": initial, "final_accuracy": final, "projection_sha256": projection_hash})
    resolved = config.to_dict(); resolved["device"] = str(device)
    result = {"status": "pass", "dataset": "ucihar", "dataset_doi": DOI, "dataset_sha256": data.manifest.get("sha256", ""), "seed": config.seed, "device": str(device), "similarity": config.similarity, "update_mode": config.update_mode, "normalize_prototypes": config.normalize_prototypes, "normalize_update_hypervectors": config.normalize_update_hypervectors, "torch_version": torch.__version__, "num_clients": 3, "client_ids": config.subject_ids, "num_classes": 6, "input_dim": 561, "hd_dim": config.hd_dim, "projection_shape": [561, config.hd_dim], "projection_sha256": projection_hash, "initial_mean_accuracy": sum(initial_scores) / 3, "final_mean_accuracy": sum(final_scores) / 3, "num_updates": updates, "per_client": records, "output_files": ["result.json", "config.resolved.yaml", "environment.json", "client_metrics.csv"]}
    (output_path / "config.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    (output_path / "environment.json").write_text(json.dumps({"python": platform.python_version(), "torch": torch.__version__, "device": str(device)}, indent=2) + "\n", encoding="utf-8")
    with (output_path / "client_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    (output_path / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _accuracy(memory: PrototypeMemory, encoded: torch.Tensor, labels: torch.Tensor, similarity: str) -> float:
    return float((memory.predict(encoded, similarity) == labels).float().mean().item())
