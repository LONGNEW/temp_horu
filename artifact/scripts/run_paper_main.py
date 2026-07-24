#!/usr/bin/env python3
"""Run the paper-main protocol only after its provenance gate is satisfied."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "artifact" / "manifests" / "paper_main_v1.json"


def build_common_args(manifest: dict[str, object], device: str) -> list[str]:
    protocol = manifest["protocol"]
    assert isinstance(protocol, dict)
    seed_protocol = manifest["seed_protocol"]
    assert isinstance(seed_protocol, dict)
    neural_protocol = manifest["neural_protocol"]
    assert isinstance(neural_protocol, dict)
    femnist = neural_protocol["femnist"]
    non_femnist = neural_protocol["non_femnist"]
    assert isinstance(femnist, dict) and isinstance(non_femnist, dict)
    seeds = [str(seed) for seed in seed_protocol["seeds"]]
    return [
        "--device", device,
        "--seeds", *seeds,
        "--round-checkpoints", str(protocol["rounds"]),
        "--local-epochs", str(protocol["local_epochs"]),
        "--batch-size", str(protocol["batch_size"]),
        "--client-participation", str(protocol["client_participation"]),
        "--hd-dim", str(protocol["hd_dim"]),
        "--hd-lr", str(protocol["hd_lr"]),
        "--nn-lr", str(non_femnist["lr"]),
        "--cnn-lr", str(femnist["lr"]),
        "--cnn-optimizer", str(femnist["optimizer"]),
        "--cnn-momentum", str(femnist["momentum"]),
        "--cnn-weight-decay", str(femnist["weight_decay"]),
        "--dfl-align-weight", str(femnist["dfl_align_weight"]),
        "--dfl-disentangle-weight", str(femnist["dfl_disentangle_weight"]),
        "--subspace-shared-rank", str(protocol["shared_rank"]),
        "--subspace-intersection-rank", str(protocol["common_rank"]),
        "--subspace-personal-rank", str(protocol["personal_rank"]),
        "--large-dataset-train-threshold", str(protocol["large_dataset_train_threshold"]),
        "--large-dataset-train-cap", str(protocol["large_dataset_train_cap"]),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, help="explicit device selected by the evaluator, e.g. cuda or cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    preflight = subprocess.run([sys.executable, "artifact/scripts/preflight.py", "--profile", "paper-main"], cwd=REPO_ROOT)
    if preflight.returncode != 0:
        return preflight.returncode
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    protocol = manifest["protocol"]
    datasets = list(protocol["datasets"])
    common = build_common_args(manifest, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    jobs = [
        ("hd", datasets, ["horu_hd", "fedhdc", "hyperfeel"]),
        ("nn_vector", [d for d in datasets if d != "femnist"], ["fedavg_mlp", "dfl_mlp"]),
        ("nn_femnist", ["femnist"], ["fedavg_cnn", "dfl_cnn"]),
    ]
    for name, job_datasets, methods in jobs:
        command = [
            sys.executable,
            "run_hd_checkpoint_comparison.py",
            "--datasets", *job_datasets,
            "--methods", *methods,
            *common,
            "--json-out", str(args.output_dir / f"{name}.json"),
            "--md-out", str(args.output_dir / f"{name}.md"),
        ]
        print("Run Manifest")
        print("Command:")
        print(" ".join(command))
        print("Result status: VALID_EXPERIMENT_CANDIDATE")
        completed = subprocess.run(command, cwd=REPO_ROOT, env={**os.environ, "HORU_SOURCE_DATA_ROOT": os.environ.get("HORU_SOURCE_DATA_ROOT", manifest["data_contract"]["default_source_root"])})
        if completed.returncode != 0:
            return completed.returncode
    return subprocess.run(
        [sys.executable, "artifact/scripts/summarize_paper_main.py", "--input-dir", str(args.output_dir)],
        cwd=REPO_ROOT,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
