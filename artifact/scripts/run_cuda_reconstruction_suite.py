#!/usr/bin/env python3
"""Run the manifest-bound six-dataset CUDA reconstruction screening suite."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "artifact" / "manifests" / "reconstruction_cuda_suite_seed42_v1.json"
SUMMARY = REPO_ROOT / "artifact" / "scripts" / "summarize_reconstruction_suite.py"


def command_for(dataset: str, source_root: Path, output: Path, protocol: dict[str, object]) -> list[str]:
    return [
        sys.executable, "run_hd_checkpoint_comparison.py",
        "--datasets", dataset, "--methods", *protocol["methods"],
        "--device", str(protocol["device"]), "--deterministic-algorithms",
        "--torch-num-threads", str(protocol["torch_num_threads"]),
        "--seeds", str(protocol["seed"]),
        "--round-checkpoints", *(str(value) for value in protocol["round_checkpoints"]),
        "--local-epochs", str(protocol["local_epochs"]), "--batch-size", str(protocol["batch_size"]),
        "--client-participation", str(protocol["client_participation"]),
        "--hd-dim", str(protocol["hd_dim"]), "--hd-lr", str(protocol["hd_lr"]),
        "--subspace-shared-rank", str(protocol["subspace_shared_rank"]),
        "--subspace-intersection-rank", str(protocol["subspace_intersection_rank"]),
        "--subspace-personal-rank", str(protocol["subspace_personal_rank"]),
        "--json-out", str(output / f"{dataset}.json"), "--md-out", str(output / f"{dataset}.md"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for dataset in ("uci_har", "isolet_raw", "femnist", "wisdm", "synthetic", "ninapro_db1"):
        parser.add_argument(f"--{dataset.replace('_', '-')}-source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    protocol = manifest["protocol"]
    if args.output_dir.exists():
        print(f"Refusing to overwrite existing suite output: {args.output_dir}", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True)
    reports: list[str] = []
    for dataset in protocol["datasets"]:
        source_root = getattr(args, f"{dataset}_source_root")
        if not source_root.is_dir():
            print(f"Missing source root for {dataset}: {source_root}", file=sys.stderr)
            return 2
        dataset_output = args.output_dir / dataset
        dataset_output.mkdir()
        command = command_for(dataset, source_root, dataset_output, protocol)
        print("Run Manifest\nCommand:\n" + " ".join(command))
        completed = subprocess.run(command, cwd=REPO_ROOT, env={**os.environ, "HORU_SOURCE_DATA_ROOT": str(source_root)}, check=False)
        if completed.returncode != 0:
            return completed.returncode
        reports.extend(["--report", f"{dataset}={dataset_output / f'{dataset}.json'}"])
    summary_command = [
        sys.executable,
        str(SUMMARY),
        *reports,
        "--manifest",
        str(MANIFEST_PATH),
        "--output",
        str(args.output_dir / "summary.json"),
    ]
    return subprocess.run(summary_command, cwd=REPO_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
