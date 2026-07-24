#!/usr/bin/env python3
"""Run the manifest-bound six-dataset CUDA reconstruction screening suite."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "artifact" / "manifests" / "reconstruction_cuda_suite_seed42_v1.json"
SUMMARY = REPO_ROOT / "artifact" / "scripts" / "summarize_reconstruction_suite.py"
ALL_DATASETS = ("uci_har", "isolet_raw", "femnist", "wisdm", "synthetic", "ninapro_db1")
DATASET_RELATIVE_ROOTS = {
    "uci_har": Path("data/tiers/on_device_hdc/uci_har/UCI HAR Dataset"),
    "isolet_raw": Path("data/raw/isolet"),
    "femnist": Path("data/tiers/standard_pfl/femnist"),
    "wisdm": Path("data/tiers/on_device_hdc/wisdm"),
    "synthetic": Path("data/leaf_synthetic/data"),
    "ninapro_db1": Path("data/tiers/on_device_hdc/ninapro_db1"),
}


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


def _selected_datasets(requested: list[str] | None, protocol: dict[str, object]) -> list[str]:
    allowed = list(protocol["datasets"])
    if not requested:
        return allowed
    selected: list[str] = []
    seen: set[str] = set()
    for dataset in requested:
        if dataset not in allowed:
            raise ValueError(f"Dataset {dataset} is not in the manifest protocol")
        if dataset not in seen:
            seen.add(dataset)
            selected.append(dataset)
    return selected


def _materialize_source_root(dataset: str, source_root: Path, dataset_output: Path) -> Path:
    expected = source_root / DATASET_RELATIVE_ROOTS[dataset]
    if expected.exists():
        return source_root

    staged_root = dataset_output / "_source_root"
    target = staged_root / DATASET_RELATIVE_ROOTS[dataset]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.symlink_to(source_root.resolve(), target_is_directory=source_root.is_dir())
    return staged_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", choices=ALL_DATASETS, dest="datasets")
    for dataset in ALL_DATASETS:
        parser.add_argument(f"--{dataset.replace('_', '-')}-source-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    protocol = manifest["protocol"]
    selected = _selected_datasets(args.datasets, protocol)
    if args.output_dir.exists():
        print(f"Refusing to overwrite existing suite output: {args.output_dir}", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True)
    reports: list[str] = []
    for dataset in selected:
        source_root = getattr(args, f"{dataset}_source_root")
        if source_root is None:
            print(f"Missing source root flag for selected dataset {dataset}", file=sys.stderr)
            return 2
        if not source_root.is_dir():
            print(f"Missing source root for {dataset}: {source_root}", file=sys.stderr)
            return 2
        dataset_output = args.output_dir / dataset
        dataset_output.mkdir()
        runtime_source_root = _materialize_source_root(dataset, source_root, dataset_output)
        command = command_for(dataset, source_root, dataset_output, protocol)
        print("Run Manifest\nCommand:\n" + " ".join(command))
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env={**os.environ, "HORU_SOURCE_DATA_ROOT": str(runtime_source_root)},
            check=False,
        )
        if completed.returncode != 0:
            return completed.returncode
        reports.extend(["--report", f"{dataset}={dataset_output / f'{dataset}.json'}"])
    if len(selected) != len(protocol["datasets"]):
        partial_summary = {
            "status": "PARTIAL_DATASET_RUN",
            "selected_datasets": selected,
            "manifest": str(MANIFEST_PATH),
            "report_paths": {
                dataset: str(args.output_dir / dataset / f"{dataset}.json")
                for dataset in selected
            },
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(partial_summary, indent=2) + "\n",
            encoding="utf-8",
        )
        return 0
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
