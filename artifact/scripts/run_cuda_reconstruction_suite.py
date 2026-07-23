#!/usr/bin/env python3
"""Run the current six-dataset accuracy suite through ``horu_artifact``.

This wrapper keeps the old script location but routes execution through the
active cache builders and accuracy-suite runner under ``src/horu_artifact``.
It is a convenience entry point, not the immutable seed-42 reference-suite
generator used to create ``reference_results/cuda_suite_seed42``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "artifact" / "manifests" / "reconstruction_cuda_suite_seed42_v1.json"


def _run(command: list[str], env: dict[str, str]) -> int:
    print("Command:\n" + " ".join(command))
    return subprocess.run(command, cwd=REPO_ROOT, env=env, check=False).returncode


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
    for dataset in ("isolet_raw", "femnist", "wisdm", "ninapro_db1"):
        source_root = getattr(args, f"{dataset}_source_root")
        if not source_root.exists():
            print(f"Missing source root for {dataset}: {source_root}", file=sys.stderr)
            return 2
    data_root = args.output_dir / "data"
    results_root = args.output_dir / "results"
    datasets_config = args.output_dir / "datasets.generated.json"
    suite_config = args.output_dir / "accuracy_suite.generated.json"
    datasets_config.write_text(json.dumps({
        "seed": int(protocol["seed"]),
        "sources": {
            "isolet": str(args.isolet_raw_source_root),
            "femnist": str(args.femnist_source_root),
            "wisdm": str(args.wisdm_source_root),
            "ninapro": str(args.ninapro_db1_source_root),
        },
        "wisdm_client_ids": list(range(1600, 1651)),
        "wisdm_recover_missing_from_raw": True,
        "source_roots_record_only": {
            "uci_har": str(args.uci_har_source_root),
            "synthetic": str(args.synthetic_source_root),
        },
    }, indent=2) + "\n", encoding="utf-8")
    suite_config.write_text(json.dumps({
        "datasets": ["ucihar", "isolet", "femnist", "wisdm", "synthetic", "ninapro"],
        "methods": ["fedhdc", "hyperfeel", "horu"],
        "seeds": [int(protocol["seed"])],
        "rounds": int(protocol["rounds"]),
        "participation": float(protocol["client_participation"]),
        "local_epochs": int(protocol["local_epochs"]),
        "batch_size": int(protocol["batch_size"]),
        "hd_dim": int(protocol["hd_dim"]),
        "hd_learning_rate": float(protocol["hd_lr"]),
        "device": str(protocol["device"]),
        "horu": {
            "common_rank": int(protocol["subspace_intersection_rank"]),
            "global_rank": int(protocol["subspace_shared_rank"]) - int(protocol["subspace_intersection_rank"]),
            "personal_rank": int(protocol["subspace_personal_rank"]),
            "eta_shared": float(protocol["hd_lr"]),
            "eta_personal": float(protocol["hd_lr"]),
            "eta_global": float(protocol["hd_lr"]),
        },
    }, indent=2) + "\n", encoding="utf-8")
    env = {"PYTHONPATH": str(REPO_ROOT / "src"), **dict(__import__("os").environ)}
    prepare = [sys.executable, "-m", "horu_artifact", "prepare-data", "all", "--config", str(datasets_config), "--data-root", str(data_root)]
    if _run(prepare, env) != 0:
        return 2
    run_suite = [sys.executable, "-m", "horu_artifact", "run-suite", "--config", str(suite_config), "--data-root", str(data_root), "--output", str(results_root)]
    if _run(run_suite, env) != 0:
        return 2
    validate = [sys.executable, "-m", "horu_artifact", "validate-results", "--results", str(results_root)]
    return _run(validate, env)


if __name__ == "__main__":
    raise SystemExit(main())
