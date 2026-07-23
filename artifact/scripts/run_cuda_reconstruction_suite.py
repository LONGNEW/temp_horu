#!/usr/bin/env python3
"""Run the CUDA reconstruction suite through ``horu_artifact``.

This wrapper keeps the old script location but routes execution through the
active cache builders and accuracy-suite runner under ``src/horu_artifact``.
It accepts either the top-level preparation directories documented in the
README or the already-resolved inner data paths. It can also run a dataset
subset, which is useful for splitting the six-dataset suite across machines.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "artifact" / "manifests" / "reconstruction_cuda_suite_v1.json"
MANIFEST_DATASETS = ("uci_har", "isolet_raw", "femnist", "wisdm", "synthetic", "ninapro_db1")
RUNTIME_DATASETS = {
    "uci_har": "ucihar",
    "isolet_raw": "isolet",
    "femnist": "femnist",
    "wisdm": "wisdm",
    "synthetic": "synthetic",
    "ninapro_db1": "ninapro",
}
PREPARE_DATASETS = {
    "uci_har": "ucihar",
    "isolet_raw": "isolet",
    "femnist": "femnist",
    "wisdm": "wisdm",
    "synthetic": "synthetic",
    "ninapro_db1": "ninapro",
}


def _run(command: list[str], env: dict[str, str]) -> int:
    print("Command:\n" + " ".join(command))
    return subprocess.run(command, cwd=REPO_ROOT, env=env, check=False).returncode


def _resolve_existing_path(label: str, candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Missing prepared input for {label}. Checked:\n{tried}")


def _contains_any(path: Path, pattern: str) -> bool:
    return path.is_dir() and any(path.glob(pattern))


def _resolve_isolet_source(root: Path) -> Path:
    candidates = []
    if root.is_dir() and (root / "isolet1+2+3+4.data").is_file() and (root / "isolet5.data").is_file():
        candidates.append(root)
    candidates.append(root / "data" / "raw" / "isolet")
    return _resolve_existing_path("isolet_raw", candidates)


def _resolve_femnist_source(root: Path) -> Path:
    candidates = []
    if (root / "train").is_dir() and (root / "test").is_dir():
        candidates.append(root)
    candidates.append(root / "data" / "tiers" / "standard_pfl" / "femnist")
    return _resolve_existing_path("femnist", candidates)


def _resolve_wisdm_source(root: Path) -> Path:
    candidates = []
    if root.is_file():
        candidates.append(root)
    candidates.append(root / "wisdm-dataset.zip")
    candidates.append(root / "data" / "tiers" / "on_device_hdc" / "wisdm" / "wisdm-dataset.zip")
    return _resolve_existing_path("wisdm", candidates)


def _resolve_ninapro_source(root: Path) -> Path:
    candidates = []
    if _contains_any(root, "S*_A1_E*.mat"):
        candidates.append(root)
    candidates.append(root / "data" / "tiers" / "on_device_hdc" / "ninapro_db1")
    return _resolve_existing_path("ninapro_db1", candidates)


def _selected_datasets(requested: list[str] | None) -> list[str]:
    ordered = requested or list(MANIFEST_DATASETS)
    seen = set()
    result = []
    for dataset in ordered:
        if dataset not in seen:
            seen.add(dataset)
            result.append(dataset)
    return result


def _resolve_selected_sources(args: argparse.Namespace, selected: list[str]) -> dict[str, str]:
    sources: dict[str, str] = {}
    if "isolet_raw" in selected:
        if args.isolet_raw_source_root is None:
            raise FileNotFoundError("Missing --isolet-raw-source-root for selected dataset isolet_raw")
        sources["isolet"] = str(_resolve_isolet_source(args.isolet_raw_source_root))
    if "femnist" in selected:
        if args.femnist_source_root is None:
            raise FileNotFoundError("Missing --femnist-source-root for selected dataset femnist")
        sources["femnist"] = str(_resolve_femnist_source(args.femnist_source_root))
    if "wisdm" in selected:
        if args.wisdm_source_root is None:
            raise FileNotFoundError("Missing --wisdm-source-root for selected dataset wisdm")
        sources["wisdm"] = str(_resolve_wisdm_source(args.wisdm_source_root))
    if "ninapro_db1" in selected:
        if args.ninapro_db1_source_root is None:
            raise FileNotFoundError("Missing --ninapro-db1-source-root for selected dataset ninapro_db1")
        sources["ninapro"] = str(_resolve_ninapro_source(args.ninapro_db1_source_root))
    return sources


def _datasets_payload(protocol: dict, args: argparse.Namespace, sources: dict[str, str]) -> dict:
    payload = {
        "seed": int(protocol["seed"]),
        "sources": sources,
        "wisdm_client_ids": list(range(1600, 1651)),
        "wisdm_recover_missing_from_raw": True,
    }
    record_only = {}
    if args.uci_har_source_root is not None:
        record_only["uci_har"] = str(args.uci_har_source_root)
    if args.synthetic_source_root is not None:
        record_only["synthetic"] = str(args.synthetic_source_root)
    if record_only:
        payload["source_roots_record_only"] = record_only
    return payload


def _suite_payload(protocol: dict, selected: list[str]) -> dict:
    return {
        "datasets": [RUNTIME_DATASETS[dataset] for dataset in selected],
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
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=MANIFEST_DATASETS,
        dest="datasets",
        help="Limit the run to one or more manifest dataset identifiers. Defaults to the full six-dataset suite.",
    )
    for dataset in MANIFEST_DATASETS:
        parser.add_argument(f"--{dataset.replace('_', '-')}-source-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    protocol = manifest["protocol"]
    selected = _selected_datasets(args.datasets)
    if args.output_dir.exists():
        print(f"Refusing to overwrite existing suite output: {args.output_dir}", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True)

    try:
        resolved_sources = _resolve_selected_sources(args, selected)
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 2

    data_root = args.output_dir / "data"
    results_root = args.output_dir / "results"
    datasets_config = args.output_dir / "datasets.generated.json"
    suite_config = args.output_dir / "accuracy_suite.generated.json"
    datasets_config.write_text(
        json.dumps(_datasets_payload(protocol, args, resolved_sources), indent=2) + "\n",
        encoding="utf-8",
    )
    suite_config.write_text(
        json.dumps(_suite_payload(protocol, selected), indent=2) + "\n",
        encoding="utf-8",
    )

    env = {"PYTHONPATH": str(REPO_ROOT / "src"), **dict(os.environ)}
    for dataset in selected:
        prepare = [
            sys.executable,
            "-m",
            "horu_artifact",
            "prepare-data",
            PREPARE_DATASETS[dataset],
            "--config",
            str(datasets_config),
            "--data-root",
            str(data_root),
        ]
        if _run(prepare, env) != 0:
            return 2

    run_suite = [
        sys.executable,
        "-m",
        "horu_artifact",
        "run-suite",
        "--config",
        str(suite_config),
        "--data-root",
        str(data_root),
        "--output",
        str(results_root),
    ]
    if _run(run_suite, env) != 0:
        return 2
    validate = [sys.executable, "-m", "horu_artifact", "validate-results", "--results", str(results_root)]
    return _run(validate, env)


if __name__ == "__main__":
    raise SystemExit(main())
