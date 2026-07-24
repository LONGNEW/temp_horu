#!/usr/bin/env python3
"""Prepare and run the six-dataset CUDA reconstruction benchmark one dataset at a time."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_SCRIPT = Path(__file__).with_name("prepare_and_run_cuda_reconstruction_suite.py")
SPEC = importlib.util.spec_from_file_location("prepare_and_run_cuda_reconstruction_suite", SUITE_SCRIPT)
SUITE_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(SUITE_MODULE)


def _run(command: list[str], env: dict[str, str]) -> int:
    print("Command:\n" + " ".join(command))
    return subprocess.run(command, cwd=REPO_ROOT, env=env, check=False).returncode


def _dataset_source_flags(args: argparse.Namespace, dataset: str) -> list[str]:
    flags = []
    if dataset == "uci_har":
        flags.extend(["--uci-har-source-root", str(args.uci_har_source_root)])
    elif dataset == "isolet_raw":
        flags.extend(["--isolet-raw-source-root", str(args.isolet_raw_source_root)])
    elif dataset == "femnist":
        flags.extend(["--femnist-source-root", str(args.femnist_source_root)])
    elif dataset == "wisdm":
        flags.extend(["--wisdm-source-root", str(args.wisdm_source_root)])
    elif dataset == "synthetic":
        flags.extend(["--synthetic-source-root", str(args.synthetic_source_root)])
    elif dataset == "ninapro_db1":
        flags.extend(["--ninapro-db1-source-root", str(args.ninapro_db1_source_root)])
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return flags


def _dataset_run_command(args: argparse.Namespace, dataset: str, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "artifact/scripts/run_cuda_reconstruction_suite.py",
        "--dataset",
        dataset,
        "--output-dir",
        str(output_dir),
        *_dataset_source_flags(args, dataset),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=SUITE_MODULE.MANIFEST_DATASETS,
        dest="datasets",
        help="Limit the sequential run to one or more manifest dataset identifiers. Defaults to the full six-dataset suite.",
    )
    parser.add_argument("--uci-har-source-root", type=Path)
    parser.add_argument("--uci-har-archive", type=Path)
    parser.add_argument("--isolet-raw-source-root", type=Path)
    parser.add_argument("--isolet-download-dir", type=Path)
    parser.add_argument("--femnist-source-root", type=Path)
    parser.add_argument("--wisdm-source-root", type=Path)
    parser.add_argument("--wisdm-outer-archive", type=Path)
    parser.add_argument("--synthetic-source-root", type=Path)
    parser.add_argument("--ninapro-db1-source-root", type=Path)
    parser.add_argument("--ninapro-download-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running later datasets even if one dataset preparation or execution fails.",
    )
    args = parser.parse_args()

    try:
        selected = SUITE_MODULE._selected_datasets(args.datasets)
        commands_by_dataset = {
            dataset: SUITE_MODULE._prepare_commands(args, [dataset])
            for dataset in selected
        }
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.output_dir.exists():
        print(f"Refusing to overwrite existing sequential output root: {args.output_dir}", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True)

    env = {"PYTHONPATH": str(REPO_ROOT / "src"), **dict(os.environ)}
    status_rows: list[dict[str, object]] = []
    overall_rc = 0

    for dataset in selected:
        dataset_root = args.output_dir / dataset
        dataset_root.mkdir()
        dataset_rc = 0
        stage = "prepare"
        for command in commands_by_dataset[dataset]:
            dataset_rc = _run(command, env)
            if dataset_rc != 0:
                break
        if dataset_rc == 0:
            stage = "run"
            dataset_rc = _run(_dataset_run_command(args, dataset, dataset_root), env)
        status_rows.append(
            {
                "dataset": dataset,
                "status": "pass" if dataset_rc == 0 else "failed",
                "failed_stage": None if dataset_rc == 0 else stage,
                "return_code": dataset_rc,
                "output_dir": str(dataset_root),
            }
        )
        if dataset_rc != 0:
            overall_rc = 2
            if not args.continue_on_error:
                break

    summary = {
        "mode": "sequential_by_dataset",
        "selected_datasets": selected,
        "continue_on_error": bool(args.continue_on_error),
        "rows": status_rows,
    }
    summary_path = args.output_dir / "sequential_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"saved_summary={summary_path}")
    print(json.dumps(summary, indent=2))
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
