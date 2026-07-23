#!/usr/bin/env python3
"""Prepare public inputs and run the six-dataset CUDA reconstruction suite."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DATASETS = ("uci_har", "isolet_raw", "femnist", "wisdm", "synthetic", "ninapro_db1")


def _run(command: list[str], env: dict[str, str]) -> int:
    print("Command:\n" + " ".join(command))
    return subprocess.run(command, cwd=REPO_ROOT, env=env, check=False).returncode


def _selected_datasets(requested: list[str] | None) -> list[str]:
    ordered = requested or list(MANIFEST_DATASETS)
    seen = set()
    result = []
    for dataset in ordered:
        if dataset not in seen:
            seen.add(dataset)
            result.append(dataset)
    return result


def _prepare_commands(args: argparse.Namespace, selected: list[str]) -> list[list[str]]:
    commands: list[list[str]] = []
    if "uci_har" in selected:
        if args.uci_har_source_root is None or args.uci_har_archive is None:
            raise ValueError("uci_har requires --uci-har-source-root and --uci-har-archive")
        commands.append(
            [
                sys.executable,
                "artifact/scripts/acquire_uci_har_prototype.py",
                "--source-root",
                str(args.uci_har_source_root),
                "--archive",
                str(args.uci_har_archive),
            ]
        )
    if "isolet_raw" in selected:
        if args.isolet_raw_source_root is None or args.isolet_download_dir is None:
            raise ValueError("isolet_raw requires --isolet-raw-source-root and --isolet-download-dir")
        commands.append(
            [
                sys.executable,
                "artifact/scripts/acquire_isolet_prototype.py",
                "--source-root",
                str(args.isolet_raw_source_root),
                "--download-dir",
                str(args.isolet_download_dir),
            ]
        )
    if "femnist" in selected:
        if args.femnist_source_root is None:
            raise ValueError("femnist requires --femnist-source-root")
        commands.append(
            [
                sys.executable,
                "artifact/scripts/prepare_femnist_reconstruction.py",
                "--source-root",
                str(args.femnist_source_root),
            ]
        )
    if "wisdm" in selected:
        if args.wisdm_source_root is None or args.wisdm_outer_archive is None:
            raise ValueError("wisdm requires --wisdm-source-root and --wisdm-outer-archive")
        commands.append(
            [
                sys.executable,
                "artifact/scripts/acquire_wisdm_reconstruction.py",
                "--source-root",
                str(args.wisdm_source_root),
                "--outer-archive",
                str(args.wisdm_outer_archive),
            ]
        )
    if "synthetic" in selected:
        if args.synthetic_source_root is None:
            raise ValueError("synthetic requires --synthetic-source-root")
        commands.append(
            [
                sys.executable,
                "artifact/scripts/prepare_synthetic_reconstruction.py",
                "--source-root",
                str(args.synthetic_source_root),
            ]
        )
    if "ninapro_db1" in selected:
        if args.ninapro_db1_source_root is None or args.ninapro_download_dir is None:
            raise ValueError("ninapro_db1 requires --ninapro-db1-source-root and --ninapro-download-dir")
        commands.append(
            [
                sys.executable,
                "artifact/scripts/acquire_ninapro_db1_reconstruction.py",
                "--source-root",
                str(args.ninapro_db1_source_root),
                "--download-dir",
                str(args.ninapro_download_dir),
            ]
        )
    return commands


def _suite_command(args: argparse.Namespace, selected: list[str]) -> list[str]:
    command = [
        sys.executable,
        "artifact/scripts/run_cuda_reconstruction_suite.py",
        "--output-dir",
        str(args.output_dir),
    ]
    for dataset in selected:
        command.extend(["--dataset", dataset])
    if "uci_har" in selected:
        command.extend(["--uci-har-source-root", str(args.uci_har_source_root)])
    if "isolet_raw" in selected:
        command.extend(["--isolet-raw-source-root", str(args.isolet_raw_source_root)])
    if "femnist" in selected:
        command.extend(["--femnist-source-root", str(args.femnist_source_root)])
    if "wisdm" in selected:
        command.extend(["--wisdm-source-root", str(args.wisdm_source_root)])
    if "synthetic" in selected:
        command.extend(["--synthetic-source-root", str(args.synthetic_source_root)])
    if "ninapro_db1" in selected:
        command.extend(["--ninapro-db1-source-root", str(args.ninapro_db1_source_root)])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=MANIFEST_DATASETS,
        dest="datasets",
        help="Limit the run to one or more manifest dataset identifiers. Defaults to the full six-dataset suite.",
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
    args = parser.parse_args()

    try:
        selected = _selected_datasets(args.datasets)
        commands = _prepare_commands(args, selected)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    env = {"PYTHONPATH": str(REPO_ROOT / "src"), **dict(os.environ)}
    for command in commands:
        if _run(command, env) != 0:
            return 2
    return _run(_suite_command(args, selected), env)


if __name__ == "__main__":
    raise SystemExit(main())
