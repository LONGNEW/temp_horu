#!/usr/bin/env python3
"""Prepare the controlled-systems fixture and reproduce Tables I, II, and III."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from horu_artifact.datasets.controlled_systems import prepare_data
from horu_artifact.experiments.tables123 import reproduce_tables


def _prepare_command(data_root: Path) -> list[str]:
    return ["prepare-controlled-systems", "--data-root", str(data_root)]


def _reproduce_command(data_root: Path, output_dir: Path, warmup: int, repeats: int, threads: int) -> list[str]:
    return [
        "reproduce-tables123",
        "--data-root",
        str(data_root),
        "--output-dir",
        str(output_dir),
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
        "--threads",
        str(threads),
    ]


def _print_csv(path: Path) -> None:
    print(f"\n== {path.name} ==")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        print("(empty)")
        return
    headers = list(rows[0].keys())
    widths = {
        header: max(len(header), *(len(str(row.get(header, ""))) for row in rows))
        for header in headers
    }
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers))


def _print_result_summary(output_dir: Path) -> None:
    print(f"\nOutputs written to: {output_dir}")
    for name in ("table1.csv", "table2.csv", "table3.csv", "raw_timings.csv", "environment.json", "result.json"):
        path = output_dir / name
        if path.exists():
            print(f"- {path}")
    result_path = output_dir / "result.json"
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        print(f"\nStatus: {payload.get('status', 'unknown')}")
    for name in ("table1.csv", "table2.csv", "table3.csv"):
        path = output_dir / name
        if path.exists():
            _print_csv(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    print("Command:\n" + " ".join(_prepare_command(args.data_root)))
    prepare_data(args.data_root)

    print(
        "Command:\n"
        + " ".join(_reproduce_command(args.data_root, args.output_dir, args.warmup, args.repeats, args.threads))
    )
    reproduce_tables(args.data_root, args.output_dir, warmup=args.warmup, repeats=args.repeats, threads=args.threads)
    _print_result_summary(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
