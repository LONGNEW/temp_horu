#!/usr/bin/env python3
"""Verify the manifest-bound C1 Synthetic checkpoint screening output."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_single_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected exactly one data row in {path}, found {len(rows)}")
    return rows[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    protocol = manifest["protocol"]
    output_manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = read_single_csv_row(args.output_dir / "summary.csv")
    failures: list[str] = []

    for key in ("cluster", "seed", "rounds", "round_checkpoints"):
        if output_manifest.get(key) != protocol.get(key):
            failures.append(f"output manifest {key!r} differs from artifact manifest")
    expected = {
        "cluster_id": protocol["cluster"],
        "dataset": protocol["dataset"],
        "seed": str(protocol["seed"]),
        "rounds": str(protocol["rounds"]),
        "config_label": "baseline",
        "heterogeneity_pack": protocol["heterogeneity_pack"],
        "device_type": "cuda",
        "status": "done",
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            failures.append(f"summary {key!r} is {summary.get(key)!r}, expected {value!r}")
    for checkpoint in protocol["round_checkpoints"]:
        value = summary.get(f"R{checkpoint}")
        if value is None or not math.isfinite(float(value)):
            failures.append(f"missing finite checkpoint R{checkpoint}")
    if not math.isclose(float(summary["final_acc"]), float(summary["R25"]), rel_tol=0.0, abs_tol=1e-12):
        failures.append("final_acc does not equal R25")

    with (args.output_dir / "round_metrics.csv").open(newline="", encoding="utf-8") as handle:
        round_rows = list(csv.DictReader(handle))
    final_rows = [row for row in round_rows if row.get("round") == str(protocol["rounds"])]
    if len(final_rows) != 1:
        failures.append("round_metrics.csv must contain exactly one final-round row")
    elif not math.isclose(float(final_rows[0]["personalized_acc"]), float(summary["R25"]), rel_tol=0.0, abs_tol=1e-12):
        failures.append("final personalized_acc does not equal summary R25")

    if failures:
        print("SPECIALIZED C1 SCREENING VERIFICATION FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"SPECIALIZED C1 SCREENING VERIFIED: {manifest['result_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
