#!/usr/bin/env python3
"""Verify a manifest-bound PAMAP2 HoRU/EG-Lite screening report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    protocol = manifest["protocol"]
    summary = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
    comparison = json.loads((args.output_dir / "comparison.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    for key in ("seed", "round_checkpoints", "local_epochs", "batch_size", "hd_dim", "hd_lr", "shared_rank", "personal_rank", "intersection_rank", "torch_num_threads"):
        if summary.get("run_config", {}).get(key) != protocol.get(key):
            failures.append(f"summary config {key!r} differs from manifest")
    if summary.get("run_config", {}).get("deterministic_algorithms") is not True:
        failures.append("summary does not record deterministic algorithms")
    rows = {row.get("method"): row for row in comparison.get("datasets", {}).get("pamap2", {}).get("rows", [])}
    summary_rows = {row.get("method"): row.get("r25") for row in summary.get("rows", [])}
    for method in protocol["methods"]:
        row = rows.get(method)
        value = None if row is None else row.get("checkpoint_metrics", {}).get("25", {}).get("mean_personalized_accuracy")
        if row is None or row.get("metric_key") != "mean_personalized_accuracy" or not isinstance(value, (int, float)):
            failures.append(f"missing personalized R25 metric for {method}")
        elif not math.isclose(float(value), float(summary_rows.get(method, float("nan"))), rel_tol=0.0, abs_tol=1e-12):
            failures.append(f"summary mismatch for {method}")
    if failures:
        print("PAMAP2 EG-LITE SCREENING VERIFICATION FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"PAMAP2 EG-LITE SCREENING VERIFIED: {manifest['result_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
