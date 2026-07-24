#!/usr/bin/env python3
"""Verify a manifest-bound seed-42 HoRU EG-Lite screening directory."""

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
    cfg = summary.get("run_config", {})
    failures: list[str] = []
    for key in ("seed", "round_checkpoints", "local_epochs", "batch_size", "hd_dim", "hd_lr", "torch_num_threads"):
        if cfg.get(key) != protocol.get(key):
            failures.append(f"summary config {key!r} differs from manifest")
    if cfg.get("deterministic_algorithms") is not True:
        failures.append("summary does not record deterministic algorithms")
    rows_by_dataset = {row.get("dataset"): row for row in summary.get("rows", [])}
    for dataset in protocol["datasets"]:
        row = rows_by_dataset.get(dataset)
        report_path = args.output_dir / f"{dataset}_horu_eg_lite_hd.json"
        if row is None or not report_path.is_file():
            failures.append(f"missing summary/report for {dataset}")
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        result_rows = report.get("datasets", {}).get(dataset, {}).get("rows", [])
        methods = {entry.get("method"): entry for entry in result_rows}
        for method in protocol["methods"]:
            entry = methods.get(method)
            if entry is None:
                failures.append(f"{dataset} lacks {method}")
                continue
            value = entry.get("checkpoint_metrics", {}).get("25", {}).get("mean_personalized_accuracy")
            if entry.get("metric_key") != "mean_personalized_accuracy" or not isinstance(value, (float, int)):
                failures.append(f"{dataset}/{method} lacks R25 personalized metric")
                continue
            expected = row["baseline_r25"] if method == "horu_hd" else row["variant_r25"]
            if not math.isclose(float(value), float(expected), rel_tol=0.0, abs_tol=1e-12):
                failures.append(f"{dataset}/{method} conflicts with summary")
    if failures:
        print("EG-LITE SCREENING VERIFICATION FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"EG-LITE SCREENING VERIFIED: {manifest['result_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
