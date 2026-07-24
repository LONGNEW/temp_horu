#!/usr/bin/env python3
"""Verify the declared Synthetic seed-42 rank-sweep screening report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = json.loads(args.report.read_text(encoding="utf-8"))
    protocol = manifest["protocol"]
    config = report.get("run_config", {})
    failures: list[str] = []
    expected = {
        "datasets": [protocol["dataset"]],
        "seed": protocol["seed"],
        "round_checkpoints": protocol["round_checkpoints"],
        "local_epochs": protocol["local_epochs"],
        "batch_size": protocol["batch_size"],
        "hd_dim": protocol["hd_dim"],
        "hd_lr": protocol["hd_lr"],
        "preset": protocol["rank_preset"],
        "method_key": "horu_hd",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            failures.append(f"run_config {key!r} differs: {config.get(key)!r} != {value!r}")
    rows = report.get("datasets", {}).get(protocol["dataset"], {}).get("rows", [])
    configs = config.get("configs", [])
    if len(rows) != len(configs) or len(rows) != 9:
        failures.append(f"expected nine rank rows/configs, got rows={len(rows)} configs={len(configs)}")
    labels = {item.get("label") for item in configs}
    for row in rows:
        if row.get("seed") != protocol["seed"] or row.get("method") != "horu_hd":
            failures.append("row has unexpected seed or method")
            continue
        if row.get("metric_key") != "mean_personalized_accuracy":
            failures.append(f"row {row.get('config', {}).get('label')} has an unexpected metric")
        label = row.get("config", {}).get("label")
        value = row.get("checkpoint_metrics", {}).get("25", {}).get("mean_personalized_accuracy")
        if label not in labels or not isinstance(value, (int, float)) or not math.isfinite(value):
            failures.append(f"row {label!r} lacks a finite R25 personalized score")
        elif not math.isclose(float(row.get("primary_value", float("nan"))), float(value), rel_tol=0.0, abs_tol=1e-12):
            failures.append(f"row {label!r} primary value does not equal R25 personalized score")
    if failures:
        print("RANK SWEEP VERIFICATION FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"RANK SWEEP VERIFIED: {manifest['result_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
