#!/usr/bin/env python3
"""Verify the PAMAP2 three-method metric-contract screening report."""

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
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["protocol"]
    report = json.loads(args.report.read_text(encoding="utf-8"))
    failures: list[str] = []
    config = report.get("run_config", {})
    for key in ("datasets", "methods", "local_epochs", "batch_size", "hd_dim", "hd_lr", "subspace_shared_rank", "subspace_intersection_rank", "subspace_personal_rank", "torch_num_threads"):
        expected = [manifest["dataset"]] if key == "datasets" else manifest.get(key)
        actual = config.get(key)
        if actual != expected:
            failures.append(f"run config {key!r} differs from manifest: {actual!r} != {expected!r}")
    if report.get("seeds") != [manifest["seed"]]:
        failures.append("top-level seeds differ from manifest")
    if report.get("round_checkpoints") != manifest["round_checkpoints"]:
        failures.append("top-level round checkpoints differ from manifest")
    if config.get("deterministic_algorithms") is not True:
        failures.append("run config does not record deterministic algorithms")
    summary = report.get("datasets", {}).get(manifest["dataset"], {}).get("summary", {})
    expected_metrics = {"horu_hd": "mean_personalized_accuracy", "hyperfeel": "mean_personalized_accuracy", "fedhdc": "global_test_accuracy"}
    for method, metric_key in expected_metrics.items():
        row = summary.get(method, {})
        value = row.get("per_round", {}).get("25", {}).get("mean")
        if row.get("metric_key") != metric_key:
            failures.append(f"{method} metric key is not {metric_key}")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            failures.append(f"{method} has no finite R25")
    if failures:
        print("PAMAP2 HD COMPARISON VERIFICATION FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("PAMAP2 HD COMPARISON VERIFIED: CUDA_RECONSTRUCTION_SCREENING_ONLY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
