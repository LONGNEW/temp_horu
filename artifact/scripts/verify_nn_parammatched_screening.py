#!/usr/bin/env python3
"""Verify the declared Synthetic seed-42 NN parameter-matched screen."""

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
    failures: list[str] = []
    if report.get("seeds") != [protocol["seed"]] or report.get("round_checkpoints") != protocol["round_checkpoints"]:
        failures.append("seed or checkpoints differ from manifest")
    config = report.get("run_config", {})
    for key in ("datasets", "methods", "local_epochs", "batch_size", "client_participation", "hd_lr", "nn_lr", "max_hidden_dim"):
        expected = [protocol["dataset"]] if key == "datasets" else protocol.get(key)
        if expected is not None and config.get(key) != expected:
            failures.append(f"run config {key!r} differs from manifest")
    rows = report.get("datasets", {}).get(protocol["dataset"], {}).get("rows", [])
    by_method = {row.get("method"): row for row in rows}
    for method in protocol["methods"]:
        row = by_method.get(method)
        expected_metric = protocol["metric_contract"].get(method)
        if row is None:
            failures.append(f"missing {method}")
            continue
        if row.get("metric_key") != expected_metric:
            failures.append(f"{method} metric differs from contract")
            continue
        value = row.get("checkpoint_metrics", {}).get("25", {}).get(expected_metric)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            failures.append(f"{method} has no finite R25 score")
        elif not math.isclose(float(value), float(row.get("primary_value", float("nan"))), rel_tol=0.0, abs_tol=1e-12):
            failures.append(f"{method} primary score differs from its R25 metric")
    if failures:
        print("NN PARAMETER-MATCHED SCREENING VERIFICATION FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"NN PARAMETER-MATCHED SCREENING VERIFIED: {manifest['result_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
