#!/usr/bin/env python3
"""Compute a complete, explicitly labeled seed-42 reconstruction-screening mean."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_DATASETS = ("uci_har", "isolet_raw", "femnist", "wisdm", "synthetic", "ninapro_db1")
EXPECTED_METHODS = ("horu_hd", "hyperfeel", "fedhdc")
EXPECTED_METRICS = {
    "horu_hd": "mean_personalized_accuracy",
    "hyperfeel": "mean_personalized_accuracy",
    "fedhdc": "global_test_accuracy",
}


def parse_report_argument(value: str) -> tuple[str, Path]:
    dataset, separator, path = value.partition("=")
    if not separator or dataset not in EXPECTED_DATASETS or not path:
        raise argparse.ArgumentTypeError("report must be DATASET=/absolute/path.json for a manuscript dataset")
    return dataset, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", type=parse_report_argument, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    protocol = manifest.get("protocol", {})
    if tuple(protocol.get("datasets", [])) != EXPECTED_DATASETS:
        print("Manifest dataset contract differs from the reconstruction suite")
        return 2
    if set(protocol.get("methods", [])) != set(EXPECTED_METHODS):
        print("Manifest method contract differs from the reconstruction suite")
        return 2
    if protocol.get("seed") != 42 or 25 not in protocol.get("round_checkpoints", []):
        print("Manifest seed or round-25 contract differs from the reconstruction suite")
        return 2
    reports = dict(args.report)
    missing = sorted(set(EXPECTED_DATASETS) - set(reports))
    if missing:
        print(json.dumps({"status": "INCOMPLETE", "missing_datasets": missing}, indent=2))
        return 2
    rows: dict[str, dict[str, float]] = {method: {} for method in EXPECTED_METHODS}
    for dataset, path in reports.items():
        if not path.is_file():
            print(f"Missing report file: {path}")
            return 2
        payload = json.loads(path.read_text(encoding="utf-8"))
        observed = {str(row.get("method")): row for row in payload.get("datasets", {}).get(dataset, {}).get("rows", [])}
        for method in EXPECTED_METHODS:
            row = observed.get(method)
            if row is None or int(row.get("seed", -1)) != 42:
                print(f"Missing seed-42 {method} row in {path}")
                return 2
            if row.get("metric_key") != EXPECTED_METRICS[method]:
                print(f"Metric contract differs for {dataset}/{method}: {row.get('metric_key')}")
                return 2
            value = row.get("checkpoint_metrics", {}).get("25", {}).get(str(row.get("metric_key", "")))
            if value is None:
                print(f"Missing round-25 primary metric for {dataset}/{method} in {path}")
                return 2
            rows[method][dataset] = float(value)
    summary = {
        "result_status": manifest.get("result_status"),
        "suite_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "aggregate": "unweighted mean of one seed-42 round-25 primary metric per dataset",
        "aggregate_provenance": "COMMON_PRACTICE_ASSUMPTION; not the manuscript's unreported multi-run reduction rule",
        "datasets": list(EXPECTED_DATASETS),
        "report_sha256": {dataset: hashlib.sha256(reports[dataset].read_bytes()).hexdigest() for dataset in EXPECTED_DATASETS},
        "methods": {method: {"mean_accuracy_percent": 100.0 * sum(rows[method].values()) / len(EXPECTED_DATASETS), "per_dataset_accuracy_percent": {dataset: 100.0 * rows[method][dataset] for dataset in EXPECTED_DATASETS}} for method in EXPECTED_METHODS},
    }
    if args.output.exists():
        print(f"Refusing to overwrite existing summary: {args.output}")
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
