#!/usr/bin/env python3
"""Verify a complete CUDA reconstruction suite against its immutable manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


EXPECTED_DATASETS = ("uci_har", "isolet_raw", "femnist", "wisdm", "synthetic", "ninapro_db1")
EXPECTED_METHODS = ("horu_hd", "hyperfeel", "fedhdc")
EXPECTED_METRICS = {
    "horu_hd": "mean_personalized_accuracy",
    "hyperfeel": "mean_personalized_accuracy",
    "fedhdc": "global_test_accuracy",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--suite-output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    summary_path = args.suite_output / "summary.json"
    failures: list[str] = []
    if not summary_path.is_file():
        failures.append(f"missing suite summary: {summary_path}")
        summary: dict[str, object] = {}
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    protocol = manifest.get("protocol", {})
    if tuple(protocol.get("datasets", [])) != EXPECTED_DATASETS:
        failures.append("manifest dataset contract is not the complete six-dataset suite")
    if set(protocol.get("methods", [])) != set(EXPECTED_METHODS):
        failures.append("manifest method contract differs from the complete suite")
    if protocol.get("seed") != 42 or 25 not in protocol.get("round_checkpoints", []):
        failures.append("manifest does not require seed 42 and round 25")
    if summary.get("result_status") != manifest.get("result_status"):
        failures.append("summary result status differs from the manifest")
    if summary.get("suite_manifest_sha256") != sha256(args.manifest):
        failures.append("summary manifest hash differs from the manifest supplied for verification")
    if tuple(summary.get("datasets", [])) != EXPECTED_DATASETS:
        failures.append("summary dataset order differs from the complete suite")
    report_hashes = summary.get("report_sha256")
    if not isinstance(report_hashes, dict):
        failures.append("summary lacks report hash inventory")
        report_hashes = {}
    values: dict[str, dict[str, float]] = {method: {} for method in EXPECTED_METHODS}
    for dataset in EXPECTED_DATASETS:
        report_path = args.suite_output / dataset / f"{dataset}.json"
        if not report_path.is_file():
            failures.append(f"missing report: {report_path}")
            continue
        if report_hashes.get(dataset) != sha256(report_path):
            failures.append(f"report hash differs from summary inventory: {dataset}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("device") != protocol.get("device"):
            failures.append(f"device contract differs in {dataset}")
        if report.get("seeds") != [42] or report.get("round_checkpoints") != protocol.get("round_checkpoints"):
            failures.append(f"seed or checkpoint contract differs in {dataset}")
        config = report.get("run_config", {})
        for key in (
            "local_epochs",
            "batch_size",
            "client_participation",
            "hd_dim",
            "hd_lr",
            "subspace_shared_rank",
            "subspace_intersection_rank",
            "subspace_personal_rank",
            "torch_num_threads",
            "deterministic_algorithms",
        ):
            if config.get(key) != protocol.get(key):
                failures.append(f"run config {key} differs in {dataset}")
        rows = {str(row.get("method")): row for row in report.get("datasets", {}).get(dataset, {}).get("rows", [])}
        for method in EXPECTED_METHODS:
            row = rows.get(method)
            if row is None:
                failures.append(f"missing {method} result in {dataset}")
                continue
            if row.get("metric_key") != EXPECTED_METRICS[method]:
                failures.append(f"metric contract differs for {dataset}/{method}")
                continue
            metric = row.get("checkpoint_metrics", {}).get("25", {}).get(str(row.get("metric_key", "")))
            if metric is None:
                failures.append(f"missing round-25 metric for {dataset}/{method}")
                continue
            values[method][dataset] = 100.0 * float(metric)
    methods = summary.get("methods", {})
    for method in EXPECTED_METHODS:
        expected_values = values[method]
        actual = methods.get(method, {}) if isinstance(methods, dict) else {}
        if not isinstance(actual, dict):
            failures.append(f"aggregate record is invalid for {method}")
            continue
        expected_per_dataset = {
            dataset: expected_values[dataset]
            for dataset in EXPECTED_DATASETS
            if dataset in expected_values
        }
        if actual.get("per_dataset_accuracy_percent") != expected_per_dataset:
            failures.append(f"per-dataset aggregate differs for {method}")
        if len(expected_values) == len(EXPECTED_DATASETS):
            expected_mean = sum(expected_values.values()) / len(EXPECTED_DATASETS)
            if not math.isclose(float(actual.get("mean_accuracy_percent", float("nan"))), expected_mean, rel_tol=0.0, abs_tol=1e-12):
                failures.append(f"mean aggregate differs for {method}")
    if failures:
        print("RECONSTRUCTION SUITE VERIFICATION FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 2
    print(f"RECONSTRUCTION SUITE VERIFIED: {manifest['result_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
