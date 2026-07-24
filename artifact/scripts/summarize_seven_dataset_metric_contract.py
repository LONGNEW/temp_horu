#!/usr/bin/env python3
"""Combine the verified six-dataset suite with PAMAP2 under the HD metric contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPECTED_METRICS = {
    "horu_hd": "mean_personalized_accuracy",
    "hyperfeel": "mean_personalized_accuracy",
    "fedhdc": "global_test_accuracy",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--six-suite", type=Path, required=True)
    parser.add_argument("--pamap-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    six = json.loads(args.six_suite.read_text(encoding="utf-8"))
    pamap = json.loads(args.pamap_report.read_text(encoding="utf-8"))
    pamap_summary = pamap["datasets"]["pamap2"]["summary"]
    records: dict[str, dict[str, float]] = {}
    for method, metric_key in EXPECTED_METRICS.items():
        six_values = six["methods"][method]["per_dataset_accuracy_percent"]
        value = pamap_summary[method]["per_round"]["25"]["mean"]
        if pamap_summary[method].get("metric_key") != metric_key or not math.isfinite(float(value)):
            raise ValueError(f"PAMAP2 {method} does not satisfy {metric_key}")
        values = {str(dataset): float(score) for dataset, score in six_values.items()}
        values["pamap2"] = float(value) * 100.0
        records[method] = {
            "mean_accuracy_percent": sum(values.values()) / len(values),
            "per_dataset_accuracy_percent": values,
        }
    output = {
        "result_status": "CUDA_RECONSTRUCTION_SCREENING_ONLY",
        "aggregate": "unweighted mean of one seed-42 R25 method-specific primary metric per dataset",
        "aggregate_provenance": "COMMON_PRACTICE_ASSUMPTION; not the manuscript's unreported multi-run reduction rule",
        "datasets": [*six["datasets"], "pamap2"],
        "metric_contract": EXPECTED_METRICS,
        "methods": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"SEVEN-DATASET METRIC CONTRACT SUMMARY WRITTEN: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
