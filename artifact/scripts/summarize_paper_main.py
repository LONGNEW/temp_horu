#!/usr/bin/env python3
"""Merge paper-main component reports and enforce their manifest contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "artifact" / "manifests" / "paper_main_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    protocol = manifest["protocol"]
    reports = [json.loads((args.input_dir / name).read_text(encoding="utf-8")) for name in ("hd.json", "nn_vector.json", "nn_femnist.json")]
    rows = [row for report in reports for dataset in report["datasets"].values() for row in dataset.get("rows", [])]
    expected = {
        (dataset, method)
        for dataset in protocol["datasets"]
        for method in (["horu_hd", "fedhdc", "hyperfeel"] if dataset != "femnist" else ["horu_hd", "fedhdc", "hyperfeel", "fedavg_cnn", "dfl_cnn"])
    }
    # Vector NN methods apply to each non-FEMNIST paper benchmark.
    expected |= {(dataset, method) for dataset in protocol["datasets"] if dataset != "femnist" for method in ("fedavg_mlp", "dfl_mlp")}
    observed = {(str(row["dataset"]), str(row["method"])) for row in rows}
    missing = sorted(expected - observed)
    if missing:
        raise RuntimeError(f"paper-main component reports are incomplete: {missing}")
    by_method: dict[str, list[float]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(float(row["primary_value"]))
    aggregate = {method: sum(values) / len(values) for method, values in by_method.items()}
    result = {
        "result_status": "VALID_EXPERIMENT_CANDIDATE",
        "paper_reproduction_claim": False,
        "manifest": str(MANIFEST.relative_to(REPO_ROOT)),
        "rows": rows,
        "aggregate_means": aggregate,
        "paper_claimed_means_percent": manifest["paper"]["claimed_summary"],
    }
    path = args.input_dir / "paper_main_summary.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote candidate paper-main summary: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
