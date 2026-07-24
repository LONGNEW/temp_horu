#!/usr/bin/env python3
"""Verify and summarize historical Git result candidates without treating them as paper evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = REPO_ROOT / "artifact" / "manifests" / "historical_result_candidates.json"


def git_show(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=REPO_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional JSON destination for this audit")
    args = parser.parse_args()
    candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))["candidates"]
    rows: list[dict[str, object]] = []
    invalid = False
    for candidate in candidates:
        try:
            contents = git_show(candidate["git_commit"], candidate["path"])
        except subprocess.CalledProcessError:
            rows.append({**candidate, "verified": False, "error": "Git object is unavailable"})
            invalid = True
            continue
        digest = hashlib.sha256(contents).hexdigest()
        payload = json.loads(contents)
        row = {
            **candidate,
            "verified": digest == candidate["sha256"],
            "observed_sha256": digest,
            "seeds": payload.get("seeds", payload.get("run_config", {}).get("seed")),
            "datasets": payload.get("run_config", {}).get("datasets"),
            "methods": payload.get("run_config", {}).get("methods", [payload.get("run_config", {}).get("method_key")]),
        }
        invalid = invalid or not bool(row["verified"])
        rows.append(row)
    result = {"schema_version": 1, "paper_evidence": False, "candidates": rows}
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote historical-result audit: {args.output}")
    else:
        print(rendered, end="")
    return 2 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
