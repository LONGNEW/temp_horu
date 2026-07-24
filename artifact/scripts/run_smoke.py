#!/usr/bin/env python3
"""Run the bounded synthetic artifact smoke test outside the repository."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import hashlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "artifact" / "manifests" / "smoke_synthetic_v1.json"


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="development-only override; generated output will be labeled with the dirty tree state",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory for generated smoke outputs (defaults under LONGNEW_DATA_ROOT)",
    )
    parser.add_argument(
        "--allow-existing-report",
        action="store_true",
        help="allow the runner to reuse an existing report; provenance will mark it as reused",
    )
    args = parser.parse_args()
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    dirty_paths = git_output("status", "--porcelain")
    if dirty_paths and not args.allow_dirty:
        print("Refusing to generate an artifact report from a dirty worktree. Commit changes first, or use --allow-dirty for development only.")
        return 2
    data_root = Path(os.environ.get("LONGNEW_DATA_ROOT", "/home/longnew/data"))
    output_dir = args.output_dir or (data_root / "projects" / "horu" / "artifact-smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "synthetic_horu.json"
    if report_path.exists() and not args.allow_existing_report:
        print(f"Refusing to reuse existing report: {report_path}. Choose a new --output-dir or pass --allow-existing-report explicitly.")
        return 2
    command = [*manifest["command"], "--json-out", str(report_path), "--md-out", str(output_dir / "synthetic_horu.md")]
    print("Run Manifest")
    print("Command:")
    print(" ".join(command))
    print("Result status: SMOKE_TEST_ONLY")
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        return completed.returncode
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["artifact_provenance"] = {
        "result_status": manifest["result_status"],
        "manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "manifest_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_dirty": bool(dirty_paths),
        "git_dirty_paths": dirty_paths.splitlines(),
        "execution_mode": "reused" if args.allow_existing_report else "fresh",
        "python_version": sys.version,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
