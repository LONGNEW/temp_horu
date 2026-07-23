#!/usr/bin/env python3
"""Verify a LEAF Synthetic reconstruction input inventory and its pinned source."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "artifact" / "manifests" / "reconstruction_synthetic_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    record_path = args.source_root / "synthetic_reconstruction_input.json"
    failures: list[str] = []
    if not record_path.is_file():
        failures.append(f"missing input record: {record_path}")
        record: dict[str, object] = {}
    else:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("result_status") != manifest.get("result_status"):
        failures.append("input result status differs from manifest")
    if record.get("manifest_sha256") != sha256(MANIFEST_PATH):
        failures.append("input record manifest hash differs from manifest")
    if Path(str(record.get("source_root", ""))).resolve() != args.source_root.resolve():
        failures.append("input record source root differs from verification root")
    leaf_root = args.source_root / "leaf"
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=leaf_root, text=True).strip()
        if revision != manifest["source"]["revision"]:
            failures.append("LEAF revision differs from manifest")
    except (OSError, subprocess.CalledProcessError):
        failures.append(f"pinned LEAF checkout is unavailable: {leaf_root}")
    files = record.get("generated_files", {})
    if not isinstance(files, dict):
        failures.append("input record generated-file inventory is invalid")
        files = {}
    for split in ("train", "test"):
        entry = files.get(split, {})
        if not isinstance(entry, dict):
            failures.append(f"missing {split} shard inventory")
            continue
        path = Path(str(entry.get("path", "")))
        if not path.is_file():
            failures.append(f"missing {split} shard: {path}")
            continue
        if entry.get("sha256") != sha256(path):
            failures.append(f"{split} shard hash differs from input record")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not {"users", "num_samples", "user_data"}.issubset(payload):
                failures.append(f"{split} shard lacks LEAF JSON keys")
        except json.JSONDecodeError:
            failures.append(f"{split} shard is not valid JSON")
    if failures:
        print("SYNTHETIC INPUT VERIFICATION FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 2
    print("SYNTHETIC RECONSTRUCTION INPUT VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
