#!/usr/bin/env python3
"""Verify the official PAMAP2 reconstruction-input inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


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
    record_path = args.source_root / "pamap2_reconstruction_input.json"
    if not record_path.is_file():
        print(f"Missing acquisition record: {record_path}")
        return 1
    record = json.loads(record_path.read_text(encoding="utf-8"))
    manifest = REPO_ROOT / str(record.get("manifest_path", ""))
    failures: list[str] = []
    if not manifest.is_file() or record.get("manifest_sha256") != sha256(manifest):
        failures.append("manifest hash differs")
    root = Path(str(record.get("dataset_root", "")))
    if not root.is_dir() or not str(root.resolve()).startswith(str(args.source_root.resolve())):
        failures.append("dataset root is missing or outside source root")
    for name, expected in record.get("protocol_files_sha256", {}).items():
        path = root / "Protocol" / str(name)
        if not path.is_file() or sha256(path) != expected:
            failures.append(f"protocol file hash differs: {name}")
    if failures:
        print("PAMAP2 INPUT VERIFICATION FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("PAMAP2 INPUT VERIFIED: RECONSTRUCTION_INPUT_ONLY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
