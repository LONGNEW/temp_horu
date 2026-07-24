#!/usr/bin/env python3
"""Create a content-hash inventory for a supplied paper-main data snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED_PATHS = {
    "uci_har": "data/tiers/on_device_hdc/uci_har/UCI HAR Dataset",
    "isolet_raw": "data/raw/isolet",
    "femnist": "data/tiers/standard_pfl/femnist",
    "wisdm": "data/tiers/on_device_hdc/wisdm",
    "ninapro_db1": "data/tiers/on_device_hdc/ninapro_db1",
}


def digest_tree(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        relative = item.relative_to(path).as_posix().encode("utf-8")
        file_digest = hashlib.sha256(item.read_bytes()).hexdigest().encode("ascii")
        digest.update(relative + b"\0" + file_digest + b"\n")
        file_count += 1
        total_bytes += item.stat().st_size
    return digest.hexdigest(), file_count, total_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-id", required=True, help="immutable repository/archive identifier supplied by the data owner")
    args = parser.parse_args()

    rows: dict[str, dict[str, object]] = {}
    missing: list[str] = []
    for dataset, relative_path in REQUIRED_PATHS.items():
        path = args.source_root / relative_path
        if not path.is_dir():
            missing.append(f"{dataset}: {path}")
            continue
        tree_hash, file_count, total_bytes = digest_tree(path)
        rows[dataset] = {
            "relative_path": relative_path,
            "sha256_tree": tree_hash,
            "file_count": file_count,
            "total_bytes": total_bytes,
        }
    if missing:
        print("Inventory not written; required paths are missing:")
        for item in missing:
            print(f"- {item}")
        return 2
    payload = {
        "schema_version": 1,
        "source_id": args.source_id,
        "source_root": str(args.source_root.resolve()),
        "datasets": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote data inventory: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
