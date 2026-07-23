#!/usr/bin/env python3
"""Acquire public ISOLET inputs for the seed-42 prototype profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "artifact" / "manifests" / "prototype_isolet_seed42_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    data = manifest["data"]
    destination = args.source_root / data["source_root_relative"]
    if destination.exists():
        print(f"Refusing to alter existing extracted ISOLET data: {destination}")
        return 2
    args.download_dir.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=False)
    archive_records: dict[str, dict[str, str]] = {}
    try:
        for role, spec in data["source_archives"].items():
            archive = args.download_dir / spec["archive_filename"]
            if archive.exists() and not args.reuse_existing:
                raise ValueError(f"Refusing to overwrite archive: {archive}; pass --reuse-existing")
            if not archive.exists():
                temporary = archive.with_suffix(archive.suffix + ".partial")
                if temporary.exists():
                    raise ValueError(f"Refusing to overwrite partial download: {temporary}")
                print(f"Downloading {spec['url']}")
                urllib.request.urlretrieve(spec["url"], temporary)
                temporary.replace(archive)
            extracted = destination / spec["extracted_filename"]
            with extracted.open("wb") as handle:
                completed = subprocess.run(["uncompress", "-c", str(archive)], stdout=handle, stderr=subprocess.PIPE)
            if completed.returncode != 0 or not extracted.is_file() or extracted.stat().st_size == 0:
                raise ValueError(f"Failed to decompress ISOLET archive: {archive}")
            archive_records[str(role)] = {
                "archive": str(archive),
                "archive_sha256": sha256(archive),
                "extracted": str(extracted),
                "extracted_sha256": sha256(extracted),
            }
    except (OSError, ValueError, urllib.error.URLError) as error:
        print(f"ISOLET acquisition failed: {error}", file=sys.stderr)
        return 2
    record = {
        "result_status": "PROTOTYPE_INPUT_ONLY",
        "manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "manifest_sha256": sha256(MANIFEST_PATH),
        "extracted_root": str(destination),
        "archives": archive_records,
    }
    record_path = args.source_root / "isolet_prototype_input.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
