#!/usr/bin/env python3
"""Extract and hash the nested official WISDM archive for reconstruction screening."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "artifact" / "manifests" / "reconstruction_wisdm_seed42_v1.json"

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-archive", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reuse-existing", action="store_true", help="Validate an existing outer archive instead of downloading it again.")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    data = manifest["data"]
    args.outer_archive.parent.mkdir(parents=True, exist_ok=True)
    if args.outer_archive.exists() and not args.reuse_existing:
        print(f"Refusing to overwrite outer archive: {args.outer_archive}; pass --reuse-existing after independent inspection.", file=sys.stderr)
        return 2
    destination = args.source_root / data["source_root_relative"] / data["outer_archive_member"]
    record_path = args.source_root / "wisdm_reconstruction_input.json"
    if destination.exists() or record_path.exists():
        print(f"Refusing to alter existing WISDM reconstruction input: {destination}", file=sys.stderr)
        return 2
    destination.parent.mkdir(parents=True, exist_ok=False)
    try:
        if not args.outer_archive.exists():
            partial = args.outer_archive.with_suffix(args.outer_archive.suffix + ".partial")
            if partial.exists():
                raise ValueError(f"Refusing to overwrite partial download: {partial}")
            print(f"Downloading {data['outer_archive_url']}")
            urllib.request.urlretrieve(data["outer_archive_url"], partial)
            partial.replace(args.outer_archive)
        with zipfile.ZipFile(args.outer_archive) as outer:
            bad_member = outer.testzip()
            if bad_member is not None:
                raise ValueError(f"Outer ZIP CRC failure: {bad_member}")
            member = data["outer_archive_member"]
            if member not in outer.namelist():
                raise ValueError(f"Outer ZIP lacks required member: {member}")
            with outer.open(member) as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
        with zipfile.ZipFile(destination) as inner:
            bad_member = inner.testzip()
            if bad_member is not None:
                raise ValueError(f"Nested WISDM ZIP CRC failure: {bad_member}")
            prefix = "wisdm-dataset/raw/phone/accel/"
            members = [name for name in inner.namelist() if name.startswith(prefix) and name.endswith(".txt")]
            if len(members) != int(data["clients"]):
                raise ValueError(f"Expected {data['clients']} phone-accelerometer users, found {len(members)}")
    except (OSError, ValueError, urllib.error.URLError, zipfile.BadZipFile) as error:
        print(f"WISDM acquisition failed: {error}", file=sys.stderr)
        return 2
    record = {
        "result_status": "RECONSTRUCTION_INPUT_ONLY",
        "manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "manifest_sha256": sha256(MANIFEST_PATH),
        "outer_archive": str(args.outer_archive),
        "outer_archive_sha256": sha256(args.outer_archive),
        "nested_archive_member": data["outer_archive_member"],
        "nested_archive": str(destination),
        "nested_archive_sha256": sha256(destination),
        "phone_accelerometer_users": len(members),
    }
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
