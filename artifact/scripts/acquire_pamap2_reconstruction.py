#!/usr/bin/env python3
"""Acquire and hash the official PAMAP2 Protocol input for screening."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "artifact" / "manifests" / "reconstruction_pamap2_seed42_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_dataset_root(root: Path, name: str) -> Path:
    candidates = [path for path in root.rglob(name) if path.is_dir() and (path / "Protocol").is_dir()]
    if len(candidates) != 1:
        raise ValueError(f"Expected one {name}/Protocol directory, found {len(candidates)}")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    data = manifest["data"]
    target_parent = args.source_root / data["source_root_relative"]
    target = target_parent / data["dataset_directory"]
    record_path = args.source_root / "pamap2_reconstruction_input.json"
    if target.exists() or record_path.exists():
        print(f"Refusing to alter existing PAMAP2 input: {target}", file=sys.stderr)
        return 2
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    if args.archive.exists() and not args.reuse_existing:
        print(f"Refusing to overwrite archive: {args.archive}; pass --reuse-existing after inspection.", file=sys.stderr)
        return 2
    try:
        args.source_root.parent.mkdir(parents=True, exist_ok=True)
        if not args.archive.exists():
            partial = args.archive.with_suffix(args.archive.suffix + ".partial")
            if partial.exists():
                raise ValueError(f"Refusing to overwrite partial download: {partial}")
            print(f"Downloading {data['archive_url']}", flush=True)
            urllib.request.urlretrieve(data["archive_url"], partial)
            partial.replace(args.archive)
        if not zipfile.is_zipfile(args.archive):
            raise ValueError(f"Not a ZIP archive: {args.archive}")
        with tempfile.TemporaryDirectory(dir=args.source_root.parent, prefix="pamap2-extract-") as temporary:
            temporary_root = Path(temporary)
            with zipfile.ZipFile(args.archive) as outer:
                bad_member = outer.testzip()
                if bad_member is not None:
                    raise ValueError(f"Archive CRC failure: {bad_member}")
                nested = "PAMAP2_Dataset.zip"
                if nested in outer.namelist():
                    with zipfile.ZipFile(io.BytesIO(outer.read(nested))) as inner:
                        bad_member = inner.testzip()
                        if bad_member is not None:
                            raise ValueError(f"Nested archive CRC failure: {bad_member}")
                        inner.extractall(temporary_root)
                else:
                    outer.extractall(temporary_root)
            extracted = find_dataset_root(temporary_root, str(data["dataset_directory"]))
            protocol = extracted / data["protocol_directory"]
            files = sorted(protocol.glob("*.dat"))
            if len(files) != int(data["subjects_available"]):
                raise ValueError(f"Expected {data['subjects_available']} protocol files, found {len(files)}")
            for path in files:
                first_row = next((line for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()), "")
                if len(first_row.split()) != int(data["columns_per_row"]):
                    raise ValueError(f"Unexpected column count in {path.name}")
            target_parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(extracted), target)
    except (OSError, ValueError, urllib.error.URLError, zipfile.BadZipFile) as error:
        print(f"PAMAP2 acquisition failed: {error}", file=sys.stderr)
        return 2
    protocol = target / data["protocol_directory"]
    files = sorted(protocol.glob("*.dat"))
    record = {
        "result_status": manifest["result_status"],
        "manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "manifest_sha256": sha256(MANIFEST_PATH),
        "archive": str(args.archive),
        "archive_sha256": sha256(args.archive),
        "dataset_root": str(target),
        "protocol_files_sha256": {path.name: sha256(path) for path in files},
    }
    args.source_root.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
