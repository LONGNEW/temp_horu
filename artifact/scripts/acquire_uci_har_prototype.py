#!/usr/bin/env python3
"""Acquire and verify the public UCI-HAR input used by the prototype profile.

This script deliberately does not assert that the downloaded archive is the
unavailable paper-run snapshot.  It records the exact URL and SHA-256 so a
prototype execution can be independently inspected or repeated.
"""

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
MANIFEST_PATH = REPO_ROOT / "artifact" / "manifests" / "prototype_uci_har_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_members(archive: zipfile.ZipFile) -> list[str]:
    names = set(archive.namelist())
    root = "UCI HAR Dataset"
    required = [
        f"{root}/train/X_train.txt",
        f"{root}/train/y_train.txt",
        f"{root}/train/subject_train.txt",
        f"{root}/test/X_test.txt",
        f"{root}/test/y_test.txt",
        f"{root}/test/subject_test.txt",
    ]
    return [member for member in required if member not in names]


def validate_archive(path: Path) -> str | None:
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Not a ZIP archive: {path}")
    with zipfile.ZipFile(path) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError(f"Corrupt ZIP member: {corrupt}")
        missing = required_members(archive)
        if not missing:
            return None
        nested_name = "UCI HAR Dataset.zip"
        if nested_name not in archive.namelist():
            raise ValueError("Archive lacks required UCI-HAR members: " + ", ".join(missing))
        with zipfile.ZipFile(io.BytesIO(archive.read(nested_name))) as nested:
            corrupt = nested.testzip()
            if corrupt is not None:
                raise ValueError(f"Corrupt nested ZIP member: {corrupt}")
            missing = required_members(nested)
            if missing:
                raise ValueError("Nested archive lacks required UCI-HAR members: " + ", ".join(missing))
        return nested_name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--reuse-existing", action="store_true", help="validate an existing archive instead of downloading")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    source_url = manifest["data"]["source_url"]
    extract_parent = args.source_root / "data/tiers/on_device_hdc/uci_har"
    expected_root = extract_parent / "UCI HAR Dataset"
    if expected_root.exists():
        print(f"Refusing to alter existing extracted dataset: {expected_root}")
        return 2
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    if args.archive.exists() and not args.reuse_existing:
        print(f"Refusing to overwrite archive: {args.archive}. Pass --reuse-existing after independently checking it.")
        return 2
    try:
        if not args.archive.exists():
            temporary = args.archive.with_suffix(args.archive.suffix + ".partial")
            if temporary.exists():
                print(f"Refusing to overwrite partial download: {temporary}")
                return 2
            print(f"Downloading {source_url}")
            urllib.request.urlretrieve(source_url, temporary)
            temporary.replace(args.archive)
        nested_member = validate_archive(args.archive)
    except (OSError, ValueError, urllib.error.URLError, zipfile.BadZipFile) as error:
        print(f"Archive validation failed: {error}", file=sys.stderr)
        return 2
    extract_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=extract_parent, prefix="uci-har-extract-") as temp_dir:
        temporary_root = Path(temp_dir)
        with zipfile.ZipFile(args.archive) as archive:
            if nested_member is None:
                archive.extractall(temporary_root)
            else:
                with zipfile.ZipFile(io.BytesIO(archive.read(nested_member))) as nested:
                    nested.extractall(temporary_root)
        extracted = temporary_root / "UCI HAR Dataset"
        if not extracted.is_dir():
            print("Extraction produced no UCI HAR Dataset directory", file=sys.stderr)
            return 2
        shutil.move(str(extracted), expected_root)
    record = {
        "result_status": "PROTOTYPE_INPUT_ONLY",
        "manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "manifest_sha256": sha256(MANIFEST_PATH),
        "source_url": source_url,
        "archive": str(args.archive),
        "archive_sha256": sha256(args.archive),
        "nested_archive_member": nested_member,
        "extracted_root": str(expected_root),
    }
    record_path = args.source_root / "uci_har_prototype_input.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
