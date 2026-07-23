#!/usr/bin/env python3
"""Acquire and verify official NinaPro DB1 subject archives for reconstruction screening."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "artifact" / "manifests" / "reconstruction_ninapro_db1_seed42_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_members(subject: int, exercises: list[int]) -> set[str]:
    return {f"S{subject}_A1_E{exercise}.mat" for exercise in exercises}


def validate_archive(path: Path, subject: int, exercises: list[int]) -> None:
    if not zipfile.is_zipfile(path):
        raise ValueError(f"not a ZIP archive: {path}")
    with zipfile.ZipFile(path) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError(f"corrupt ZIP member in {path}: {corrupt}")
        names = set(archive.namelist())
        expected = required_members(subject, exercises)
        if names != expected:
            raise ValueError(f"unexpected members in {path}: expected {sorted(expected)}, got {sorted(names)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--reuse-existing", action="store_true", help="Validate existing archives instead of downloading them again.")
    args = parser.parse_args()
    if args.source_root.exists():
        print(f"Refusing to alter existing source root: {args.source_root}", file=sys.stderr)
        return 2
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    source = manifest["source"]
    subjects = [int(value) for value in source["subjects"]]
    exercises = [int(value) for value in source["expected_exercises"]]
    args.download_dir.mkdir(parents=True, exist_ok=True)
    archives: dict[str, dict[str, str]] = {}
    try:
        for subject in subjects:
            archive = args.download_dir / f"s{subject}.zip"
            url = str(source["archive_url_template"]).format(subject=subject)
            if archive.exists() and not args.reuse_existing:
                raise RuntimeError(f"refusing to overwrite existing archive: {archive}; pass --reuse-existing after independent inspection")
            if not archive.exists():
                partial = archive.with_suffix(".zip.partial")
                if partial.exists():
                    raise RuntimeError(f"refusing to overwrite partial download: {partial}")
                print(f"Downloading {url}")
                urllib.request.urlretrieve(url, partial)
                partial.replace(archive)
            validate_archive(archive, subject, exercises)
            archives[f"s{subject}"] = {"url": url, "archive": str(archive), "sha256": sha256(archive)}
        target = args.source_root / "data" / "tiers" / "on_device_hdc" / "ninapro_db1"
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent, prefix="ninapro-db1-extract-") as temp_dir:
            temporary = Path(temp_dir)
            for subject in subjects:
                archive = args.download_dir / f"s{subject}.zip"
                with zipfile.ZipFile(archive) as handle:
                    handle.extractall(temporary)
            members = sorted(temporary.glob("S*_A1_E*.mat"))
            if len(members) != len(subjects) * len(exercises):
                raise RuntimeError("extraction did not produce the expected 81 NinaPro DB1 MAT files")
            shutil.move(str(temporary), target)
    except (OSError, RuntimeError, ValueError, urllib.error.URLError, zipfile.BadZipFile) as error:
        print(f"NinaPro DB1 acquisition failed: {error}", file=sys.stderr)
        return 2
    mat_files = sorted(target.glob("S*_A1_E*.mat"))
    record = {
        "result_status": manifest["result_status"],
        "manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "manifest_sha256": sha256(MANIFEST_PATH),
        "instructions_url": source["instructions_url"],
        "source_root": str(args.source_root),
        "archives": archives,
        "mat_files": {path.name: sha256(path) for path in mat_files},
    }
    (args.source_root / "ninapro_db1_reconstruction_input.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
