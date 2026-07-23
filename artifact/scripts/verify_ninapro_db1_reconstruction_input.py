#!/usr/bin/env python3
"""Verify a NinaPro DB1 reconstruction archive and MAT-file inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    record_path = args.source_root / "ninapro_db1_reconstruction_input.json"
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
    subjects = [int(value) for value in manifest["source"]["subjects"]]
    expected_members = {f"S{subject}_A1_E{exercise}.mat" for subject in subjects for exercise in manifest["source"]["expected_exercises"]}
    archives = record.get("archives", {})
    if not isinstance(archives, dict) or set(archives) != {f"s{subject}" for subject in subjects}:
        failures.append("archive inventory lacks one or more official subject archives")
        archives = {}
    for subject in subjects:
        entry = archives.get(f"s{subject}", {})
        if not isinstance(entry, dict):
            failures.append(f"invalid archive entry for subject {subject}")
            continue
        archive = Path(str(entry.get("archive", "")))
        if not archive.is_file() or entry.get("sha256") != sha256(archive):
            failures.append(f"archive hash differs for subject {subject}")
            continue
        try:
            with zipfile.ZipFile(archive) as handle:
                if set(handle.namelist()) != {f"S{subject}_A1_E{exercise}.mat" for exercise in manifest["source"]["expected_exercises"]}:
                    failures.append(f"archive members differ for subject {subject}")
        except zipfile.BadZipFile:
            failures.append(f"invalid ZIP archive for subject {subject}")
    data_root = args.source_root / "data" / "tiers" / "on_device_hdc" / "ninapro_db1"
    observed = {path.name: sha256(path) for path in sorted(data_root.glob("S*_A1_E*.mat"))}
    if set(observed) != expected_members:
        failures.append("extracted MAT file set differs from official subject/exercise contract")
    if observed != record.get("mat_files"):
        failures.append("extracted MAT hash inventory differs from input record")
    if failures:
        print("NINAPRO DB1 INPUT VERIFICATION FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 2
    print("NINAPRO DB1 RECONSTRUCTION INPUT VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
