#!/usr/bin/env python3
"""Validate artifact inputs before any HoRU experiment is launched."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
import os
from pathlib import Path

from run_uci_har_prototype import load_acquisition_record


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = REPO_ROOT / "artifact" / "manifests"
SMOKE_REQUIREMENTS = REPO_ROOT / "artifact" / "requirements-smoke.txt"


def load_manifest(profile: str) -> dict:
    filename = {
        "smoke": "smoke_synthetic_v1.json",
        "uci-har-prototype": "prototype_uci_har_v1.json",
        "paper-main": "paper_main_v1.json",
    }[profile]
    with (MANIFEST_DIR / filename).open(encoding="utf-8") as handle:
        return json.load(handle)


def check_smoke(manifest: dict) -> list[str]:
    failures: list[str] = []
    if manifest.get("result_status") != "SMOKE_TEST_ONLY":
        failures.append("smoke manifest must be labeled SMOKE_TEST_ONLY")
    for relative_path in manifest.get("required_inputs", []):
        if not (REPO_ROOT / relative_path).is_file():
            failures.append(f"missing required smoke input: {relative_path}")
    for line in SMOKE_REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        distribution, expected_version = line.split("==", maxsplit=1)
        try:
            actual_version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"missing smoke dependency: {distribution}=={expected_version}")
            continue
        if actual_version != expected_version:
            failures.append(
                f"smoke dependency version mismatch: {distribution} expected {expected_version}, got {actual_version}"
            )
    return failures


def git_object_exists(ref: str, path: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{ref}:{path}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def check_paper_main(manifest: dict) -> list[str]:
    failures: list[str] = []
    paper = manifest.get("paper", {})
    if not git_object_exists(str(paper.get("manuscript_git_ref", "")), str(paper.get("manuscript_path", ""))):
        failures.append("checked manuscript Git object is unavailable")
    if manifest.get("result_status") != "PAPER_REPRODUCTION_READY":
        failures.append(f"manifest status is {manifest.get('result_status')}, not PAPER_REPRODUCTION_READY")
    for blocker in manifest.get("paper_reproduction_blockers", []):
        failures.append(f"[{blocker.get('id', 'unknown')}] {blocker.get('required_resolution', blocker.get('detail', 'unresolved'))}")
    data_contract = manifest.get("data_contract", {})
    source_root = Path(os.environ.get("HORU_SOURCE_DATA_ROOT", str(data_contract.get("default_source_root", ""))))
    inventory_path = Path(str(data_contract.get("inventory_path", "")))
    if manifest.get("result_status") == "PAPER_REPRODUCTION_READY":
        if not source_root.is_dir():
            failures.append(f"paper source root is missing: {source_root}")
        if not inventory_path.is_file():
            failures.append(f"paper data inventory is missing: {inventory_path}")
        seed_protocol = manifest.get("seed_protocol", {})
        if not isinstance(seed_protocol.get("seeds"), list) or not seed_protocol["seeds"]:
            failures.append("paper seed protocol is missing a non-empty seed list")
        if seed_protocol.get("aggregation") in {None, "", "UNKNOWN_OR_UNVERIFIED"}:
            failures.append("paper seed protocol is missing an aggregation rule")
    return failures


def check_uci_har_prototype(manifest: dict, *, source_root: Path | None, archive: Path | None) -> list[str]:
    failures: list[str] = []
    if manifest.get("result_status") != "PROTOTYPE_ONLY":
        failures.append("UCI-HAR prototype manifest must be labeled PROTOTYPE_ONLY")
    if source_root is None:
        failures.append("--source-root is required for uci-har-prototype")
    if archive is None:
        failures.append("--archive is required for uci-har-prototype")
    if failures:
        return failures
    assert source_root is not None and archive is not None
    if not archive.is_file():
        failures.append(f"missing UCI-HAR archive: {archive}")
        return failures
    expected_root = source_root / str(manifest["data"]["source_root_relative"])
    required = [expected_root / "train" / "X_train.txt", expected_root / "test" / "X_test.txt"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        failures.append("missing extracted UCI-HAR input files: " + ", ".join(missing))
        return failures
    try:
        load_acquisition_record(source_root, expected_root, archive)
    except ValueError as error:
        failures.append(f"input provenance validation failed: {error}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "uci-har-prototype", "paper-main"], required=True)
    parser.add_argument("--source-root", type=Path, default=None, help="required for uci-har-prototype")
    parser.add_argument("--archive", type=Path, default=None, help="required for uci-har-prototype")
    args = parser.parse_args()
    manifest = load_manifest(args.profile)
    if args.profile == "smoke":
        failures = check_smoke(manifest)
    elif args.profile == "uci-har-prototype":
        failures = check_uci_har_prototype(manifest, source_root=args.source_root, archive=args.archive)
    else:
        failures = check_paper_main(manifest)
    if failures:
        print(f"PREFLIGHT FAILED: {args.profile}")
        for failure in failures:
            print(f"- {failure}")
        return 2
    print(f"PREFLIGHT PASSED: {args.profile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
