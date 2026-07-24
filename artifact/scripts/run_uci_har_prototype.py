#!/usr/bin/env python3
"""Run a provenance-labeled UCI-HAR HoRU prototype outside the repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "artifact" / "manifests" / "prototype_uci_har_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def load_acquisition_record(source_root: Path, expected_root: Path, archive: Path) -> tuple[Path, dict[str, object]]:
    record_path = source_root / "uci_har_prototype_input.json"
    if not record_path.is_file():
        raise ValueError(f"Missing acquisition record: {record_path}")
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid acquisition record JSON: {record_path}: {error}") from error
    if record.get("result_status") != "PROTOTYPE_INPUT_ONLY":
        raise ValueError("Acquisition record has an unexpected result status")
    if record.get("manifest_sha256") != sha256(MANIFEST_PATH):
        raise ValueError("Acquisition record was made from a different prototype manifest")
    if record.get("archive_sha256") != sha256(archive):
        raise ValueError("Acquisition record archive hash differs from --archive")
    recorded_root = Path(str(record.get("extracted_root", ""))).resolve()
    if recorded_root != expected_root.resolve():
        raise ValueError("Acquisition record extracted_root differs from the supplied --source-root")
    return record_path, record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    dirty_paths = git_output("status", "--porcelain")
    if dirty_paths and not args.allow_dirty:
        print("Refusing to run from a dirty worktree. Commit tracked artifact changes first.")
        return 2
    if not args.archive.is_file():
        print(f"Missing verified UCI-HAR archive: {args.archive}")
        return 2
    expected_root = args.source_root / manifest["data"]["source_root_relative"]
    required = [expected_root / "train" / "X_train.txt", expected_root / "test" / "X_test.txt"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print("Missing extracted UCI-HAR input files: " + ", ".join(missing))
        return 2
    try:
        acquisition_record_path, acquisition_record = load_acquisition_record(
            args.source_root, expected_root, args.archive
        )
    except ValueError as error:
        print(f"Input provenance validation failed: {error}")
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_out = args.output_dir / "uci_har_horu.json"
    md_out = args.output_dir / "uci_har_horu.md"
    if json_out.exists():
        print(f"Refusing to overwrite existing result: {json_out}")
        return 2
    protocol = manifest["protocol"]
    command = [
        sys.executable,
        "run_hd_checkpoint_comparison.py",
        "--datasets", "uci_har",
        "--methods", *protocol["methods"],
        "--device", protocol["device"],
        *( ["--deterministic-algorithms"] if protocol.get("deterministic_algorithms", False) else [] ),
        "--torch-num-threads", str(protocol.get("torch_num_threads", 1)),
        "--seeds", str(protocol["seed"]),
        "--round-checkpoints", *(str(value) for value in protocol["round_checkpoints"]),
        "--local-epochs", str(protocol["local_epochs"]),
        "--batch-size", str(protocol["batch_size"]),
        "--client-participation", str(protocol["client_participation"]),
        "--hd-dim", str(protocol["hd_dim"]),
        "--hd-lr", str(protocol["hd_lr"]),
        "--subspace-shared-rank", str(protocol["subspace_shared_rank"]),
        "--subspace-intersection-rank", str(protocol["subspace_intersection_rank"]),
        "--subspace-personal-rank", str(protocol["subspace_personal_rank"]),
        "--json-out", str(json_out),
        "--md-out", str(md_out),
    ]
    print("Run Manifest")
    print("Command:")
    print(" ".join(command))
    print(f"Config: {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    print("Assumed/prototype-only fields: seed=13, device=cpu, official UCI download instead of unavailable repository data snapshot")
    print("Result status: PROTOTYPE_ONLY")
    env = os.environ.copy()
    env["HORU_SOURCE_DATA_ROOT"] = str(args.source_root)
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
    if completed.returncode != 0:
        return completed.returncode
    report = json.loads(json_out.read_text(encoding="utf-8"))
    report["artifact_provenance"] = {
        "result_status": manifest["result_status"],
        "manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "manifest_sha256": sha256(MANIFEST_PATH),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_dirty": bool(dirty_paths),
        "execution_mode": "fresh",
        "source_archive": str(args.archive),
        "source_archive_sha256": sha256(args.archive),
        "source_root": str(expected_root),
        "source_url": manifest["data"]["source_url"],
        "acquisition_record": str(acquisition_record_path),
        "acquisition_record_sha256": sha256(acquisition_record_path),
        "acquisition_nested_archive_member": acquisition_record.get("nested_archive_member"),
        "limitations": manifest["limitations"],
    }
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote provenance-labeled prototype report: {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
