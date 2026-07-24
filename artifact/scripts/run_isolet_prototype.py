#!/usr/bin/env python3
"""Run the source-traceable ISOLET seed-42 HoRU prototype."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "artifact" / "manifests" / "prototype_isolet_seed42_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def load_acquisition_record(source_root: Path, expected_root: Path, download_dir: Path) -> tuple[Path, dict[str, object]]:
    record_path = source_root / "isolet_prototype_input.json"
    if not record_path.is_file():
        raise ValueError(f"Missing acquisition record: {record_path}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("result_status") != "PROTOTYPE_INPUT_ONLY":
        raise ValueError("Acquisition record has an unexpected result status")
    if record.get("manifest_sha256") != sha256(MANIFEST_PATH):
        raise ValueError("Acquisition record was made from a different prototype manifest")
    if Path(str(record.get("extracted_root", ""))).resolve() != expected_root.resolve():
        raise ValueError("Acquisition record extracted_root differs from --source-root")
    archives = record.get("archives")
    if not isinstance(archives, dict):
        raise ValueError("Acquisition record has no archive inventory")
    for role, values in archives.items():
        if not isinstance(values, dict):
            raise ValueError(f"Acquisition record archive entry is invalid: {role}")
        archive = Path(str(values.get("archive", "")))
        extracted = Path(str(values.get("extracted", "")))
        if archive.parent.resolve() != download_dir.resolve() or not archive.is_file() or not extracted.is_file():
            raise ValueError(f"Acquisition archive/input is unavailable: {role}")
        if values.get("archive_sha256") != sha256(archive) or values.get("extracted_sha256") != sha256(extracted):
            raise ValueError(f"Acquisition hash mismatch: {role}")
    return record_path, record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if git_output("status", "--porcelain"):
        print("Refusing to run from a dirty worktree.")
        return 2
    expected_root = args.source_root / manifest["data"]["source_root_relative"]
    try:
        record_path, record = load_acquisition_record(args.source_root, expected_root, args.download_dir)
    except (ValueError, json.JSONDecodeError) as error:
        print(f"Input provenance validation failed: {error}")
        return 2
    json_out = args.output_dir / "isolet_horu.json"
    if json_out.exists():
        print(f"Refusing to overwrite existing result: {json_out}")
        return 2
    protocol = manifest["protocol"]
    command = [
        sys.executable, "run_hd_checkpoint_comparison.py",
        "--datasets", "isolet_raw", "--methods", *protocol["methods"],
        "--device", protocol["device"],
        "--deterministic-algorithms", "--torch-num-threads", str(protocol["torch_num_threads"]),
        "--seeds", str(protocol["seed"]),
        "--round-checkpoints", *(str(value) for value in protocol["round_checkpoints"]),
        "--local-epochs", str(protocol["local_epochs"]), "--batch-size", str(protocol["batch_size"]),
        "--client-participation", str(protocol["client_participation"]),
        "--hd-dim", str(protocol["hd_dim"]), "--hd-lr", str(protocol["hd_lr"]),
        "--subspace-shared-rank", str(protocol["subspace_shared_rank"]),
        "--subspace-intersection-rank", str(protocol["subspace_intersection_rank"]),
        "--subspace-personal-rank", str(protocol["subspace_personal_rank"]),
        "--json-out", str(json_out), "--md-out", str(args.output_dir / "isolet_horu.md"),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Run Manifest\nCommand:\n" + " ".join(command))
    print("Result status: PROTOTYPE_ONLY")
    completed = subprocess.run(command, cwd=REPO_ROOT, env={**os.environ, "HORU_SOURCE_DATA_ROOT": str(args.source_root)}, check=False)
    if completed.returncode != 0:
        return completed.returncode
    report = json.loads(json_out.read_text(encoding="utf-8"))
    report["artifact_provenance"] = {
        "result_status": manifest["result_status"],
        "manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "manifest_sha256": sha256(MANIFEST_PATH),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_dirty": False,
        "execution_mode": "fresh",
        "source_root": str(expected_root),
        "acquisition_record": str(record_path),
        "acquisition_record_sha256": sha256(record_path),
        "source_archives": record["archives"],
        "limitations": manifest["limitations"],
    }
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote provenance-labeled prototype report: {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
