#!/usr/bin/env python3
"""Verify a pinned LEAF FEMNIST reconstruction input inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "artifact" / "manifests" / "reconstruction_femnist_v1.json"


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
    record_path = args.source_root / "femnist_reconstruction_input.json"
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
    leaf_root = args.source_root / "leaf"
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=leaf_root, text=True).strip()
        if revision != manifest["source"]["revision"]:
            failures.append("LEAF revision differs from manifest")
    except (OSError, subprocess.CalledProcessError):
        failures.append(f"pinned LEAF checkout is unavailable: {leaf_root}")
    patched = leaf_root / "data" / "femnist" / "preprocess" / "data_to_json.py"
    if not patched.is_file() or record.get("patched_data_to_json_sha256") != sha256(patched):
        failures.append("Pillow compatibility patch hash differs from input record")
    data_root = leaf_root / "data" / "femnist" / "data"
    train = sorted((data_root / "train").glob("*.json"))
    test = sorted((data_root / "test").glob("*.json"))
    if len(train) != 36 or len(test) != 36:
        failures.append("expected exactly 36 FEMNIST train and 36 test JSON shards")
    for split, paths, expected in (("train", train, record.get("train_shards")), ("test", test, record.get("test_shards"))):
        actual = {path.name: sha256(path) for path in paths}
        if actual != expected:
            failures.append(f"{split} shard inventory differs from input record")
    users: list[str] = []
    for path in train:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            failures.append(f"invalid train JSON: {path}")
            continue
        if not {"users", "num_samples", "user_data"}.issubset(payload):
            failures.append(f"LEAF keys missing from train shard: {path}")
            continue
        users.extend(str(user) for user in payload["users"])
        if len(users) >= 200:
            break
    first_200_hash = hashlib.sha256("\n".join(users[:200]).encode("utf-8")).hexdigest()
    if len(users) < 200 or record.get("canonical_first_200_user_sha256") != first_200_hash:
        failures.append("canonical first-200 writer inventory differs from input record")
    if failures:
        print("FEMNIST INPUT VERIFICATION FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 2
    print("FEMNIST RECONSTRUCTION INPUT VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
