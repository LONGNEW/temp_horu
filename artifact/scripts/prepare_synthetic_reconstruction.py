#!/usr/bin/env python3
"""Create a pinned LEAF Synthetic seed-42 reconstruction input from scratch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "artifact" / "manifests" / "reconstruction_synthetic_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--leaf-repository", default="https://github.com/TalwalkarLab/leaf.git")
    args = parser.parse_args()
    if args.source_root.exists():
        print(f"Refusing to alter existing source root: {args.source_root}", file=sys.stderr)
        return 2
    try:
        import numpy  # noqa: F401
        import scipy  # noqa: F401
    except ImportError as error:
        print(f"Missing LEAF Synthetic preparation dependency: {error}", file=sys.stderr)
        return 2
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    source = manifest["source"]
    generator = manifest["generator"]
    preprocess = manifest["preprocess"]
    args.source_root.parent.mkdir(parents=True, exist_ok=True)
    leaf_root = args.source_root / "leaf"
    try:
        run(["git", "clone", args.leaf_repository, str(leaf_root)])
        run(["git", "checkout", "--detach", str(source["revision"])], cwd=leaf_root)
        observed_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=leaf_root, text=True).strip()
        if observed_revision != source["revision"]:
            raise RuntimeError(f"LEAF revision mismatch: expected {source['revision']}, got {observed_revision}")
        synthetic_root = leaf_root / "data" / "synthetic"
        run(
            [
                sys.executable,
                "main.py",
                "-num-tasks",
                str(generator["num_tasks"]),
                "-num-classes",
                str(generator["num_classes"]),
                "-num-dim",
                str(generator["num_dim"]),
                "-seed",
                str(generator["seed"]),
            ],
            cwd=synthetic_root,
        )
        run(
            [
                "bash",
                "preprocess.sh",
                "-s",
                str(preprocess["sampling"]),
                "--sf",
                str(preprocess["sample_fraction"]),
                "-k",
                str(preprocess["minimum_samples_per_user"]),
                "-t",
                str(preprocess["partition"]),
                "--tf",
                str(preprocess["train_fraction"]),
                "--smplseed",
                str(preprocess["sampling_seed"]),
                "--spltseed",
                str(preprocess["split_seed"]),
            ],
            cwd=synthetic_root,
        )
        train_files = sorted((synthetic_root / "data" / "train").glob("*.json"))
        test_files = sorted((synthetic_root / "data" / "test").glob("*.json"))
        if len(train_files) != 1 or len(test_files) != 1:
            raise RuntimeError("LEAF preprocessing did not produce exactly one train and one test JSON shard")
        destination = args.source_root / "data" / "leaf_synthetic" / "data"
        (destination / "train").mkdir(parents=True)
        (destination / "test").mkdir(parents=True)
        copied_train = destination / "train" / train_files[0].name
        copied_test = destination / "test" / test_files[0].name
        shutil.copy2(train_files[0], copied_train)
        shutil.copy2(test_files[0], copied_test)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Synthetic input preparation failed: {error}", file=sys.stderr)
        return 2
    record = {
        "result_status": manifest["result_status"],
        "manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "manifest_sha256": sha256(MANIFEST_PATH),
        "leaf_repository": args.leaf_repository,
        "leaf_revision": source["revision"],
        "source_root": str(args.source_root),
        "generated_files": {
            "train": {"path": str(copied_train), "sha256": sha256(copied_train)},
            "test": {"path": str(copied_test), "sha256": sha256(copied_test)},
        },
        "commands": {
            "generator": ["main.py", "-num-tasks", generator["num_tasks"], "-num-classes", generator["num_classes"], "-num-dim", generator["num_dim"], "-seed", generator["seed"]],
            "preprocess": preprocess,
        },
    }
    record_path = args.source_root / "synthetic_reconstruction_input.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
