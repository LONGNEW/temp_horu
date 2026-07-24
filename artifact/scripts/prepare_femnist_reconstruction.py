#!/usr/bin/env python3
"""Create a pinned public LEAF FEMNIST seed-42 reconstruction input."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "artifact" / "manifests" / "reconstruction_femnist_seed42_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def apply_pillow_compatibility_patch(leaf_root: Path) -> str:
    """Apply the exact Pillow-10 spelling for LEAF's removed ANTIALIAS alias."""
    target = leaf_root / "data" / "femnist" / "preprocess" / "data_to_json.py"
    before = target.read_text(encoding="utf-8")
    old = "gray.thumbnail(size, Image.ANTIALIAS)"
    new = "gray.thumbnail(size, Image.Resampling.LANCZOS)"
    if new in before:
        return hashlib.sha256(before.encode("utf-8")).hexdigest()
    if before.count(old) != 1:
        raise RuntimeError("unexpected LEAF Pillow compatibility patch target")
    after = before.replace(old, new)
    target.write_text(after, encoding="utf-8")
    return hashlib.sha256(after.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--leaf-repository", default="https://github.com/TalwalkarLab/leaf.git")
    parser.add_argument("--resume-existing", action="store_true", help="Resume a source root created by this script after a recoverable preprocessing failure.")
    args = parser.parse_args()
    if args.source_root.exists() and not args.resume_existing:
        print(f"Refusing to alter existing source root: {args.source_root}", file=sys.stderr)
        return 2
    try:
        import numpy  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError as error:
        print(f"Missing LEAF FEMNIST preparation dependency: {error}", file=sys.stderr)
        return 2
    for executable in ("git", "wget", "unzip"):
        if not shutil_which(executable):
            print(f"Required executable is unavailable: {executable}", file=sys.stderr)
            return 2
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    source, preprocess = manifest["source"], manifest["preprocess"]
    leaf_root = args.source_root / "leaf"
    try:
        if args.resume_existing:
            if not leaf_root.is_dir():
                raise RuntimeError("resume-existing requires a LEAF checkout under source-root/leaf")
        else:
            args.source_root.parent.mkdir(parents=True, exist_ok=True)
            run(["git", "clone", args.leaf_repository, str(leaf_root)])
            run(["git", "checkout", "--detach", str(source["revision"])], cwd=leaf_root)
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=leaf_root, text=True).strip()
        if revision != source["revision"]:
            raise RuntimeError(f"LEAF revision mismatch: expected {source['revision']}, got {revision}")
        compatibility_patch_sha256 = apply_pillow_compatibility_patch(leaf_root)
        femnist_root = leaf_root / "data" / "femnist"
        run(
            [
                "bash", "preprocess.sh", "-s", str(preprocess["sampling"]),
                "--sf", str(preprocess["sample_fraction"]), "-k", str(preprocess["minimum_samples_per_user"]),
                "-t", str(preprocess["partition"]), "--tf", str(preprocess["train_fraction"]),
                "--smplseed", str(preprocess["sampling_seed"]), "--spltseed", str(preprocess["split_seed"]),
            ],
            cwd=femnist_root,
        )
        train = sorted((femnist_root / "data" / "train").glob("*.json"))
        test = sorted((femnist_root / "data" / "test").glob("*.json"))
        if not train or not test or len(train) != len(test):
            raise RuntimeError("LEAF FEMNIST preprocessing did not produce matching train/test JSON shards")
        link_parent = args.source_root / "data" / "tiers" / "standard_pfl"
        link_parent.mkdir(parents=True)
        os.symlink(femnist_root / "data", link_parent / "femnist")
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"FEMNIST input preparation failed: {error}", file=sys.stderr)
        return 2
    selected_users: list[str] = []
    for path in train:
        payload = json.loads(path.read_text(encoding="utf-8"))
        selected_users.extend(str(user) for user in payload.get("users", []))
        if len(selected_users) >= 200:
            break
    record = {
        "result_status": manifest["result_status"],
        "manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "manifest_sha256": sha256(MANIFEST_PATH),
        "leaf_repository": args.leaf_repository,
        "leaf_revision": source["revision"],
        "source_root": str(args.source_root),
        "leaf_data_root": str(femnist_root / "data"),
        "pillow_compatibility_patch": "Image.ANTIALIAS -> Image.Resampling.LANCZOS",
        "patched_data_to_json_sha256": compatibility_patch_sha256,
        "train_shards": {path.name: sha256(path) for path in train},
        "test_shards": {path.name: sha256(path) for path in test},
        "canonical_first_200_user_sha256": hashlib.sha256("\n".join(selected_users[:200]).encode("utf-8")).hexdigest(),
    }
    (args.source_root / "femnist_reconstruction_input.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


def shutil_which(program: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / program
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
