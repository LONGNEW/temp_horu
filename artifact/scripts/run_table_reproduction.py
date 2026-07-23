#!/usr/bin/env python3
"""Prepare the controlled fixture and reproduce Tables I, II, and III."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(command: list[str], env: dict[str, str]) -> int:
    print("Command:\n" + " ".join(command))
    return subprocess.run(command, cwd=REPO_ROOT, env=env, check=False).returncode


def _prepare_command(data_root: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "horu_artifact",
        "prepare-data",
        "controlled-systems",
        "--data-root",
        str(data_root),
    ]


def _reproduce_command(data_root: Path, output_dir: Path, warmup: int, repeats: int, threads: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "horu_artifact",
        "reproduce-tables",
        "--data-root",
        str(data_root),
        "--output",
        str(output_dir),
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
        "--threads",
        str(threads),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    env = {"PYTHONPATH": str(REPO_ROOT / "src"), **dict(os.environ)}
    prepare = _prepare_command(args.data_root)
    if _run(prepare, env) != 0:
        return 2

    reproduce = _reproduce_command(args.data_root, args.output_dir, args.warmup, args.repeats, args.threads)
    return _run(reproduce, env)


if __name__ == "__main__":
    raise SystemExit(main())
