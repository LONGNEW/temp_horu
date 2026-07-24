#!/usr/bin/env python3
"""Create the explicitly pinned CUDA reconstruction-screening Python environment."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = REPO_ROOT / "artifact" / "requirements-cuda-screening.txt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-dir",
        type=Path,
        required=True,
        help="Directory that will contain an isolated CUDA site-packages target.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    target = args.env_dir / "site-packages"
    if target.exists() and any(target.iterdir()):
        print(f"Refusing to alter existing CUDA site-packages target: {target}")
        return 2
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--target",
        str(target),
        "--requirement",
        str(REQUIREMENTS),
    ]
    print(
        "Environment manifest:\n"
        + "\n".join(
            [
                f"- requirements: {REQUIREMENTS}",
                f"- target: {target}",
                f"- command: {' '.join(command)}",
            ]
        )
    )
    if args.dry_run:
        return 0
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch, scipy; print(torch.__version__, torch.cuda.is_available(), scipy.__version__)",
        ],
        env={**os.environ, "PYTHONPATH": str(target)},
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
