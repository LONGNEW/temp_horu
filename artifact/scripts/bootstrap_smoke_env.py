#!/usr/bin/env python3
"""Install the pinned, CPU-only smoke environment under LONGNEW_DATA_ROOT."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = REPO_ROOT / "artifact" / "requirements-smoke.txt"


def main() -> int:
    data_root = Path(os.environ.get("LONGNEW_DATA_ROOT", "/home/longnew/data"))
    target = data_root / "envs" / "horu-artifact-smoke" / "site-packages"
    target.mkdir(parents=True, exist_ok=True)
    base_requirements = [line for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#") and not line.startswith("torch==")]
    print(f"Installing smoke dependencies into {target}")
    common = [sys.executable, "-m", "pip", "install", "--upgrade", "--target", str(target), "--no-warn-script-location"]
    subprocess.run([*common, *base_requirements], check=True)
    subprocess.run(
        [
            *common,
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
            "torch==2.7.1+cpu",
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
