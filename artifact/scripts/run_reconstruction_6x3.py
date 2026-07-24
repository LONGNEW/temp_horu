#!/usr/bin/env python3
"""Alias entrypoint for the six-dataset, three-method CUDA reconstruction run."""

from __future__ import annotations

import runpy
from pathlib import Path


SCRIPT = Path(__file__).with_name("run_cuda_reconstruction_suite.py")


if __name__ == "__main__":
    runpy.run_path(str(SCRIPT), run_name="__main__")
