#!/usr/bin/env python3
"""Alias entrypoint for controlled-systems Table I/II/III reproduction."""

from __future__ import annotations

import runpy
from pathlib import Path


SCRIPT = Path(__file__).with_name("run_table_reproduction.py")


if __name__ == "__main__":
    runpy.run_path(str(SCRIPT), run_name="__main__")
