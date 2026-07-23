from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "artifact" / "scripts" / "run_table_reproduction.py"
SPEC = importlib.util.spec_from_file_location("run_table_reproduction", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_run_table_reproduction_wrapper_builds_expected_commands(tmp_path: Path) -> None:
    prepare = MODULE._prepare_command(tmp_path / "data")
    reproduce = MODULE._reproduce_command(tmp_path / "data", tmp_path / "results", 2, 7, 1)
    assert str(ROOT) in str(MODULE.REPO_ROOT)
    assert prepare[3] == "prepare-data"
    assert prepare[-2:] == ["--data-root", str(tmp_path / "data")]
    assert prepare[3] == "prepare-data"
    assert reproduce[3] == "reproduce-tables"
    assert reproduce[-6:] == ["--warmup", "2", "--repeats", "7", "--threads", "1"]
