from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "artifact" / "scripts" / "run_cuda_reconstruction_suite.py"
SPEC = importlib.util.spec_from_file_location("run_cuda_reconstruction_suite", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_selected_datasets_preserves_order_and_deduplicates() -> None:
    protocol = {"datasets": ["uci_har", "isolet_raw", "femnist", "wisdm", "synthetic", "ninapro_db1"]}
    assert MODULE._selected_datasets(["wisdm", "femnist", "wisdm"], protocol) == ["wisdm", "femnist"]
    assert MODULE._selected_datasets(None, protocol) == protocol["datasets"]


def test_command_for_uses_manifest_protocol() -> None:
    protocol = json.loads(MODULE.MANIFEST_PATH.read_text(encoding="utf-8"))["protocol"]
    command = MODULE.command_for("synthetic", Path("/tmp/source"), Path("/tmp/out"), protocol)
    assert command[:6] == [
        MODULE.sys.executable,
        "run_hd_checkpoint_comparison.py",
        "--datasets",
        "synthetic",
        "--methods",
        "horu_hd",
    ]
    assert "--json-out" in command


def test_materialize_source_root_creates_staged_symlink(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    dataset_output = tmp_path / "out"
    dataset_output.mkdir()
    staged = MODULE._materialize_source_root("synthetic", prepared, dataset_output)
    link = staged / MODULE.DATASET_RELATIVE_ROOTS["synthetic"]
    assert staged == dataset_output / "_source_root"
    assert link.is_symlink()
    assert link.resolve() == prepared.resolve()
