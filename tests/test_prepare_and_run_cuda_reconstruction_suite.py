from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "artifact" / "scripts" / "prepare_and_run_cuda_reconstruction_suite.py"
SPEC = importlib.util.spec_from_file_location("prepare_and_run_cuda_reconstruction_suite", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_selected_datasets_preserves_order_and_deduplicates() -> None:
    assert MODULE._selected_datasets(["wisdm", "femnist", "wisdm"]) == ["wisdm", "femnist"]
    assert MODULE._selected_datasets(None) == list(MODULE.MANIFEST_DATASETS)


def test_prepare_commands_require_matching_inputs(tmp_path: Path) -> None:
    args = argparse.Namespace(
        uci_har_source_root=tmp_path / "uci_har",
        uci_har_archive=tmp_path / "downloads" / "uci.zip",
        isolet_raw_source_root=tmp_path / "isolet",
        isolet_download_dir=tmp_path / "downloads" / "isolet",
        femnist_source_root=tmp_path / "femnist",
        wisdm_source_root=tmp_path / "wisdm",
        wisdm_outer_archive=tmp_path / "downloads" / "wisdm.zip",
        synthetic_source_root=tmp_path / "synthetic",
        ninapro_db1_source_root=tmp_path / "ninapro",
        ninapro_download_dir=tmp_path / "downloads" / "ninapro",
        output_dir=tmp_path / "out",
    )
    commands = MODULE._prepare_commands(args, ["uci_har", "wisdm", "synthetic"])
    assert commands[0][1:] == [
        "artifact/scripts/acquire_uci_har_prototype.py",
        "--source-root",
        str(tmp_path / "uci_har"),
        "--archive",
        str(tmp_path / "downloads" / "uci.zip"),
    ]
    assert commands[1][1:] == [
        "artifact/scripts/acquire_wisdm_reconstruction.py",
        "--source-root",
        str(tmp_path / "wisdm"),
        "--outer-archive",
        str(tmp_path / "downloads" / "wisdm.zip"),
    ]
    assert commands[2][1:] == [
        "artifact/scripts/prepare_synthetic_reconstruction.py",
        "--source-root",
        str(tmp_path / "synthetic"),
    ]


def test_suite_command_for_subset_uses_prepared_roots(tmp_path: Path) -> None:
    args = argparse.Namespace(
        uci_har_source_root=tmp_path / "uci_har",
        isolet_raw_source_root=tmp_path / "isolet",
        femnist_source_root=None,
        wisdm_source_root=None,
        synthetic_source_root=tmp_path / "synthetic",
        ninapro_db1_source_root=None,
        output_dir=tmp_path / "out",
    )
    command = MODULE._suite_command(args, ["uci_har", "isolet_raw", "synthetic"])
    assert command[1:] == [
        "artifact/scripts/run_cuda_reconstruction_suite.py",
        "--output-dir",
        str(tmp_path / "out"),
        "--dataset",
        "uci_har",
        "--dataset",
        "isolet_raw",
        "--dataset",
        "synthetic",
        "--uci-har-source-root",
        str(tmp_path / "uci_har"),
        "--isolet-raw-source-root",
        str(tmp_path / "isolet"),
        "--synthetic-source-root",
        str(tmp_path / "synthetic"),
    ]
