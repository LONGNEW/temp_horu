from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "artifact" / "scripts" / "run_cuda_reconstruction_suite.py"
SPEC = importlib.util.spec_from_file_location("run_cuda_reconstruction_suite", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_selected_datasets_preserves_order_and_deduplicates() -> None:
    assert MODULE._selected_datasets(["wisdm", "femnist", "wisdm"]) == ["wisdm", "femnist"]
    assert MODULE._selected_datasets(None) == list(MODULE.MANIFEST_DATASETS)


def test_suite_payload_respects_selected_subset() -> None:
    protocol = {
        "seed": 42,
        "rounds": 25,
        "client_participation": 1.0,
        "local_epochs": 3,
        "batch_size": 32,
        "hd_dim": 2000,
        "hd_lr": 0.035,
        "device": "cuda",
        "subspace_intersection_rank": 24,
        "subspace_shared_rank": 32,
        "subspace_personal_rank": 64,
    }
    payload = MODULE._suite_payload(protocol, ["wisdm", "synthetic"])
    assert payload["datasets"] == ["wisdm", "synthetic"]
    assert payload["seeds"] == [42]
    assert payload["horu"]["global_rank"] == 8


def test_resolve_selected_sources_accepts_top_level_preparation_dirs(tmp_path: Path) -> None:
    isolet_root = tmp_path / "isolet" / "data" / "raw" / "isolet"
    isolet_root.mkdir(parents=True)
    (isolet_root / "isolet1+2+3+4.data").write_text("0," * 617 + "1\n", encoding="utf-8")
    (isolet_root / "isolet5.data").write_text("0," * 617 + "1\n", encoding="utf-8")

    femnist_root = tmp_path / "femnist" / "data" / "tiers" / "standard_pfl" / "femnist"
    (femnist_root / "train").mkdir(parents=True)
    (femnist_root / "test").mkdir(parents=True)

    wisdm_root = tmp_path / "wisdm" / "data" / "tiers" / "on_device_hdc" / "wisdm"
    wisdm_root.mkdir(parents=True)
    (wisdm_root / "wisdm-dataset.zip").write_text("placeholder", encoding="utf-8")

    ninapro_root = tmp_path / "ninapro" / "data" / "tiers" / "on_device_hdc" / "ninapro_db1"
    ninapro_root.mkdir(parents=True)
    (ninapro_root / "S1_A1_E1.mat").write_text("placeholder", encoding="utf-8")

    args = argparse.Namespace(
        isolet_raw_source_root=tmp_path / "isolet",
        femnist_source_root=tmp_path / "femnist",
        wisdm_source_root=tmp_path / "wisdm",
        ninapro_db1_source_root=tmp_path / "ninapro",
        uci_har_source_root=None,
        synthetic_source_root=None,
    )
    resolved = MODULE._resolve_selected_sources(args, ["isolet_raw", "femnist", "wisdm", "ninapro_db1"])
    assert resolved == {
        "isolet": str(isolet_root),
        "femnist": str(femnist_root),
        "wisdm": str(wisdm_root / "wisdm-dataset.zip"),
        "ninapro": str(ninapro_root),
    }
