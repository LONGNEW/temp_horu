from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifact" / "manifests" / "reconstruction_cuda_suite_v1.json"
SUITE = ROOT / "reference_results" / "cuda_suite"


def test_verified_reference_suite_is_complete_and_immutable() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "artifact" / "scripts" / "verify_reconstruction_suite.py"),
            "--manifest",
            str(MANIFEST),
            "--suite-output",
            str(SUITE),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "RECONSTRUCTION SUITE VERIFIED" in completed.stdout


def test_reference_means_match_the_verified_cuda_screen() -> None:
    summary = json.loads((SUITE / "summary.json").read_text(encoding="utf-8"))
    expected = {
        "horu_hd": 77.45016984711891,
        "hyperfeel": 73.70544700790941,
        "fedhdc": 59.20833233640665,
    }
    assert summary["result_status"] == "CUDA_RECONSTRUCTION_SCREENING_ONLY"
    assert summary["datasets"] == [
        "uci_har",
        "isolet_raw",
        "femnist",
        "wisdm",
        "synthetic",
        "ninapro_db1",
    ]
    for method, value in expected.items():
        assert summary["methods"][method]["mean_accuracy_percent"] == value


def test_canonical_horu_and_metric_contract_are_importable() -> None:
    from horu_artifact.horu.bootstrap import bootstrap_horu
    from horu_artifact.methods import fedhdc, hyperfeel

    assert callable(bootstrap_horu)
    assert callable(fedhdc.bundled_model)
    assert callable(hyperfeel.bundled_model)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["protocol"]["metric_contract"] == {
        "horu_hd": "mean_personalized_accuracy: unweighted mean over client/subject test accuracies",
        "hyperfeel": "mean_personalized_accuracy: unweighted mean over client/subject test accuracies",
        "fedhdc": "global_test_accuracy: sample-weighted accuracy of the single global prototype over concatenated client test samples",
    }
