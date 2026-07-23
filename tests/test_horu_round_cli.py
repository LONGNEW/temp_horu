import json
import subprocess
import sys
import torch
from conftest import write_fixture_cache


def test_horu_round_cli_smoke_and_resume(tmp_path):
    root = write_fixture_cache(tmp_path / "data")
    bootstrap_config = tmp_path / "bootstrap.yaml"
    bootstrap_config.write_text("method: horu\ndataset: ucihar\nsubject_ids: [1, 2, 3]\nhd_dim: 16\ncommon_rank: 3\nglobal_rank: 2\npersonal_rank: 4\npersonal_basis_policy: reduced_svd\nseed: 0\ndevice: cpu\nbootstrap_only: true\ntest_ratio: 0.3\n")
    bootstrap = tmp_path / "bootstrap"; base = [sys.executable, "-m", "horu_artifact", "federated", "--method", "horu", "--data-root", str(root), "--device", "cpu"]
    done = subprocess.run(base + ["--config", str(bootstrap_config), "--output", str(bootstrap), "--bootstrap-only"], capture_output=True, text=True); assert done.returncode == 0, done.stderr
    round_config = tmp_path / "round.yaml"
    round_config.write_text("method: horu\ndataset: ucihar\nsubject_ids: [1, 2, 3]\nhd_dim: 16\ncommon_rank: 3\nglobal_rank: 2\npersonal_rank: 4\npersonal_basis_policy: reduced_svd\nseed: 0\nrounds: 2\nlocal_epochs: 1\nbatch_size: 16\neta_shared: 0.035\neta_personal: 0.035\neta_global: 0.035\nprovenance: USER_SPECIFIED_SMOKE\ndevice: cpu\ntest_ratio: 0.3\n")
    output = tmp_path / "round"; command = base + ["--config", str(round_config), "--output", str(output), "--bootstrap-checkpoint", str(bootstrap / "checkpoints" / "bootstrap.pt")]
    done = subprocess.run(command, capture_output=True, text=True); assert done.returncode == 0, done.stderr
    result = json.loads((output / "result.json").read_text()); assert result["rounds"] == 2 and result["result_status"] == "SMOKE_TEST_ONLY"
    manifest = json.loads((output / "state_manifest.json").read_text())
    assert (output / "timing_samples.csv").exists() and manifest["server_state"] == ["C_bar", "G_bar"]
    timing_header = (output / "timing_samples.csv").read_text().splitlines()[0].split(",")
    assert {"timing_scope", "coefficient_similarity_ms", "coefficient_update_ms", "final_train_prediction_ms", "class_error_statistics_ms", "client_round_table_ii_ms"} <= set(timing_header)
    assert result["final"]["timing_scope"] == "table_ii_and_iii.round"
    assert result["final"]["synchronization_table_ii_ms"] == (
        result["final"]["server_aggregation_total_ms"]
        + result["final"]["client_shared_branch_update_ms"]
    )
    assert result["final"]["local_round_ms"] == result["final"]["client_round_mean_ms"]
    assert result["final"]["server_step_ms"] == result["final"]["synchronization_table_ii_ms"]
    assert len(manifest["round_state_hashes"]) == 2 and set(manifest["round_state_hashes"][0]["clients"]["1"]) == {"C", "G", "delta", "P"}
    done = subprocess.run(command + ["--resume"], capture_output=True, text=True); assert done.returncode == 0, done.stderr
    resumed = json.loads((output / "result.json").read_text())
    repeat = tmp_path / "repeat"; done = subprocess.run(base + ["--config", str(round_config), "--output", str(repeat), "--bootstrap-checkpoint", str(bootstrap / "checkpoints" / "bootstrap.pt")], capture_output=True, text=True); assert done.returncode == 0, done.stderr
    assert resumed["final"]["aggregated_common_sha256"] == json.loads((repeat / "result.json").read_text())["final"]["aggregated_common_sha256"]
    tampered = tmp_path / "tampered.pt"; checkpoint = torch.load(bootstrap / "checkpoints" / "bootstrap.pt", weights_only=False); checkpoint["C_global"][0, 0] += 1; torch.save(checkpoint, tampered)
    done = subprocess.run(base + ["--config", str(round_config), "--output", str(tmp_path / "bad"), "--bootstrap-checkpoint", str(tampered)], capture_output=True, text=True)
    assert done.returncode == 2 and "tensor hash" in done.stderr
    incompatible = tmp_path / "incompatible.yaml"; incompatible.write_text(round_config.read_text().replace("seed: 0", "seed: 1"))
    done = subprocess.run(base + ["--config", str(incompatible), "--output", str(tmp_path / "incompatible"), "--bootstrap-checkpoint", str(bootstrap / "checkpoints" / "bootstrap.pt")], capture_output=True, text=True)
    assert done.returncode == 2 and "incompatible" in done.stderr
