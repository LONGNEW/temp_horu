import json
import subprocess
import sys
import torch
from conftest import write_fixture_cache


def test_horu_bootstrap_cli_checkpoint_and_cache(tmp_path):
    root = write_fixture_cache(tmp_path / "data")
    config = tmp_path / "horu.yaml"
    config.write_text("method: horu\ndataset: ucihar\nsubject_ids: [1, 2, 3]\nhd_dim: 16\ncommon_rank: 3\nglobal_rank: 2\npersonal_rank: 4\npersonal_basis_policy: reduced_svd\nseed: 0\ndevice: cpu\nbootstrap_only: true\n")
    output = tmp_path / "output"
    command = [sys.executable, "-m", "horu_artifact", "federated", "--method", "horu", "--config", str(config), "--data-root", str(root), "--output", str(output), "--device", "cpu", "--bootstrap-only"]
    done = subprocess.run(command, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    result = json.loads((output / "result.json").read_text())
    assert result["result_status"] == "SMOKE_TEST_ONLY"
    assert result["bootstrap"]["bootstrap_upload_bytes"] == 3 * 6 * 16 * 4
    assert result["bootstrap"]["bootstrap_download_bytes"] == 3 * (16 * 5 + 6 * 5) * 4
    bootstrap = result["bootstrap"]
    assert bootstrap["server_bootstrap_total_ms"] == (
        bootstrap["server_common_global_basis_ms"]
        + bootstrap["server_client_hv_projection_ms"]
        + bootstrap["server_common_global_coefficients_ms"]
    )
    checkpoint = torch.load(output / "checkpoints" / "bootstrap.pt", weights_only=False)
    assert checkpoint["B_c"].shape == (16, 3) and checkpoint["B_g"].shape == (16, 2)
    assert checkpoint["states"][1].train_cache["z_p"].shape[1] == 4
    assert torch.equal(torch.load(output / "checkpoints" / "bootstrap.pt", weights_only=False)["C_global"], checkpoint["C_global"])
    assert (output / "reconstruction_metrics.csv").exists()
