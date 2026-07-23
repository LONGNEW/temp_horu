import json, subprocess, sys
from conftest import write_fixture_cache

def test_federated_cli_offline(tmp_path):
    root = write_fixture_cache(tmp_path / "data")
    config = tmp_path / "fed.yaml"
    config.write_text("dataset: ucihar\nsubject_ids: [1, 2, 3]\ntest_ratio: 0.3\nseed: 0\nhd_dim: 16\nlearning_rate: 0.035\nlocal_epochs: 1\nbatch_size: 16\nrounds: 2\ndevice: cpu\nmethod: fedhdc\nsimilarity: dot\nnormalize_update_hypervectors: true\n")
    output = tmp_path / "out"
    done = subprocess.run([sys.executable, "-m", "horu_artifact", "federated", "--method", "fedhdc", "--config", str(config), "--data-root", str(root), "--output", str(output), "--device", "cpu"], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    result = json.loads((output / "result.json").read_text())
    assert all((output / item).exists() for item in result["output_files"])
