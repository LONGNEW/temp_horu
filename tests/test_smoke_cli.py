import json
import subprocess
import sys

from conftest import write_fixture_cache


def test_smoke_cli_offline_and_reproducible(tmp_path):
    root = write_fixture_cache(tmp_path / "data")
    config = tmp_path / "config.yaml"
    config.write_text("dataset: ucihar\nsubject_ids: [1, 2, 3]\ntest_ratio: 0.3\nseed: 0\nhd_dim: 16\nlearning_rate: 0.035\nlocal_epochs: 1\nbatch_size: 8\ndevice: cpu\n", encoding="utf-8")
    absent = subprocess.run([sys.executable, "-m", "horu_artifact", "smoke", "--config", str(config), "--data-root", str(tmp_path / "absent"), "--output", str(tmp_path / "none")], capture_output=True, text=True)
    assert absent.returncode != 0 and "prepare-data" in absent.stderr
    outputs = []
    for name in ("one", "two"):
        out = tmp_path / name
        done = subprocess.run([sys.executable, "-m", "horu_artifact", "smoke", "--config", str(config), "--data-root", str(root), "--output", str(out)], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        result = json.loads((out / "result.json").read_text())
        assert result["num_updates"] > 0 and result["projection_shape"] == [561, 16]
        assert all((out / file).exists() for file in result["output_files"])
        outputs.append(result)
    assert outputs[0]["projection_sha256"] == outputs[1]["projection_sha256"]
    assert outputs[0]["num_updates"] == outputs[1]["num_updates"]
    assert outputs[0]["final_mean_accuracy"] == outputs[1]["final_mean_accuracy"]
