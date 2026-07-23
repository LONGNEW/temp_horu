import csv, json, subprocess, sys
from conftest import write_fixture_cache


def test_hyperfeel_cli_offline_resume_and_delta_payload(tmp_path):
    root = write_fixture_cache(tmp_path / "data")
    config = tmp_path / "hyperfeel.yaml"
    config.write_text("method: hyperfeel\nimplementation_mode: paper_faithful\ndataset: ucihar\nsubject_ids: [1, 2, 3]\ntest_ratio: 0.3\nrounds: 2\nparticipation: 1.0\nlocal_epochs: 1\nbatch_size: 16\nhd_dim: 16\nlearning_rate: 0.035\nsimilarity: dot\nnormalize_update_hypervectors: false\nnormalize_prototypes: false\nserver_aggregation: sum\nseed: 0\ndevice: cpu\n")
    output = tmp_path / "out"
    command = [sys.executable, "-m", "horu_artifact", "federated", "--method", "hyperfeel", "--config", str(config), "--data-root", str(root), "--output", str(output), "--device", "cpu"]
    done = subprocess.run(command, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    result = json.loads((output / "result.json").read_text())
    assert result["method"] == "hyperfeel"
    assert result["final"]["timing_scope"] == "table_iii.round"
    assert abs(result["final"]["local_round_ms"] - result["final"]["similarity_ms"] - result["final"]["update_ms"]) < 1e-12
    assert result["final"]["uploaded_payload_bytes"] == 6 * 16 * 4
    assert result["bootstrap"]["bootstrap_upload_bytes"] == 3 * 6 * 16 * 4
    assert json.loads((output / "communication.csv").read_text().splitlines()[2].split(",")[2]) == 3 * 6 * 16 * 4
    bootstrap = result["bootstrap"]
    assert len(set(json.loads(bootstrap["initial_client_am_hashes"]).values())) == 1
    with (output / "client_metrics.csv").open() as stream:
        client_rows = list(csv.DictReader(stream))
    assert all(row["personalized_am_sha256"] and row["upload_delta_sha256"] for row in client_rows)
    first_round = {row["client_id"]: row for row in client_rows if row["round"] == "1"}
    second_round = {row["client_id"]: row for row in client_rows if row["round"] == "2"}
    assert all(second_round[cid]["round_start_personalized_am_sha256"] == first_round[cid]["personalized_am_sha256"] for cid in first_round)
    resumed = subprocess.run(command + ["--resume"], capture_output=True, text=True)
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads((output / "result.json").read_text())["final"]["global_delta_sha256"] == result["final"]["global_delta_sha256"]
