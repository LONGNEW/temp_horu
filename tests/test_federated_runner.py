import csv
import json
import torch
from horu_artifact.config import FederatedConfig
from horu_artifact.federated.runner import _pooled_global_accuracy, run_federated
from conftest import write_fixture_cache

def test_runner_bootstrap_payload_clones_resume_and_determinism(tmp_path):
    root = write_fixture_cache(tmp_path / "data")
    config = FederatedConfig("ucihar", [1, 2, 3], .3, 7, 16, .035, 1, 16, 2)
    first = run_federated(config, root, tmp_path / "one")
    second = run_federated(config, root, tmp_path / "two")
    assert first["final"]["global_sha256"] == second["final"]["global_sha256"]
    assert first["evaluation_protocol"] == "global_model_on_all_participating_client_test_samples_pooled_accuracy"
    assert first["official_global_pooled_test_samples"] == 54
    assert first["final"]["timing_scope"] == "table_iii.round"
    assert abs(first["final"]["local_round_ms"] - first["final"]["similarity_ms"] - first["final"]["update_ms"]) < 1e-12
    assert first["final"]["uploaded_payload_bytes"] == 6 * 16 * 4
    bootstrap = first["bootstrap"]
    assert bootstrap["bootstrap_upload_bytes"] == 3 * 6 * 16 * 4
    assert bootstrap["bootstrap_download_bytes"] == 3 * 6 * 16 * 4
    assert all(bootstrap[key] >= 0 for key in ("client_bootstrap_sum_ms", "client_bootstrap_max_ms", "server_bootstrap_ms", "broadcast_ms"))
    assert len(json.loads(bootstrap["initial_local_model_hashes"])) == 3
    with (tmp_path / "one" / "client_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len({row["round_start_global_sha256"] for row in rows[:3]}) == 1
    resumed = run_federated(config, root, tmp_path / "one", resume=True)
    assert resumed["final"]["global_sha256"] == first["final"]["global_sha256"]

def test_official_metric_is_global_model_pooled_over_all_test_samples():
    model = torch.tensor([[1., 0.], [0., 1.]])
    clients = {
        1: {"test_h": torch.tensor([[1., 0.], [1., 0.], [0., 1.]]), "test_y": torch.tensor([0, 0, 1])},
        2: {"test_h": torch.tensor([[1., 0.]]), "test_y": torch.tensor([1])},
    }
    # Per-client mean would be .5; official pooled evaluation is 3 correct / 4.
    accuracy, samples = _pooled_global_accuracy(model, clients)
    assert samples == 4 and accuracy == .75
