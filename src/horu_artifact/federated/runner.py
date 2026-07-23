"""Single-process FedHDC runner; timing is compute/copy timing, never network latency."""
from __future__ import annotations
import csv, hashlib, json, platform, time
from pathlib import Path
import torch, yaml
from ..config import FederatedConfig, HoruBootstrapConfig, HoruRoundConfig
from ..datasets.ucihar import DOI, load_cache, split_subjects
from ..hdc.encoder import NonlinearEncoder, make_projection
from ..hdc.prototype import PrototypeMemory
from ..methods import fedhdc, hyperfeel
from ..horu.bootstrap import bootstrap_horu
from ..horu.inference import coefficient_gram, predict
from ..horu.synchronization import aggregate_shared, absorb_shared
from ..horu.training import error_statistics, norm_diagnostics, train_client
from ..runtime import resolve_device
from .metrics import summary, tensor_hash

def _ms(start: int) -> float: return (time.perf_counter_ns() - start) / 1_000_000
def _accuracy(model: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> float:
    return float((PrototypeMemory(model).predict(x, "dot") == y).float().mean().item())

def _pooled_global_accuracy(model: torch.Tensor, clients: dict) -> tuple[float, int]:
    """Official metric: one global model over every participating test row."""
    test_h = torch.cat([client["test_h"] for client in clients.values()])
    test_y = torch.cat([client["test_y"] for client in clients.values()])
    return _accuracy(model, test_h, test_y), int(test_y.numel())

def _run_fedhdc(config: FederatedConfig, data_root: str | Path, output: str | Path, device_override: str | None = None, overwrite: bool = False, resume: bool = False) -> dict:
    out = Path(output)
    if out.exists() and any(out.iterdir()) and not overwrite and not resume: raise FileExistsError(f"output directory {out} is non-empty; pass --overwrite or --resume")
    out.mkdir(parents=True, exist_ok=True); checkpoints = out / "checkpoints"; checkpoints.mkdir(exist_ok=True)
    device = resolve_device(device_override or config.device); data = load_cache(data_root); splits = split_subjects(data, config.subject_ids, config.test_ratio, config.seed)
    projection = make_projection(data.features.shape[1], config.hd_dim, config.seed); projection_hash = tensor_hash(projection); encoder = NonlinearEncoder(projection.to(device))
    clients = {}
    for cid in config.subject_ids:
        ix = splits[cid]; clients[cid] = {"train_h": encoder.encode(data.features[ix["train"]].to(device)), "train_y": data.labels[ix["train"]].to(device), "test_h": encoder.encode(data.features[ix["test"]].to(device)), "test_y": data.labels[ix["test"]].to(device)}
    payload_per_client = 6 * config.hd_dim * 4
    payload = len(clients) * payload_per_client
    bootstrap_rows, rounds, client_rows, communication = [], [], [], []
    state_path = checkpoints / "latest.pt"
    start_round = 0
    if resume and state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False); global_model = state["global_model"]; start_round = state["next_round"]
        bootstrap_rows = state["bootstrap_rows"]; rounds = state["rounds"]; client_rows = state["client_rows"]; communication = state["communication"]
    else:
        local, client_ms = [], []
        for cid, client in clients.items():
            begun = time.perf_counter_ns(); cached_h = client["train_h"].clone(); cache_read_ms = _ms(begun)
            begun = time.perf_counter_ns(); model = fedhdc.bundled_model(cached_h, client["train_y"], 6); bundling_and_normalize_ms = _ms(begun)
            begun = time.perf_counter_ns(); local.append(model.clone()); serialization_ms = _ms(begun)
            elapsed = cache_read_ms + bundling_and_normalize_ms + serialization_ms; client_ms.append(elapsed)
            bootstrap_rows.append({"client_id": cid, "encoded_cache_read_ms": cache_read_ms, "class_bundling_and_row_normalize_ms": bundling_and_normalize_ms, "initial_model_copy_ms": serialization_ms, "client_bootstrap_ms": elapsed})
        begun = time.perf_counter_ns(); received = [model.clone() for model in local]; receive_copy_ms = _ms(begun)
        begun = time.perf_counter_ns(); global_model = fedhdc.weighted_aggregate(received, [int(clients[c]["train_y"].numel()) for c in config.subject_ids]); server_ms = _ms(begun)
        begun = time.perf_counter_ns(); _ = [global_model.clone() for _ in clients]; broadcast_ms = _ms(begun)
        bootstrap_rows.append({"client_id": "server", "client_bootstrap_ms": 0.0, "initial_model_receive_copy_ms": receive_copy_ms, "server_weighted_aggregate_and_normalize_ms": server_ms, "server_bootstrap_ms": receive_copy_ms + server_ms, "broadcast_ms": broadcast_ms, "client_bootstrap_sum_ms": sum(client_ms), "client_bootstrap_max_ms": max(client_ms), "bootstrap_sequential_ms": sum(client_ms) + receive_copy_ms + server_ms + broadcast_ms, "bootstrap_parallel_estimate_ms": max(client_ms) + receive_copy_ms + server_ms + broadcast_ms, "bootstrap_upload_bytes": payload, "bootstrap_download_bytes": payload, "initial_global_sha256": tensor_hash(global_model), "initial_local_model_hashes": json.dumps([tensor_hash(x) for x in local])})
        communication.append({"stage": "bootstrap", "round": 0, "upload_bytes": payload, "download_bytes": payload})
    for round_id in range(start_round, config.rounds):
        models, sample_counts, timings, similarity_times, update_times = [], [], [], [], []
        for cid, client in clients.items():
            round_start_hash = tensor_hash(global_model)
            model = global_model.clone()
            table_iii_timing = {"similarity_ms": 0.0, "update_ms": 0.0}
            for epoch in range(config.local_epochs):
                generator = torch.Generator(device="cpu").manual_seed(config.seed + cid * 1009 + round_id * 100_003 + epoch)
                order = torch.randperm(client["train_h"].shape[0], generator=generator).to(device)
                # Deterministic shuffle occurs once per epoch; batch semantics remain fixed-prediction.
                updates = fedhdc.train_batches(model, client["train_h"][order], client["train_y"][order], config.learning_rate, config.batch_size, table_iii_timing)
            elapsed = table_iii_timing["similarity_ms"] + table_iii_timing["update_ms"]
            models.append(model); sample_counts.append(int(client["train_y"].numel())); timings.append(elapsed); similarity_times.append(table_iii_timing["similarity_ms"]); update_times.append(table_iii_timing["update_ms"])
            accuracy = _accuracy(model, client["test_h"], client["test_y"])
            client_rows.append({"timing_scope": "table_iii.client", "round": round_id + 1, "client_id": cid, "accuracy": accuracy, "updates": updates, "similarity_ms": table_iii_timing["similarity_ms"], "update_ms": table_iii_timing["update_ms"], "local_round_table_iii_ms": elapsed, "round_start_global_sha256": round_start_hash, "model_sha256": tensor_hash(model)})
        begin = time.perf_counter_ns(); global_model = fedhdc.weighted_aggregate(models, sample_counts); server_ms = _ms(begin)
        client_global_accs = [_accuracy(global_model, c["test_h"], c["test_y"]) for c in clients.values()]
        pooled_accuracy, pooled_samples = _pooled_global_accuracy(global_model, clients)
        item = {"timing_scope": "table_iii.round", "round": round_id + 1, "local_round_ms": sum(timings) / len(timings), "similarity_ms": sum(similarity_times) / len(similarity_times), "update_ms": sum(update_times) / len(update_times), "server_step_ms": server_ms, "uploaded_payload_bytes": payload_per_client, "client_train_sum_aux_ms": sum(timings), "client_train_max_aux_ms": max(timings), "round_sequential_ms": sum(timings) + server_ms, "round_parallel_estimate_ms": max(timings) + server_ms, "global_sha256": tensor_hash(global_model), "global_pooled_test_accuracy": pooled_accuracy, "global_pooled_test_samples": pooled_samples, **{f"global_model_client_{key}": value for key, value in summary(client_global_accs).items()}}
        rounds.append(item); communication.append({"stage": "round", "round": round_id + 1, "upload_bytes": payload, "download_bytes": payload, "upload_per_client_bytes": payload_per_client, "download_per_client_bytes": payload_per_client})
        torch.save({"global_model": global_model, "next_round": round_id + 1, "bootstrap_rows": bootstrap_rows, "rounds": rounds, "client_rows": client_rows, "communication": communication}, state_path)
    _write_csv(out / "bootstrap_metrics.csv", bootstrap_rows); _write_csv(out / "round_metrics.csv", rounds); _write_csv(out / "client_metrics.csv", client_rows); _write_csv(out / "communication.csv", communication)
    resolved = config.to_dict(); resolved["device"] = str(device); resolved["client_selection_sha256"] = hashlib.sha256(",".join(map(str, config.subject_ids)).encode()).hexdigest(); resolved["evaluation_protocol"] = "global_model_on_all_participating_client_test_samples_pooled_accuracy"
    (out / "config.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    environment = {"python": platform.python_version(), "torch": torch.__version__, "device": str(device), "git_commit": "unavailable_no_git_repository"}
    (out / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    result = {"status": "pass", "method": "fedhdc", "dataset": "ucihar", "dataset_doi": DOI, "dataset_sha256": data.manifest.get("sha256", ""), "data_prepare_ms": 0.0, "projection_sha256": projection_hash, "client_selection_sha256": resolved["client_selection_sha256"], "evaluation_protocol": resolved["evaluation_protocol"], "num_clients": len(clients), "rounds": config.rounds, "bootstrap": bootstrap_rows[-1], "final": rounds[-1], "official_global_pooled_test_accuracy": rounds[-1]["global_pooled_test_accuracy"], "official_global_pooled_test_samples": rounds[-1]["global_pooled_test_samples"], "sequential_total_ms": bootstrap_rows[-1]["bootstrap_sequential_ms"] + sum(x["round_sequential_ms"] for x in rounds), "parallel_estimate_total_ms": bootstrap_rows[-1]["bootstrap_parallel_estimate_ms"] + sum(x["round_parallel_estimate_ms"] for x in rounds), "output_files": ["result.json", "config.resolved.yaml", "environment.json", "bootstrap_metrics.csv", "round_metrics.csv", "client_metrics.csv", "communication.csv", "checkpoints"]}
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); return result

def _write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def _run_horu_bootstrap(config: HoruBootstrapConfig, data_root: str | Path, output: str | Path, device_override: str | None, overwrite: bool) -> dict:
    """Run the explicitly bootstrap-only HoRU flow, without federated rounds."""
    out = Path(output)
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory {out} is non-empty; pass --overwrite")
    out.mkdir(parents=True, exist_ok=True); checkpoints = out / "checkpoints"; checkpoints.mkdir(exist_ok=True)
    device = resolve_device(device_override or config.device); data = load_cache(data_root)
    splits = split_subjects(data, config.subject_ids, config.test_ratio, config.seed)
    projection = make_projection(data.features.shape[1], config.hd_dim, config.seed)
    encoder = NonlinearEncoder(projection.to(device)); clients = {}
    split_hashes = {}
    for cid in config.subject_ids:
        indices = splits[cid]; split_hashes[str(cid)] = {key: tensor_hash(value) for key, value in indices.items()}
        clients[cid] = {"train_h": encoder.encode(data.features[indices["train"]].to(device)), "train_y": data.labels[indices["train"]].to(device), "test_h": encoder.encode(data.features[indices["test"]].to(device)), "test_y": data.labels[indices["test"]].to(device)}
    states, common_basis, global_basis, server, client_rows, reconstruction_rows = bootstrap_horu(clients, config.common_rank, config.global_rank, config.personal_rank, config.personal_basis_policy)
    # Required Table-I-style payload: client prototype upload and shared broadcast only.
    upload_per_client = 6 * config.hd_dim * 4
    download_per_client = (config.hd_dim * (config.common_rank + config.global_rank) + 6 * (config.common_rank + config.global_rank)) * 4
    broadcast_start = time.perf_counter_ns(); broadcast = {cid: {"B_c": common_basis.clone(), "B_g": global_basis.clone(), "C_global": states[cid].common.clone(), "G_global": states[cid].global_coefficients.clone()} for cid in states}; broadcast_copy_ms = _ms(broadcast_start)
    if len({id(item["B_c"]) for item in broadcast.values()}) != len(broadcast):
        raise RuntimeError("broadcast bases must be independently cloned")
    server["broadcast_copy_ms"] = broadcast_copy_ms
    server["upload_per_client_bytes"] = upload_per_client; server["download_per_client_bytes"] = download_per_client
    server["bootstrap_upload_bytes"] = len(states) * upload_per_client; server["bootstrap_download_bytes"] = len(states) * download_per_client
    checkpoint = {"config": config.to_dict(), "projection": projection.cpu(), "projection_sha256": tensor_hash(projection), "B_c": common_basis.cpu(), "B_g": global_basis.cpu(), "C_global": next(iter(states.values())).common.cpu(), "G_global": next(iter(states.values())).global_coefficients.cpu(), "states": states, "server": server, "split_hashes": split_hashes}
    torch.save(checkpoint, checkpoints / "bootstrap.pt")
    _write_csv(out / "bootstrap_metrics.csv", client_rows + [{"client_id": "server", **server}]); _write_csv(out / "reconstruction_metrics.csv", reconstruction_rows)
    manifest = {"bootstrap_only": True, "state_shapes": {str(cid): {"M_i": list(state.prototype.shape), "C_i": list(state.common.shape), "G_i": list(state.global_coefficients.shape), "delta_i": list(state.delta.shape), "B_p_i": list(state.personal_basis.shape), "P_i": list(state.personal.shape), "query_caches": {"train": {key: {"shape": list(value.shape), "dtype": str(value.dtype), "source_split_sha256": split_hashes[str(cid)]["train"], "basis_projector_sha256": server["projector_sha256"] if key in {"z_c", "z_g"} else tensor_hash(state.personal_basis @ state.personal_basis.T)} for key,value in state.train_cache.items()}, "test": {key: {"shape": list(value.shape), "dtype": str(value.dtype), "source_split_sha256": split_hashes[str(cid)]["test"], "basis_projector_sha256": server["projector_sha256"] if key in {"z_c", "z_g"} else tensor_hash(state.personal_basis @ state.personal_basis.T)} for key,value in state.test_cache.items()}}} for cid,state in states.items()}, "source_split_sha256": split_hashes, "projection_sha256": tensor_hash(projection), "shared_basis_projector_sha256": server["projector_sha256"], "personal_rank_ambiguity": "full_svd uses USER_SPECIFIED_NUMERICAL_COMPLETION for zero-singular directions; reduced_svd is unique only through available reduced vectors.", "communicated_state": ["B_c", "B_g", "C_global", "G_global"], "local_only_state": ["B_p_i", "P_i", "delta_i", "train/test query coefficient caches"]}
    (out / "state_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    result_status = "SMOKE_TEST_ONLY" if len(config.subject_ids) == 3 else "PAPER_RANK_DIAGNOSTIC_ONLY"
    resolved = config.to_dict(); resolved.update({"device": str(device), "test_ratio_provenance": "REPO_EXISTING", "result_status": result_status, "projection_sha256": tensor_hash(projection)})
    (out / "config.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    (out / "environment.json").write_text(json.dumps({"python": platform.python_version(), "torch": torch.__version__, "device": str(device), "git_commit": "unavailable_no_git_repository"}, indent=2) + "\n", encoding="utf-8")
    result = {"status": "pass", "result_status": result_status, "method": "horu", "bootstrap_only": True, "dataset": "ucihar", "dataset_doi": DOI, "dataset_sha256": data.manifest.get("sha256", ""), "num_clients": len(states), "projection_sha256": tensor_hash(projection), "bootstrap": server, "output_files": ["result.json", "config.resolved.yaml", "environment.json", "bootstrap_metrics.csv", "state_manifest.json", "reconstruction_metrics.csv", "checkpoints/bootstrap.pt"]}
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _run_hyperfeel(config: FederatedConfig, data_root: str | Path, output: str | Path, device_override: str | None, overwrite: bool, resume: bool) -> dict:
    """Run HyperFeel with persistent per-client AMs and delta-only transport."""
    out = Path(output)
    if out.exists() and any(out.iterdir()) and not overwrite and not resume:
        raise FileExistsError(f"output directory {out} is non-empty; pass --overwrite or --resume")
    out.mkdir(parents=True, exist_ok=True); checkpoints = out / "checkpoints"; checkpoints.mkdir(exist_ok=True)
    device = resolve_device(device_override or config.device); data = load_cache(data_root)
    splits = split_subjects(data, config.subject_ids, config.test_ratio, config.seed)
    projection = make_projection(data.features.shape[1], config.hd_dim, config.seed); projection_hash = tensor_hash(projection)
    encoder = NonlinearEncoder(projection.to(device)); clients = {}
    for cid in config.subject_ids:
        ix = splits[cid]
        clients[cid] = {"train_h": encoder.encode(data.features[ix["train"]].to(device)), "train_y": data.labels[ix["train"]].to(device), "test_h": encoder.encode(data.features[ix["test"]].to(device)), "test_y": data.labels[ix["test"]].to(device)}
    num_classes = 6
    payload_per_client = num_classes * config.hd_dim * 4
    payload = len(clients) * payload_per_client
    bootstrap_rows: list[dict] = []; rounds: list[dict] = []; client_rows: list[dict] = []; communication: list[dict] = []
    state_path = checkpoints / "latest.pt"; start_round = 0
    if resume and state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        personalized, previous_delta, previous_weights = state["personalized"], state["previous_delta"], state["previous_weights"]
        start_round, bootstrap_rows, rounds, client_rows, communication = state["next_round"], state["bootstrap_rows"], state["rounds"], state["client_rows"], state["communication"]
    else:
        local: list[torch.Tensor] = []; timings = []
        for cid, client in clients.items():
            begun = time.perf_counter_ns(); cached = client["train_h"].clone(); cache_ms = _ms(begun)
            begun = time.perf_counter_ns(); model = hyperfeel.bundled_model(cached, client["train_y"], num_classes); bundle_ms = _ms(begun)
            begun = time.perf_counter_ns(); local.append(model.clone()); copy_ms = _ms(begun); elapsed = cache_ms + bundle_ms + copy_ms; timings.append(elapsed)
            bootstrap_rows.append({"client_id": cid, "encoded_cache_read_ms": cache_ms, "class_bundling_ms": bundle_ms, "initial_am_upload_copy_ms": copy_ms, "client_bootstrap_ms": elapsed, "local_am_sha256": tensor_hash(model)})
        begun = time.perf_counter_ns(); received = [x.clone() for x in local]; receive_ms = _ms(begun)
        begun = time.perf_counter_ns(); central = hyperfeel.sum_deltas(received)
        if config.normalize_prototypes: central = hyperfeel.normalize_rows(central)
        aggregate_ms = _ms(begun)
        begun = time.perf_counter_ns(); personalized = {cid: central.clone() for cid in clients}; broadcast_ms = _ms(begun)
        previous_delta = torch.zeros_like(central); previous_weights = {cid: torch.zeros(num_classes, device=device) for cid in clients}
        bootstrap_rows.append({"client_id": "server", "initial_am_receive_copy_ms": receive_ms, "server_central_am_sum_ms": aggregate_ms, "broadcast_copy_ms": broadcast_ms, "client_bootstrap_sum_ms": sum(timings), "client_bootstrap_max_ms": max(timings), "bootstrap_sequential_ms": sum(timings) + receive_ms + aggregate_ms + broadcast_ms, "bootstrap_parallel_estimate_ms": max(timings) + receive_ms + aggregate_ms + broadcast_ms, "bootstrap_upload_bytes": payload, "bootstrap_download_bytes": payload, "central_am_sha256": tensor_hash(central), "initial_local_am_hashes": json.dumps([tensor_hash(x) for x in local]), "initial_client_am_hashes": json.dumps({str(cid): tensor_hash(x) for cid, x in personalized.items()})})
        communication.append({"stage": "bootstrap", "round": 0, "upload_bytes": payload, "download_bytes": payload})
    for round_id in range(start_round, config.rounds):
        deltas = []; timings = []; similarity_times = []; update_times = []
        for cid, client in clients.items():
            memory = personalized[cid]; start_hash = tensor_hash(memory)
            table_iii_timing = {"similarity_ms": 0.0, "update_ms": 0.0}
            begun = time.perf_counter_ns()
            hyperfeel.apply_personalization(memory, previous_delta, previous_weights[cid], config.learning_rate)
            delta = torch.zeros_like(memory); errors = torch.zeros(num_classes, device=device); counts = torch.zeros(num_classes, device=device); updates = 0
            table_iii_timing["update_ms"] += _ms(begun)
            # T006 applies the common local_epochs budget to each method.  The
            # HyperFeel primitive is one sequential local pass, so aggregate
            # exactly that primitive over the configured number of passes.
            for _ in range(config.local_epochs):
                epoch_delta, epoch_errors, epoch_counts, epoch_updates = hyperfeel.retrain_batches(
                    memory, client["train_h"], client["train_y"], config.learning_rate,
                    config.batch_size, config.normalize_update_hypervectors,
                    config.normalize_prototypes, table_iii_timing,
                )
                begun = time.perf_counter_ns()
                delta += epoch_delta; errors += epoch_errors; counts += epoch_counts; updates += epoch_updates
                table_iii_timing["update_ms"] += _ms(begun)
            begun = time.perf_counter_ns(); weights = hyperfeel.personalization_weights(errors, counts); previous_weights[cid] = weights; table_iii_timing["update_ms"] += _ms(begun)
            elapsed = table_iii_timing["similarity_ms"] + table_iii_timing["update_ms"]; timings.append(elapsed); similarity_times.append(table_iii_timing["similarity_ms"]); update_times.append(table_iii_timing["update_ms"])
            accuracy = _accuracy(memory, client["test_h"], client["test_y"]); deltas.append(delta)
            client_rows.append({"timing_scope": "table_iii.client", "round": round_id + 1, "client_id": cid, "accuracy": accuracy, "updates": updates, "similarity_ms": table_iii_timing["similarity_ms"], "update_ms": table_iii_timing["update_ms"], "local_round_table_iii_ms": elapsed, "round_start_personalized_am_sha256": start_hash, "personalized_am_sha256": tensor_hash(memory), "upload_delta_sha256": tensor_hash(delta), "class_counts": json.dumps(counts.cpu().tolist()), "class_errors": json.dumps(errors.cpu().tolist()), "personalization_weights": json.dumps(weights.cpu().tolist())})
        begun = time.perf_counter_ns(); previous_delta = hyperfeel.sum_deltas(deltas); aggregate_ms = _ms(begun)
        accuracies = [_accuracy(personalized[cid], c["test_h"], c["test_y"]) for cid, c in clients.items()]
        all_predictions = torch.cat([PrototypeMemory(personalized[cid]).predict(c["test_h"], "dot") for cid, c in clients.items()]); all_labels = torch.cat([c["test_y"] for c in clients.values()])
        pooled = float((all_predictions == all_labels).float().mean().item())
        rounds.append({"timing_scope": "table_iii.round", "round": round_id + 1, "local_round_ms": sum(timings) / len(timings), "similarity_ms": sum(similarity_times) / len(similarity_times), "update_ms": sum(update_times) / len(update_times), "server_step_ms": aggregate_ms, "uploaded_payload_bytes": payload_per_client, "client_train_sum_aux_ms": sum(timings), "client_train_max_aux_ms": max(timings), "round_sequential_ms": sum(timings) + aggregate_ms, "round_parallel_estimate_ms": max(timings) + aggregate_ms, "global_delta_sha256": tensor_hash(previous_delta), "personalized_pooled_test_accuracy": pooled, "personalized_pooled_test_samples": int(all_labels.numel()), **{f"personalized_{key}": value for key, value in summary(accuracies).items()}})
        communication.append({"stage": "round", "round": round_id + 1, "upload_bytes": payload, "download_bytes": payload, "upload_per_client_bytes": payload_per_client, "download_per_client_bytes": payload_per_client})
        torch.save({"personalized": personalized, "previous_delta": previous_delta, "previous_weights": previous_weights, "next_round": round_id + 1, "bootstrap_rows": bootstrap_rows, "rounds": rounds, "client_rows": client_rows, "communication": communication}, state_path)
    _write_csv(out / "bootstrap_metrics.csv", bootstrap_rows); _write_csv(out / "round_metrics.csv", rounds); _write_csv(out / "client_metrics.csv", client_rows); _write_csv(out / "communication.csv", communication)
    resolved = config.to_dict(); resolved["device"] = str(device); resolved["client_selection_sha256"] = hashlib.sha256(",".join(map(str, config.subject_ids)).encode()).hexdigest(); resolved["evaluation_protocol"] = "personalized_client_am_on_each_participating_client_test_samples_pooled_accuracy"; resolved["source_pdf"] = "10.1109/ASP-DAC58780.2024.10473907"; resolved["source_pdf_sha256"] = "c298831aec5d1c929e67e6670ebcd5823be125ad6440b805a1cc9a7c2d4aa734"; resolved["config_sha256"] = hashlib.sha256(yaml.safe_dump(config.to_dict(), sort_keys=True).encode()).hexdigest()
    (out / "config.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    (out / "environment.json").write_text(json.dumps({"python": platform.python_version(), "torch": torch.__version__, "device": str(device), "git_commit": "unavailable_no_git_repository"}, indent=2) + "\n", encoding="utf-8")
    result = {"status": "pass", "method": "hyperfeel", "implementation_mode": config.implementation_mode, "source_pdf": resolved["source_pdf"], "source_pdf_sha256": resolved["source_pdf_sha256"], "config_sha256": resolved["config_sha256"], "dataset": "ucihar", "dataset_doi": DOI, "dataset_sha256": data.manifest.get("sha256", ""), "projection_sha256": projection_hash, "client_selection_sha256": resolved["client_selection_sha256"], "evaluation_protocol": resolved["evaluation_protocol"], "num_clients": len(clients), "rounds": config.rounds, "bootstrap": bootstrap_rows[-1], "final": rounds[-1], "personalized_pooled_test_accuracy": rounds[-1]["personalized_pooled_test_accuracy"], "sequential_total_ms": bootstrap_rows[-1]["bootstrap_sequential_ms"] + sum(x["round_sequential_ms"] for x in rounds), "parallel_estimate_total_ms": bootstrap_rows[-1]["bootstrap_parallel_estimate_ms"] + sum(x["round_parallel_estimate_ms"] for x in rounds), "output_files": ["result.json", "config.resolved.yaml", "environment.json", "bootstrap_metrics.csv", "round_metrics.csv", "client_metrics.csv", "communication.csv", "checkpoints"]}
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _horu_accuracy(state, cache_name: str, labels: torch.Tensor, gram: torch.Tensor) -> float:
    cache = getattr(state, cache_name)
    predictions = [predict(cache["z_c"][i], cache["z_g"][i], cache["z_p"][i], state.common, state.global_coefficients, state.delta, state.personal, gram) for i in range(labels.numel())]
    return float((torch.tensor(predictions, device=labels.device) == labels).float().mean().item())


def _validate_horu_checkpoint(checkpoint_path: str | Path, config: HoruRoundConfig, projection: torch.Tensor, split_hashes: dict) -> dict:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"bootstrap checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=projection.device, weights_only=False)
    required = {"config", "projection_sha256", "B_c", "B_g", "C_global", "states", "server", "split_hashes"}
    if not required.issubset(checkpoint):
        raise ValueError("bootstrap checkpoint is missing required T004 state")
    expected = {"dataset": config.dataset, "subject_ids": config.subject_ids, "hd_dim": config.hd_dim, "common_rank": config.common_rank, "global_rank": config.global_rank, "personal_rank": config.personal_rank, "personal_basis_policy": config.personal_basis_policy, "seed": config.seed, "test_ratio": config.test_ratio}
    if any(checkpoint["config"].get(key) != value for key, value in expected.items()):
        raise ValueError("bootstrap checkpoint config is incompatible with recurring config")
    if checkpoint["projection_sha256"] != tensor_hash(projection) or checkpoint["split_hashes"] != split_hashes:
        raise ValueError("bootstrap checkpoint projection or split hash does not match current data")
    if checkpoint["B_c"].shape != (config.hd_dim, config.common_rank) or checkpoint["B_g"].shape != (config.hd_dim, config.global_rank) or checkpoint["C_global"].shape != (6, config.common_rank):
        raise ValueError("bootstrap checkpoint tensor integrity check failed")
    raw_basis = torch.cat([checkpoint["B_c"], checkpoint["B_g"]], dim=1)
    if checkpoint["server"].get("raw_basis_sha256") != tensor_hash(raw_basis) or checkpoint["server"].get("common_consensus_sha256") != tensor_hash(checkpoint["C_global"]):
        raise ValueError("bootstrap checkpoint tensor hash does not match its T004 manifest")
    for cid, state in checkpoint["states"].items():
        if state.client_id != cid or not torch.equal(state.common, checkpoint["C_global"]):
            raise ValueError("bootstrap checkpoint client shared state is inconsistent")
    return checkpoint


def _run_horu_rounds(config: HoruRoundConfig, data_root: str | Path, output: str | Path, device_override: str | None, overwrite: bool, resume: bool, bootstrap_checkpoint: str | Path | None) -> dict:
    out = Path(output)
    if out.exists() and any(out.iterdir()) and not overwrite and not resume:
        raise FileExistsError(f"output directory {out} is non-empty; pass --overwrite or --resume")
    if bootstrap_checkpoint is None:
        raise ValueError("HoRU recurring rounds require a bootstrap checkpoint")
    out.mkdir(parents=True, exist_ok=True); checkpoints = out / "checkpoints"; checkpoints.mkdir(exist_ok=True)
    device = resolve_device(device_override or config.device); data = load_cache(data_root); splits = split_subjects(data, config.subject_ids, config.test_ratio, config.seed)
    projection = make_projection(data.features.shape[1], config.hd_dim, config.seed).to(device); split_hashes = {str(cid): {key: tensor_hash(value) for key, value in splits[cid].items()} for cid in config.subject_ids}
    initial = _validate_horu_checkpoint(bootstrap_checkpoint, config, projection, split_hashes)
    basis_c, basis_g = initial["B_c"].to(device), initial["B_g"].to(device)
    gram_by_client = {cid: coefficient_gram(basis_c, basis_g, state.personal_basis.to(device)) for cid, state in initial["states"].items()}
    state_path = checkpoints / "latest.pt"; rounds: list[dict] = []; client_rows: list[dict] = []; timing_rows: list[dict] = []; communication: list[dict] = []; state_hashes: list[dict] = []; start_round = 0
    if resume and state_path.exists():
        saved = torch.load(state_path, map_location=device, weights_only=False); states = saved["client_states"]; rounds, client_rows, timing_rows, communication, state_hashes, start_round = saved["rounds"], saved["client_rows"], saved["timing_rows"], saved["communication"], saved["state_hashes"], saved["next_round"]
    else:
        states = initial["states"]
        for cid, state in states.items():
            # T004 checkpoints predate label fields; labels are recovered from their hash-validated split.
            if not hasattr(state, "train_labels"):
                state.train_labels = data.labels[splits[cid]["train"]]
                state.test_labels = data.labels[splits[cid]["test"]]
            for name in ("prototype", "class_counts", "common", "global_coefficients", "delta", "personal_basis", "personal", "train_labels", "test_labels"):
                setattr(state, name, getattr(state, name).to(device))
            state.train_cache = {key: value.to(device) for key, value in state.train_cache.items()}; state.test_cache = {key: value.to(device) for key, value in state.test_cache.items()}
    for round_id in range(start_round, config.rounds):
        local_times = []; class_stats = {}; local_hashes = {}
        for cid in config.subject_ids:
            state = states[cid]
            # Integrity diagnostics are deliberately outside the latency region.
            local_hashes[cid] = {"delta": tensor_hash(state.delta), "personal": tensor_hash(state.personal), "basis": tensor_hash(state.personal_basis)}
            local_start = time.perf_counter_ns()
            updates, timing = train_client(state, config.local_epochs, config.batch_size, config.eta_shared, config.eta_personal, gram_by_client[cid], config.seed, round_id)
            local_predict_update_total_ms = _ms(local_start)
            counts, errors, ratios, final_predict_ms, statistics_ms = error_statistics(state, gram_by_client[cid]); class_stats[cid] = (counts, errors)
            table_ii_total = timing["coefficient_similarity_ms"] + timing["coefficient_update_ms"] + final_predict_ms + statistics_ms
            local_times.append(table_ii_total)
            accuracy = _horu_accuracy(state, "test_cache", state.test_labels, gram_by_client[cid])
            row = {"timing_scope": "table_ii.client", "round": round_id + 1, "client_id": cid, "accuracy": accuracy, "updates": updates, "class_train_counts": json.dumps(counts.cpu().tolist()), "class_errors": json.dumps(errors.cpu().tolist()), "error_ratios": json.dumps(ratios.cpu().tolist()), "local_predict_update_wall_ms": local_predict_update_total_ms, **timing, "final_train_prediction_ms": final_predict_ms, "class_error_statistics_ms": statistics_ms, "client_round_table_ii_ms": table_ii_total, **norm_diagnostics(state)}
            client_rows.append(row); timing_rows.append({"timing_scope": "table_ii.client", "round": round_id + 1, "client_id": cid, "local_predict_update_wall_ms": local_predict_update_total_ms, **timing, "final_train_prediction_ms": final_predict_ms, "class_error_statistics_ms": statistics_ms, "client_round_table_ii_ms": table_ii_total})
        aggregate_common, aggregate_global, server_timing = aggregate_shared([states[cid] for cid in config.subject_ids])
        absorption_times = []
        for cid in config.subject_ids:
            state = states[cid]; local_hashes[cid] = {"delta": tensor_hash(state.delta), "personal": tensor_hash(state.personal), "basis": tensor_hash(state.personal_basis)}
            counts, errors = class_stats[cid]
            absorption_times.append(absorb_shared(state, aggregate_common, aggregate_global, counts, errors, config.eta_global, config.gate_alpha, config.gate_min, config.gate_max))
            if tensor_hash(state.delta) != local_hashes[cid]["delta"] or tensor_hash(state.personal) != local_hashes[cid]["personal"] or tensor_hash(state.personal_basis) != local_hashes[cid]["basis"]:
                raise RuntimeError("shared absorption modified local-only HoRU state")
        accuracies = [_horu_accuracy(states[cid], "test_cache", states[cid].test_labels, gram_by_client[cid]) for cid in config.subject_ids]
        labels = torch.cat([states[cid].test_labels for cid in config.subject_ids]); predictions = torch.cat([torch.tensor([predict(states[cid].test_cache["z_c"][i], states[cid].test_cache["z_g"][i], states[cid].test_cache["z_p"][i], states[cid].common, states[cid].global_coefficients, states[cid].delta, states[cid].personal, gram_by_client[cid]) for i in range(states[cid].test_labels.numel())], device=device) for cid in config.subject_ids])
        pooled = float((predictions == labels).float().mean().item())
        client_round_mean = sum(local_times) / len(local_times)
        client_update_mean = sum(absorption_times) / len(absorption_times)
        synchronization_table_ii = server_timing["server_aggregation_total_ms"] + client_update_mean
        mean_similarity = sum(row["coefficient_similarity_ms"] for row in timing_rows if row["round"] == round_id + 1) / len(local_times)
        mean_update = sum(row["coefficient_update_ms"] for row in timing_rows if row["round"] == round_id + 1) / len(local_times)
        upload_per_client = 6 * (config.common_rank + config.global_rank) * 4
        rounds.append({"timing_scope": "table_ii_and_iii.round", "round": round_id + 1, "local_round_ms": client_round_mean, "similarity_ms": mean_similarity, "update_ms": mean_update, "server_step_ms": synchronization_table_ii, "uploaded_payload_bytes": upload_per_client, "client_round_mean_ms": client_round_mean, "client_round_sum_aux_ms": sum(local_times), "client_round_max_aux_ms": max(local_times), **server_timing, "client_shared_branch_update_ms": client_update_mean, "client_shared_branch_update_sum_aux_ms": sum(absorption_times), "client_shared_branch_update_max_aux_ms": max(absorption_times), "synchronization_table_ii_ms": synchronization_table_ii, "round_sequential_aux_ms": sum(local_times) + server_timing["server_aggregation_total_ms"] + sum(absorption_times), "round_table_iii_ms": client_round_mean + synchronization_table_ii, "aggregated_common_sha256": tensor_hash(aggregate_common), "aggregated_global_sha256": tensor_hash(aggregate_global), "personalized_pooled_test_accuracy": pooled, "personalized_pooled_test_samples": int(labels.numel()), **{f"personalized_{key}": value for key, value in summary(accuracies).items()}})
        upload = len(states) * 6 * (config.common_rank + config.global_rank) * 4; communication.append({"stage": "round", "round": round_id + 1, "upload_bytes": upload, "download_bytes": upload, "upload_per_client_bytes": upload // len(states), "download_per_client_bytes": upload // len(states)})
        state_hashes.append({"round": round_id + 1, "clients": {str(cid): {"C": tensor_hash(states[cid].common), "G": tensor_hash(states[cid].global_coefficients), "delta": tensor_hash(states[cid].delta), "P": tensor_hash(states[cid].personal)} for cid in config.subject_ids}, "C_bar": tensor_hash(aggregate_common), "G_bar": tensor_hash(aggregate_global)})
        torch.save({"server_shared": {"C_bar": aggregate_common.cpu(), "G_bar": aggregate_global.cpu()}, "client_states": states, "next_round": round_id + 1, "rounds": rounds, "client_rows": client_rows, "timing_rows": timing_rows, "communication": communication, "state_hashes": state_hashes}, state_path)
    _write_csv(out / "round_metrics.csv", rounds); _write_csv(out / "client_metrics.csv", client_rows); _write_csv(out / "timing_samples.csv", timing_rows); _write_csv(out / "communication.csv", communication)
    result_status = "SMOKE_TEST_ONLY" if config.run_profile == "smoke" else "PAPER_RANK_DIAGNOSTIC_ONLY"
    config_hash = hashlib.sha256(yaml.safe_dump(config.to_dict(), sort_keys=True).encode()).hexdigest(); bootstrap_hash = hashlib.sha256(Path(bootstrap_checkpoint).read_bytes()).hexdigest(); resolved = config.to_dict(); resolved.update({"device": str(device), "result_status": result_status, "config_sha256": config_hash, "bootstrap_checkpoint_sha256": bootstrap_hash, "projection_sha256": tensor_hash(projection), "split_hashes": split_hashes})
    (out / "config.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    (out / "environment.json").write_text(json.dumps({"python": platform.python_version(), "torch": torch.__version__, "device": str(device), "git_commit": "unavailable_no_git_repository"}, indent=2) + "\n", encoding="utf-8")
    manifest = {"bootstrap_checkpoint_sha256": bootstrap_hash, "config_sha256": config_hash, "projection_sha256": tensor_hash(projection), "split_hashes": split_hashes, "server_state": ["C_bar", "G_bar"], "local_only_state": ["delta", "personal", "personal_basis", "query_caches", "error_statistics"], "round_state_hashes": state_hashes}
    (out / "state_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    result = {"status": "pass", "result_status": result_status, "method": "horu", "dataset": "ucihar", "num_clients": len(states), "rounds": config.rounds, "bootstrap_checkpoint_sha256": bootstrap_hash, "final": rounds[-1], "personalized_pooled_test_accuracy": rounds[-1]["personalized_pooled_test_accuracy"], "output_files": ["result.json", "config.resolved.yaml", "environment.json", "round_metrics.csv", "client_metrics.csv", "timing_samples.csv", "communication.csv", "state_manifest.json", "checkpoints/latest.pt"]}
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); return result


def run_federated(config: FederatedConfig | HoruBootstrapConfig | HoruRoundConfig, data_root: str | Path, output: str | Path, device_override: str | None = None, overwrite: bool = False, resume: bool = False, bootstrap_checkpoint: str | Path | None = None) -> dict:
    if isinstance(config, HoruBootstrapConfig):
        if resume: raise ValueError("HoRU bootstrap has no round state to resume")
        return _run_horu_bootstrap(config, data_root, output, device_override, overwrite)
    if isinstance(config, HoruRoundConfig):
        return _run_horu_rounds(config, data_root, output, device_override, overwrite, resume, bootstrap_checkpoint)
    if config.method == "hyperfeel":
        return _run_hyperfeel(config, data_root, output, device_override, overwrite, resume)
    return _run_fedhdc(config, data_root, output, device_override, overwrite, resume)
