"""Shared-cache T006/T007 experiment runner.

This runner deliberately has no dataset-specific training branches: every
method receives the exact tensors and sample ids persisted by ``federated``.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import torch
import yaml

from ..datasets.federated import ClientData, FederatedDataset, load_federated, tensor_hash
from ..hdc.encoder import NonlinearEncoder, make_projection
from ..hdc.prototype import PrototypeMemory
from ..horu.bootstrap import bootstrap_horu
from ..horu.inference import coefficient_gram, predict
from ..horu.synchronization import absorb_shared, aggregate_shared
from ..horu.training import error_statistics, train_client
from ..methods import fedhdc, hyperfeel
from ..runtime import resolve_device


def _cap_train(dataset: FederatedDataset, maximum: int = 50_000) -> tuple[dict[str, ClientData], str]:
    """Apply the task-specified global cap, preserving client/class proportions.

    The cache itself is immutable; the selected ids are recorded in every run.
    """
    total = sum(c.train_y.numel() for c in dataset.clients.values())
    if total <= maximum:
        clients = dataset.clients
        return clients, hashlib.sha256(b"uncapped:" + dataset.split_hash().encode()).hexdigest()
    entries = [(cid, label, torch.nonzero(c.train_y == label).flatten())
               for cid, c in dataset.clients.items() for label in range(dataset.num_classes)
               if torch.any(c.train_y == label)]
    sizes = [int(ix.numel()) for _, _, ix in entries]
    raw = [size * maximum / total for size in sizes]
    take = [min(size, int(x)) for size, x in zip(sizes, raw)]
    remaining = maximum - sum(take)
    # Deterministic largest-remainder allocation; lexical entry order breaks ties.
    for i in sorted(range(len(entries)), key=lambda i: (raw[i] - take[i], entries[i][0], entries[i][1]), reverse=True):
        if not remaining: break
        if take[i] < sizes[i]: take[i] += 1; remaining -= 1
    selected: dict[str, list[torch.Tensor]] = {cid: [] for cid in dataset.clients}
    for (cid, label, ix), n in zip(entries, take):
        if n:
            g = torch.Generator().manual_seed(20260723 + label * 104729 + int(hashlib.sha256(cid.encode()).hexdigest()[:8], 16))
            selected[cid].append(ix[torch.randperm(ix.numel(), generator=g)[:n]].sort().values)
    clients = {}
    for cid, c in dataset.clients.items():
        ix = torch.cat(selected[cid]).sort().values
        clients[cid] = ClientData(c.train_x[ix], c.train_y[ix], c.test_x, c.test_y, c.train_ids[ix], c.test_ids)
    cap_hash = hashlib.sha256(b"".join(cid.encode() + tensor_hash(c.train_ids).encode() for cid, c in clients.items())).hexdigest()
    return clients, cap_hash


def _encode(dataset: FederatedDataset, clients: dict[str, ClientData], hd_dim: int, seed: int, device: torch.device) -> tuple[dict[str, dict], str]:
    projection = make_projection(dataset.num_features, hd_dim, seed).to(device)
    encoder = NonlinearEncoder(projection)
    encoded = {cid: {"train_h": encoder.encode(c.train_x.to(device)), "train_y": c.train_y.to(device),
                     "test_h": encoder.encode(c.test_x.to(device)), "test_y": c.test_y.to(device)}
               for cid, c in clients.items()}
    return encoded, tensor_hash(projection)


def _pooled(models: dict[str, torch.Tensor], clients: dict[str, dict]) -> float:
    predictions = torch.cat([PrototypeMemory(models[cid]).predict(c["test_h"], "dot") for cid, c in clients.items()])
    labels = torch.cat([c["test_y"] for c in clients.values()])
    return float((predictions == labels).float().mean().item())


def _run_fedhdc(clients: dict[str, dict], classes: int, cfg: dict) -> tuple[float, list[dict]]:
    local = {cid: fedhdc.bundled_model(c["train_h"], c["train_y"], classes) for cid, c in clients.items()}
    global_model = fedhdc.weighted_aggregate(list(local.values()), [int(c["train_y"].numel()) for c in clients.values()])
    rows = []
    for rnd in range(cfg["rounds"]):
        updated = {}
        for position, (cid, c) in enumerate(clients.items()):
            model = global_model.clone()
            for epoch in range(cfg["local_epochs"]):
                g = torch.Generator().manual_seed(cfg["seed"] + position * 1009 + rnd * 100_003 + epoch)
                order = torch.randperm(c["train_y"].numel(), generator=g).to(model.device)
                fedhdc.train_batches(model, c["train_h"][order], c["train_y"][order], cfg["hd_learning_rate"], cfg["batch_size"])
            updated[cid] = model
        global_model = fedhdc.weighted_aggregate(list(updated.values()), [int(c["train_y"].numel()) for c in clients.values()])
        rows.append({"round": rnd + 1, "pooled_client_test_accuracy": _pooled({cid: global_model for cid in clients}, clients)})
    return rows[-1]["pooled_client_test_accuracy"], rows


def _run_hyperfeel(clients: dict[str, dict], classes: int, cfg: dict) -> tuple[float, list[dict]]:
    local = [hyperfeel.bundled_model(c["train_h"], c["train_y"], classes) for c in clients.values()]
    central = hyperfeel.sum_deltas(local)
    personalized = {cid: central.clone() for cid in clients}
    previous_delta = torch.zeros_like(central); previous_weights = {cid: torch.zeros(classes, device=central.device) for cid in clients}
    rows = []
    for rnd in range(cfg["rounds"]):
        deltas = []
        for position, (cid, c) in enumerate(clients.items()):
            memory = personalized[cid].clone()
            hyperfeel.apply_personalization(memory, previous_delta, previous_weights[cid], cfg["hd_learning_rate"])
            total_delta = torch.zeros_like(memory); errors = torch.zeros(classes, device=memory.device); counts = torch.zeros(classes, device=memory.device)
            for epoch in range(cfg["local_epochs"]):
                g = torch.Generator().manual_seed(cfg["seed"] + position * 1009 + rnd * 100_003 + epoch)
                order = torch.randperm(c["train_y"].numel(), generator=g).to(memory.device)
                delta, e, n, _ = hyperfeel.retrain_batches(
                    memory, c["train_h"][order], c["train_y"][order],
                    cfg["hd_learning_rate"], cfg["batch_size"],
                )
                total_delta.add_(delta); errors.add_(e); counts.add_(n)
            personalized[cid] = memory; previous_weights[cid] = hyperfeel.personalization_weights(errors, counts); deltas.append(total_delta)
        previous_delta = hyperfeel.sum_deltas(deltas)
        rows.append({"round": rnd + 1, "pooled_client_test_accuracy": _pooled(personalized, clients)})
    return rows[-1]["pooled_client_test_accuracy"], rows


def _run_horu(clients: dict[str, dict], classes: int, cfg: dict) -> tuple[float, list[dict]]:
    numeric = {i: client for i, client in enumerate(clients.values())}
    states, common, global_basis, _, _, _ = bootstrap_horu(numeric, cfg["horu"]["common_rank"], cfg["horu"]["global_rank"], cfg["horu"]["personal_rank"], "full_svd", classes)
    grams = {cid: coefficient_gram(common, global_basis, state.personal_basis) for cid, state in states.items()}
    rows = []
    for rnd in range(cfg["rounds"]):
        class_stats = {}
        for cid, state in states.items():
            train_client(state, cfg["local_epochs"], cfg["batch_size"], cfg["horu"]["eta_shared"], cfg["horu"]["eta_personal"], grams[cid], cfg["seed"], rnd)
            counts, errors, _, _, _ = error_statistics(state, grams[cid], classes)
            class_stats[cid] = (counts, errors)
        shared_common, shared_global, _ = aggregate_shared(list(states.values()))
        for cid, state in states.items():
            counts, errors = class_stats[cid]
            absorb_shared(
                state,
                shared_common,
                shared_global,
                counts,
                errors,
                cfg["horu"]["eta_global"],
                cfg["horu"].get("gate_alpha", 1.0),
                cfg["horu"].get("gate_min", 0.1),
                cfg["horu"].get("gate_max", 0.9),
            )
        predictions, labels = [], []
        for cid, state in states.items():
            cache = state.test_cache
            predictions.extend(predict(cache["z_c"][i], cache["z_g"][i], cache["z_p"][i], state.common, state.global_coefficients, state.delta, state.personal, grams[cid]) for i in range(state.test_labels.numel()))
            labels.append(state.test_labels)
        y = torch.cat(labels); accuracy = float((torch.tensor(predictions, device=y.device) == y).float().mean().item())
        rows.append({"round": rnd + 1, "pooled_client_test_accuracy": accuracy})
    return rows[-1]["pooled_client_test_accuracy"], rows


def run_suite(config_path: str | Path, data_root: str | Path, output: str | Path) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")); root = Path(output); root.mkdir(parents=True, exist_ok=True)
    suite_config_sha256 = hashlib.sha256(yaml.safe_dump(cfg, sort_keys=True).encode()).hexdigest()
    runs = [{"dataset": d, "method": m, "seed": int(s)} for d in cfg["datasets"] for m in cfg["methods"] for s in cfg["seeds"]]
    (root / "summary").mkdir(exist_ok=True); (root / "summary" / "requested_runs.json").write_text(json.dumps(runs, indent=2) + "\n")
    completed = []
    for item in runs:
        out = root / "runs" / item["dataset"] / item["method"] / str(item["seed"])
        result_path = out / "result.json"
        if result_path.exists():
            prior = json.loads(result_path.read_text())
            if prior.get("suite_config_sha256") == suite_config_sha256:
                completed.append(prior); continue
        dataset = load_federated(data_root, item["dataset"]); selected, selection_hash = _cap_train(dataset)
        device = resolve_device(cfg.get("device", "auto")); encoded, projection_hash = _encode(dataset, selected, cfg["hd_dim"], item["seed"], device)
        run_cfg = dict(cfg); run_cfg["seed"] = item["seed"]
        started = time.perf_counter()
        if item["method"] == "fedhdc": accuracy, rows = _run_fedhdc(encoded, dataset.num_classes, run_cfg)
        elif item["method"] == "hyperfeel": accuracy, rows = _run_hyperfeel(encoded, dataset.num_classes, run_cfg)
        else: accuracy, rows = _run_horu(encoded, dataset.num_classes, run_cfg)
        out.mkdir(parents=True, exist_ok=True)
        (out / "round_metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
        result = {"status":"pass", "result_status":"VALID_EXPERIMENT_CANDIDATE", "dataset":dataset.name, "method":item["method"], "seed":item["seed"], "num_clients":len(selected), "rounds":cfg["rounds"], "official_global_pooled_test_accuracy":accuracy, "evaluation_protocol":"pooled_client_test_accuracy", "final":rows[-1], "dataset_split_sha256":dataset.split_hash(), "train_selection_sha256":selection_hash, "projection_sha256":projection_hash, "train_samples":sum(c.train_y.numel() for c in selected.values()), "test_samples":sum(c.test_y.numel() for c in selected.values()), "device":str(device), "elapsed_seconds":time.perf_counter()-started, "parameter_provenance":"USER_SPECIFIED_T006_T007", "suite_config_sha256":suite_config_sha256}
        (out / "config.resolved.yaml").write_text(yaml.safe_dump(run_cfg, sort_keys=False))
        (out / "environment.json").write_text(json.dumps({"python":platform.python_version(), "torch":torch.__version__, "device":str(device)}, indent=2)+"\n")
        result_path.write_text(json.dumps(result, indent=2)+"\n"); completed.append(result)
    return {"requested_runs":len(runs), "completed_now":len(completed)}
