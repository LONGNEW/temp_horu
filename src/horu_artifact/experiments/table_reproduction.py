"""CPU reproduction harness for paper Tables I, II, and III."""

from __future__ import annotations

import copy
import csv
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import torch

from ..datasets.controlled_systems import ControlledSystemsFixture, load_fixture
from ..horu.bootstrap import bootstrap_horu
from ..horu.inference import coefficient_gram
from ..horu.synchronization import absorb_shared, aggregate_shared
from ..horu.training import error_statistics, train_client
from ..methods import fedhdc, hyperfeel


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _summary(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    q1, median, q3 = torch.quantile(tensor, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)).tolist()
    return {"median_ms": median, "q1_ms": q1, "q3_ms": q3, "iqr_ms": q3 - q1}


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _clients(fixture: ControlledSystemsFixture) -> dict[int, dict[str, torch.Tensor]]:
    empty_h = fixture.train_h[:0]
    empty_y = fixture.train_y[:0]
    return {
        cid: {"train_h": fixture.train_h, "train_y": fixture.train_y, "test_h": empty_h, "test_y": empty_y}
        for cid in range(fixture.config.clients)
    }


def _bootstrap_once(fixture: ControlledSystemsFixture):
    cfg = fixture.config
    return bootstrap_horu(_clients(fixture), cfg.common_rank, cfg.global_rank, cfg.personal_rank, "full_svd", cfg.classes)


def _table_i_sample(fixture: ControlledSystemsFixture) -> tuple[dict[str, float], tuple]:
    result = _bootstrap_once(fixture)
    states, common, global_basis, server, clients, _ = result
    sample = {
        "server_common_global_basis_ms": server["server_common_global_basis_ms"],
        "server_client_hv_projection_ms": server["server_client_hv_projection_ms"],
        "server_common_global_coefficients_ms": server["server_common_global_coefficients_ms"],
        "server_bootstrap_total_ms": server["server_bootstrap_total_ms"],
        "residual_construction_ms": _mean([row["residual_construction_ms"] for row in clients]),
        "personal_basis_svd_ms": _mean([row["personal_basis_svd_ms"] for row in clients]),
        "residual_coefficient_projection_ms": _mean([row["residual_coefficient_projection_ms"] for row in clients]),
        "query_coefficient_cache_ms": _mean([row["query_coefficient_cache_ms"] for row in clients]),
        "client_bootstrap_ms": _mean([row["client_bootstrap_ms"] for row in clients]),
    }
    return sample, (states, common, global_basis)


def _table_ii_horu_sample(snapshot: tuple, fixture: ControlledSystemsFixture, repeat: int) -> dict[str, float]:
    states = copy.deepcopy(snapshot[0])
    common, global_basis = snapshot[1], snapshot[2]
    cfg = fixture.config
    grams = {cid: coefficient_gram(common, global_basis, state.personal_basis) for cid, state in states.items()}
    similarity, update, final_prediction, statistics = [], [], [], []
    counts_by_client, errors_by_client = {}, {}
    for cid, state in states.items():
        _, timing = train_client(state, cfg.local_epochs, cfg.batch_size, 0.035, 0.035, grams[cid], cfg.seed, repeat)
        counts, errors, _, prediction_ms, statistics_ms = error_statistics(state, grams[cid], cfg.classes)
        similarity.append(timing["coefficient_similarity_ms"])
        update.append(timing["coefficient_update_ms"])
        final_prediction.append(prediction_ms)
        statistics.append(statistics_ms)
        counts_by_client[cid], errors_by_client[cid] = counts, errors
    aggregate_common, aggregate_global, server = aggregate_shared(list(states.values()))
    absorption = [
        absorb_shared(states[cid], aggregate_common, aggregate_global, counts_by_client[cid], errors_by_client[cid], 0.035)
        for cid in states
    ]
    result = {
        "local_similarity_ms": _mean(similarity),
        "local_coefficients_update_ms": _mean(update),
        "final_train_prediction_ms": _mean(final_prediction),
        "class_error_statistics_ms": _mean(statistics),
        "common_aggregation_ms": server["common_aggregation_ms"],
        "global_aggregation_ms": server["global_aggregation_ms"],
        "client_shared_branch_update_ms": _mean(absorption),
    }
    result["client_round_total_ms"] = sum(result[key] for key in ("local_similarity_ms", "local_coefficients_update_ms", "final_train_prediction_ms", "class_error_statistics_ms"))
    result["synchronization_step_total_ms"] = result["common_aggregation_ms"] + result["global_aggregation_ms"] + result["client_shared_branch_update_ms"]
    return result


def _fedhdc_sample(fixture: ControlledSystemsFixture) -> dict[str, float]:
    cfg = fixture.config
    models, similarity, update = [], [], []
    for _ in range(cfg.clients):
        model = fixture.initial_prototypes.clone()
        timing = {"similarity_ms": 0.0, "update_ms": 0.0}
        fedhdc.train_batches(model, fixture.train_h, fixture.train_y, 0.035, cfg.batch_size, timing)
        models.append(model); similarity.append(timing["similarity_ms"]); update.append(timing["update_ms"])
    begun = time.perf_counter_ns()
    fedhdc.weighted_aggregate(models, [cfg.samples_per_client] * cfg.clients)
    server_ms = (time.perf_counter_ns() - begun) / 1e6
    return {"local_round_ms": _mean(similarity) + _mean(update), "similarity_ms": _mean(similarity), "update_ms": _mean(update), "server_step_ms": server_ms, "uploaded_payload_bytes": cfg.classes * cfg.hd_dim * fixture.initial_prototypes.element_size()}


def _hyperfeel_sample(fixture: ControlledSystemsFixture) -> dict[str, float]:
    cfg = fixture.config
    deltas, similarity, update = [], [], []
    zero_delta = torch.zeros_like(fixture.initial_prototypes)
    zero_weights = torch.zeros(cfg.classes)
    for _ in range(cfg.clients):
        memory = fixture.initial_prototypes.clone()
        timing = {"similarity_ms": 0.0, "update_ms": 0.0}
        begun = time.perf_counter_ns()
        hyperfeel.apply_personalization(memory, zero_delta, zero_weights, 0.035)
        timing["update_ms"] += (time.perf_counter_ns() - begun) / 1e6
        delta, errors, counts, _ = hyperfeel.retrain_batches(
            memory, fixture.train_h, fixture.train_y, 0.035, cfg.batch_size,
            False, False, timing,
        )
        begun = time.perf_counter_ns()
        hyperfeel.personalization_weights(errors, counts)
        timing["update_ms"] += (time.perf_counter_ns() - begun) / 1e6
        deltas.append(delta); similarity.append(timing["similarity_ms"]); update.append(timing["update_ms"])
    begun = time.perf_counter_ns()
    hyperfeel.sum_deltas(deltas)
    server_ms = (time.perf_counter_ns() - begun) / 1e6
    return {"local_round_ms": _mean(similarity) + _mean(update), "similarity_ms": _mean(similarity), "update_ms": _mean(update), "server_step_ms": server_ms, "uploaded_payload_bytes": cfg.classes * cfg.hd_dim * fixture.initial_prototypes.element_size()}


def reproduce_tables(data_root: str | Path, output: str | Path, warmup: int = 5, repeats: int = 30, threads: int = 1) -> dict:
    if warmup < 0 or repeats <= 0 or threads <= 0:
        raise ValueError("warmup must be nonnegative and repeats/threads must be positive")
    fixture = load_fixture(data_root)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    table_i_raw, snapshot = [], None
    for iteration in range(warmup + repeats):
        sample, current_snapshot = _table_i_sample(fixture)
        if snapshot is None:
            snapshot = current_snapshot
        if iteration >= warmup:
            table_i_raw.append({"table": "I", "repeat": iteration - warmup, **sample})

    table_ii_raw, method_raw = [], []
    total = warmup + repeats
    methods = ("horu", "fedhdc", "hyperfeel")
    for iteration in range(total):
        ordered = methods[iteration % len(methods):] + methods[:iteration % len(methods)]
        samples = {}
        for method in ordered:
            if method == "horu":
                table_ii = _table_ii_horu_sample(snapshot, fixture, iteration)
                samples[method] = {"local_round_ms": table_ii["client_round_total_ms"], "similarity_ms": table_ii["local_similarity_ms"], "update_ms": table_ii["local_coefficients_update_ms"], "server_step_ms": table_ii["synchronization_step_total_ms"], "uploaded_payload_bytes": fixture.config.classes * (fixture.config.common_rank + fixture.config.global_rank) * fixture.initial_prototypes.element_size()}
            elif method == "fedhdc":
                samples[method] = _fedhdc_sample(fixture)
            else:
                samples[method] = _hyperfeel_sample(fixture)
        if iteration >= warmup:
            repeat = iteration - warmup
            table_ii_raw.append({"table": "II", "repeat": repeat, **table_ii})
            method_raw.extend({"table": "III", "repeat": repeat, "method": method, **samples[method]} for method in methods)

    raw = table_i_raw + table_ii_raw + method_raw
    _write_csv(out / "raw_timings.csv", raw)

    table_i_components = [
        ("Server", "Common/global basis computation", "server_common_global_basis_ms"),
        ("Server", "Projection of client hypervectors", "server_client_hv_projection_ms"),
        ("Server", "Common/global coefficients build", "server_common_global_coefficients_ms"),
        ("Server", "Server bootstrap total", "server_bootstrap_total_ms"),
        ("Client", "Residual class hypervectors build", "residual_construction_ms"),
        ("Client", "Personal basis computation", "personal_basis_svd_ms"),
        ("Client", "Projection of residual class hypervectors", "residual_coefficient_projection_ms"),
        ("Client", "Cached common/global/personal query coefficient build", "query_coefficient_cache_ms"),
        ("Client", "Client bootstrap wall time", "client_bootstrap_ms"),
    ]
    table_i = [{"scope": scope, "component": component, **_summary([row[field] for row in table_i_raw])} for scope, component, field in table_i_components]
    _write_csv(out / "table1.csv", table_i)

    table_ii_components = [
        ("Client", "Local similarity", "local_similarity_ms"),
        ("Client", "Local coefficients update", "local_coefficients_update_ms"),
        ("Client", "Final train dataset prediction", "final_train_prediction_ms"),
        ("Client", "Class-wise error statistics", "class_error_statistics_ms"),
        ("Client", "Client round total", "client_round_total_ms"),
        ("Server", "Aggregation of common coefficients", "common_aggregation_ms"),
        ("Server", "Aggregation of global coefficients", "global_aggregation_ms"),
        ("Client", "client-side shared branch update with class-wise update weights", "client_shared_branch_update_ms"),
        ("Mix", "synchronization step total", "synchronization_step_total_ms"),
    ]
    table_ii = [{"scope": scope, "component": component, **_summary([row[field] for row in table_ii_raw])} for scope, component, field in table_ii_components]
    _write_csv(out / "table2.csv", table_ii)

    table_iii = []
    for method in methods:
        rows = [row for row in method_raw if row["method"] == method]
        table_iii.append({"method": "HoRU (Ours)" if method == "horu" else ("FedHDC" if method == "fedhdc" else "HyperFeel"), "local_round_ms": _summary([row["local_round_ms"] for row in rows])["median_ms"], "similarity_ms": _summary([row["similarity_ms"] for row in rows])["median_ms"], "update_ms": _summary([row["update_ms"] for row in rows])["median_ms"], "server_step_ms": _summary([row["server_step_ms"] for row in rows])["median_ms"], "uploaded_payload_kb": rows[0]["uploaded_payload_bytes"] / 1000.0})
    _write_csv(out / "table3.csv", table_iii)

    cpu_model = "unknown"
    try:
        cpu_model = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    environment = {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(), "cpu_model": cpu_model, "torch_threads": threads, "torch_interop_threads": 1, "warmup": warmup, "repeats": repeats, "fixture": asdict(fixture.config)}
    (out / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    result = {"status": "pass", "outputs": ["raw_timings.csv", "table1.csv", "table2.csv", "table3.csv", "environment.json"], **environment}
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
