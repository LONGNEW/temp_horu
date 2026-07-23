from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import time
import torch

from ..data import ClientData
from ..encoder import BaseHDEncoder
from ..federated import ClientState, FederatedMethod
from ..local_update import LocalHDUpdater
from ..memory import ClassMemory
from ..similarity import SimilarityMetric, similarity_scores


@dataclass
class FedHDCMethod(FederatedMethod):
    encoder: BaseHDEncoder
    updater: LocalHDUpdater
    num_classes: int
    metric: SimilarityMetric = "cos"
    debug: bool = False
    cache_train_hv: bool = False
    enable_system_profiling: bool = False
    trace_rounds: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        self.global_memory: ClassMemory | None = None
        self._round = 0
        self._current_round_system_rows: list[dict[str, Any]] = []

    def _should_trace_round(self, round_index: int) -> bool:
        trace_rounds = {int(value) for value in self.trace_rounds}
        if not trace_rounds:
            return True
        return int(round_index) in trace_rounds

    def _init_client_state_impl(
        self,
        client: ClientData,
        *,
        profile: bool,
    ) -> tuple[ClientState, dict[str, Any]]:
        total_started = time.perf_counter()

        encode_started = time.perf_counter()
        x_hv = self.encoder.encode(client.x_train)
        init_encode_train_ms = (time.perf_counter() - encode_started) * 1000.0

        prototype_started = time.perf_counter()
        local_memory = ClassMemory.from_encoded(
            x_hv,
            client.y_train.to(x_hv.device),
            self.num_classes,
        ).normalize_()
        init_prototype_build_ms = (time.perf_counter() - prototype_started) * 1000.0

        cache_store_ms = 0.0
        extras: dict[str, Any] = {}
        if self.cache_train_hv:
            cache_started = time.perf_counter()
            extras["cached_train_hv"] = x_hv.detach().cpu().clone()
            extras["cached_train_y"] = client.y_train.detach().cpu().clone()
            cache_store_ms = (time.perf_counter() - cache_started) * 1000.0

        if self.global_memory is None:
            self.global_memory = local_memory.clone()

        state = ClientState(memory=local_memory.weight.clone(), extras=extras)
        if not profile:
            return state, {}
        return state, {
            "init_client_state_ms": (time.perf_counter() - total_started) * 1000.0,
            "init_encode_train_ms": init_encode_train_ms,
            "init_prototype_build_ms": init_prototype_build_ms,
            "init_cache_store_ms": cache_store_ms,
        }

    def init_client_state(self, client: ClientData) -> ClientState:
        state, _ = self._init_client_state_impl(client, profile=False)
        return state

    def profiled_init_client_state(self, client: ClientData) -> tuple[ClientState, dict[str, Any]]:
        return self._init_client_state_impl(client, profile=True)

    def client_step(self, client: ClientData, state: ClientState) -> tuple[dict[str, Any], ClientState]:
        assert self.global_memory is not None
        round_index = int(self._round) + 1
        trace_round = self._should_trace_round(round_index)
        client_started = time.perf_counter()

        encode_started = time.perf_counter()
        extras = {} if state.extras is None else dict(state.extras)
        cached_train_hv = extras.get("cached_train_hv")
        cached_train_y = extras.get("cached_train_y")
        cache_hit = bool(isinstance(cached_train_hv, torch.Tensor) and isinstance(cached_train_y, torch.Tensor))
        if cache_hit:
            x_hv = cached_train_hv.to(self.encoder.device)
            y_train = cached_train_y.to(self.encoder.device).long()
        else:
            x_hv = self.encoder.encode(client.x_train)
            y_train = client.y_train.to(x_hv.device)
        encode_ms = (time.perf_counter() - encode_started) * 1000.0

        materialize_started = time.perf_counter()
        base_memory = ClassMemory(weight=self.global_memory.weight.clone())
        materialize_ms = (time.perf_counter() - materialize_started) * 1000.0

        updated_memory, updater_metrics = self.updater.profiled_step(base_memory, x_hv, y_train)
        delta = updated_memory.weight - base_memory.weight
        if self.debug:
            delta_norm = torch.linalg.norm(delta).item()
            base_norm = torch.linalg.norm(base_memory.weight).item()
            updated_norm = torch.linalg.norm(updated_memory.weight).item()
            changed = int(torch.count_nonzero(delta).item())
            print(
                f"[debug][fedhdc][round={self._round + 1}][client={client.client_id}] "
                f"base_norm={base_norm:.4f} updated_norm={updated_norm:.4f} "
                f"delta_norm={delta_norm:.4f} changed={changed}"
            )
        payload_pack_started = time.perf_counter()
        payload = {"memory": updated_memory.weight.detach().clone()}
        next_state = ClientState(memory=updated_memory.weight.detach().clone(), extras=extras)
        payload_pack_ms = (time.perf_counter() - payload_pack_started) * 1000.0

        if self.enable_system_profiling and trace_round:
            payload_bytes = float(payload["memory"].numel() * payload["memory"].element_size())
            self._current_round_system_rows.append(
                {
                    "round": round_index,
                    "client_id": client.client_id,
                    "materialize_pending_update_ms": materialize_ms,
                    "encode_train_ms": encode_ms,
                    "local_update_ms": float(updater_metrics["updater_step_ms"]),
                    "payload_pack_ms": payload_pack_ms,
                    "eval_ms": 0.0,
                    "train_samples": int(client.y_train.numel()),
                    "test_samples": int(client.y_test.numel()),
                    "round_comm_bytes": payload_bytes,
                    "cache_hit": 1.0 if cache_hit else 0.0,
                    "clone_global_memory_ms": materialize_ms,
                    "updater_shuffle_ms": float(updater_metrics["updater_shuffle_ms"]),
                    "updater_batch_slice_ms": float(updater_metrics["updater_batch_slice_ms"]),
                    "updater_similarity_ms": float(updater_metrics["updater_similarity_ms"]),
                    "updater_error_update_ms": float(updater_metrics["updater_error_update_ms"]),
                    "updater_normalize_ms": float(updater_metrics["updater_normalize_ms"]),
                    "updater_wrong_batches": float(updater_metrics["updater_wrong_batches"]),
                    "updater_wrong_samples": float(updater_metrics["updater_wrong_samples"]),
                    "client_e2e_round_ms": (time.perf_counter() - client_started) * 1000.0,
                    "system_granularity": "internal_stage_breakdown",
                }
            )
        return payload, next_state

    def server_step(self, payloads: list[dict[str, Any]]) -> None:
        if not payloads:
            self._round += 1
            return
        aggregated_memory = torch.zeros_like(payloads[0]["memory"])
        for payload in payloads:
            aggregated_memory.add_(payload["memory"])
        self.global_memory = ClassMemory(weight=aggregated_memory).normalize_()
        if self.debug:
            global_norm = torch.linalg.norm(self.global_memory.weight).item()
            print(
                f"[debug][fedhdc][round={self._round + 1}][server] "
                f"num_payloads={len(payloads)} global_norm={global_norm:.4f}"
            )
        self._round += 1

    def collect_round_artifacts(
        self,
        clients: list[ClientData],
        states: list[ClientState],
        payloads: list[dict[str, Any]],
        *,
        round_index: int,
        selected_indices: list[int],
        server_step_ms: float,
        round_runtime_sec: float,
    ) -> dict[str, Any] | None:
        del clients, states, payloads, round_runtime_sec
        if not self.enable_system_profiling or not self._should_trace_round(round_index):
            self._current_round_system_rows = []
            return None

        amortized_server_ms = float(server_step_ms) / max(len(selected_indices), 1)
        system_rows: list[dict[str, Any]] = []
        for row in self._current_round_system_rows:
            item = dict(row)
            item["server_step_ms_total"] = float(server_step_ms)
            item["amortized_server_ms"] = amortized_server_ms
            item["client_e2e_round_ms"] = float(item["client_e2e_round_ms"]) + amortized_server_ms
            system_rows.append(item)
        self._current_round_system_rows = []
        return {"system_metrics": system_rows}

    def evaluate(self, clients: list[ClientData], states: list[ClientState]) -> dict[str, float]:
        assert self.global_memory is not None
        global_accs = []
        global_correct = 0
        global_total = 0
        local_test_accs = []
        local_train_accs = []
        for client, state in zip(clients, states):
            x_hv = self.encoder.encode(client.x_test)
            pred = similarity_scores(x_hv, self.global_memory.weight, self.metric).argmax(dim=1)
            acc = (pred.cpu() == client.y_test.cpu()).float().mean().item()
            global_accs.append(acc)
            global_correct += int((pred.cpu() == client.y_test.cpu()).sum().item())
            global_total += int(client.y_test.numel())

            local_test_pred = similarity_scores(x_hv, state.memory, self.metric).argmax(dim=1)
            local_test_acc = (local_test_pred.cpu() == client.y_test.cpu()).float().mean().item()
            local_test_accs.append(local_test_acc)

            x_train_hv = self.encoder.encode(client.x_train)
            local_train_pred = similarity_scores(x_train_hv, state.memory, self.metric).argmax(dim=1)
            local_train_acc = (local_train_pred.cpu() == client.y_train.cpu()).float().mean().item()
            local_train_accs.append(local_train_acc)
        return {
            # `global_test_accuracy` is the sample-weighted score of the single
            # synchronized FedHDC memory over every client test example.
            # Keep `mean_global_accuracy` for backward-compatible client means.
            "global_test_accuracy": float(global_correct) / max(global_total, 1),
            "mean_global_accuracy": sum(global_accs) / max(len(global_accs), 1),
            "mean_local_test_accuracy": sum(local_test_accs) / max(len(local_test_accs), 1),
            "mean_local_train_accuracy": sum(local_train_accs) / max(len(local_train_accs), 1),
        }
