from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import time
import torch

from ..data import ClientData
from ..encoder import BaseHDEncoder
from ..federated import ClientState, FederatedMethod
from ..memory import ClassMemory
from ..similarity import SimilarityMetric, similarity_scores


@dataclass
class HyperFeelMethod(FederatedMethod):
    encoder: BaseHDEncoder
    num_classes: int
    local_epochs: int = 1
    batch_size: int = 32
    lr: float = 1.0
    metric: SimilarityMetric = "cos"
    debug: bool = False
    cache_train_hv: bool = False
    enable_system_profiling: bool = False
    trace_rounds: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        self.global_prototypes = ClassMemory.zeros(
            self.num_classes,
            self.encoder.hd_dim,
            device=self.encoder.device,
        )
        self.global_delta = torch.zeros_like(self.global_prototypes.weight)
        self._round = 0
        self._current_round_system_rows: list[dict[str, Any]] = []
        self._last_server_stage_metrics: dict[str, float] = {
            "server_delta_stack_ms": 0.0,
            "server_delta_sum_ms": 0.0,
            "server_apply_global_ms": 0.0,
        }

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
        local_prototypes = ClassMemory.from_encoded(
            x_hv,
            client.y_train.to(x_hv.device),
            self.num_classes,
        ).normalize_()
        init_prototype_build_ms = (time.perf_counter() - prototype_started) * 1000.0

        extras = {
            "class_counts": torch.bincount(
                client.y_train.to(x_hv.device).long(),
                minlength=self.num_classes,
            ).to(torch.float32),
            "class_errors": torch.zeros(self.num_classes, dtype=torch.float32, device=x_hv.device),
        }
        cache_store_ms = 0.0
        if self.cache_train_hv:
            cache_started = time.perf_counter()
            extras["cached_train_hv"] = x_hv.detach().cpu().clone()
            extras["cached_train_y"] = client.y_train.detach().cpu().clone()
            cache_store_ms = (time.perf_counter() - cache_started) * 1000.0

        state = ClientState(memory=local_prototypes.weight.clone(), extras=extras)
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

    def profiled_bootstrap(
        self,
        clients: list[ClientData],
        states: list[ClientState],
    ) -> tuple[list[ClientState], dict[str, Any]]:
        # Paper-faithful initialization:
        # clients build local class vectors first, server accumulates them into a central AM,
        # then the central model is downloaded back as the initial local model.
        bootstrap_started = time.perf_counter()
        server_aggregate_started = time.perf_counter()
        aggregated_memory = torch.zeros_like(states[0].memory)
        for state in states:
            aggregated_memory.add_(state.memory)
        self.global_prototypes = ClassMemory(weight=aggregated_memory).normalize_()
        self.global_delta.zero_()
        bootstrap_server_aggregate_ms = (time.perf_counter() - server_aggregate_started) * 1000.0
        if self.debug:
            print(
                f"[debug][hyperfeel][bootstrap][server] "
                f"num_clients={len(states)} global_proto_norm={torch.linalg.norm(self.global_prototypes.weight).item():.4f}"
            )
        new_states: list[ClientState] = []
        client_profiles: list[dict[str, Any]] = []
        per_client_bootstrap_server_ms = bootstrap_server_aggregate_ms / max(len(states), 1)
        for client, state in zip(clients, states):
            client_started = time.perf_counter()
            if self.debug:
                local_norm = torch.linalg.norm(state.memory).item()
                print(
                    f"[debug][hyperfeel][bootstrap][client={client.client_id}] "
                    f"local_init_norm={local_norm:.4f} downloaded_global_norm={torch.linalg.norm(self.global_prototypes.weight).item():.4f}"
                )
            new_states.append(
                ClientState(
                    memory=self.global_prototypes.weight.clone(),
                    extras=state.extras,
                )
            )
            client_profiles.append(
                {
                    "client_id": client.client_id,
                    "bootstrap_projection_ms": (time.perf_counter() - client_started) * 1000.0,
                    "bootstrap_server_aggregate_ms": per_client_bootstrap_server_ms,
                }
            )
        return new_states, {
            "total_bootstrap_ms": (time.perf_counter() - bootstrap_started) * 1000.0,
            "bootstrap_server_aggregate_ms": bootstrap_server_aggregate_ms,
            "client_profiles": client_profiles,
        }

    def bootstrap(self, clients: list[ClientData], states: list[ClientState]) -> list[ClientState]:
        states, _ = self.profiled_bootstrap(clients, states)
        return states

    def client_step(self, client: ClientData, state: ClientState) -> tuple[dict[str, Any], ClientState]:
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
            y = cached_train_y.to(self.encoder.device).long()
        else:
            x_hv = self.encoder.encode(client.x_train)
            y = client.y_train.to(x_hv.device).long()
        encode_ms = (time.perf_counter() - encode_started) * 1000.0

        local_prototypes = state.memory.clone()
        before_local = local_prototypes.clone()

        class_counts = extras["class_counts"].to(x_hv.device)
        prev_class_errors = extras["class_errors"].to(x_hv.device)
        class_errors = torch.zeros(self.num_classes, dtype=torch.float32, device=x_hv.device)
        absorb_ms = 0.0
        similarity_ms = 0.0
        error_update_ms = 0.0
        normalize_ms = 0.0
        wrong_batches = 0.0
        wrong_samples = 0.0

        # Paper-faithful personalization update:
        # C_ij = C_ij + (error_ij / cnt_ij) * lr * Delta_j
        if torch.count_nonzero(self.global_delta).item() > 0:
            absorb_started = time.perf_counter()
            ratio = (prev_class_errors / class_counts.clamp_min(1.0)).unsqueeze(1)
            local_prototypes = local_prototypes + ratio * self.lr * self.global_delta.to(local_prototypes.device)
            local_prototypes = ClassMemory(weight=local_prototypes).normalize_().weight
            absorb_ms = (time.perf_counter() - absorb_started) * 1000.0

        delta = torch.zeros_like(local_prototypes)
        num_samples = x_hv.shape[0]
        num_classes = local_prototypes.shape[0]
        eye = torch.eye(num_classes, dtype=torch.bool, device=x_hv.device)

        for _ in range(int(self.local_epochs)):
            order = torch.randperm(num_samples, device=x_hv.device)
            for start in range(0, num_samples, self.batch_size):
                idx = order[start:start + self.batch_size]
                x_batch = x_hv.index_select(0, idx)
                y_batch = y.index_select(0, idx)
                similarity_started = time.perf_counter()
                pred = similarity_scores(x_batch, local_prototypes, self.metric).argmax(dim=1)
                similarity_ms += (time.perf_counter() - similarity_started) * 1000.0
                wrong = pred != y_batch
                if not torch.any(wrong):
                    continue

                wrong_batches += 1.0
                wrong_samples += float(wrong.sum().item())
                y_wrong = y_batch[wrong]
                class_errors += torch.bincount(y_wrong, minlength=self.num_classes).to(torch.float32)

                # Batch implementation of the paper's single-sample online update.
                update_started = time.perf_counter()
                wrong_mask = wrong.repeat(num_classes, 1).T
                correct_update = wrong_mask & eye[y_batch]
                wrong_update = wrong_mask & eye[pred]
                batch_delta = (correct_update.float() - wrong_update.float()).T @ x_batch
                local_prototypes.add_(batch_delta, alpha=self.lr)
                delta.add_(batch_delta, alpha=self.lr)
                error_update_ms += (time.perf_counter() - update_started) * 1000.0
                normalize_started = time.perf_counter()
                local_prototypes = ClassMemory(weight=local_prototypes).normalize_().weight
                normalize_ms += (time.perf_counter() - normalize_started) * 1000.0

        if self.debug:
            train_pred = similarity_scores(x_hv, local_prototypes, self.metric).argmax(dim=1)
            train_acc = (train_pred == y).float().mean().item()
            local_shift = local_prototypes - before_local
            print(
                f"[debug][hyperfeel][round={self._round + 1}][client={client.client_id}] "
                f"local_shift_norm={torch.linalg.norm(local_shift).item():.4f} "
                f"delta_norm={torch.linalg.norm(delta).item():.4f} "
                f"errors={int(class_errors.sum().item())} train_acc={train_acc:.4f}"
            )

        payload_pack_started = time.perf_counter()
        payload = {"delta": delta.detach().clone()}
        next_state = ClientState(
            memory=local_prototypes.detach().clone(),
            extras={
                "class_counts": class_counts.detach().clone(),
                "class_errors": class_errors.detach().clone(),
                **(
                    {
                        "cached_train_hv": extras["cached_train_hv"].detach().clone(),
                        "cached_train_y": extras["cached_train_y"].detach().clone(),
                    }
                    if cache_hit
                    else {}
                ),
            },
        )
        payload_pack_ms = (time.perf_counter() - payload_pack_started) * 1000.0

        if self.enable_system_profiling and trace_round:
            payload_bytes = float(payload["delta"].numel() * payload["delta"].element_size())
            self._current_round_system_rows.append(
                {
                    "round": round_index,
                    "client_id": client.client_id,
                    "encode_train_ms": encode_ms,
                    "materialize_pending_update_ms": absorb_ms,
                    "hyperfeel_absorb_ms": absorb_ms,
                    "local_update_ms": similarity_ms + error_update_ms + normalize_ms,
                    "payload_pack_ms": payload_pack_ms,
                    "eval_ms": 0.0,
                    "train_samples": int(client.y_train.numel()),
                    "test_samples": int(client.y_test.numel()),
                    "round_comm_bytes": payload_bytes,
                    "cache_hit": 1.0 if cache_hit else 0.0,
                    "hyperfeel_similarity_ms": similarity_ms,
                    "hyperfeel_error_update_ms": error_update_ms,
                    "hyperfeel_normalize_ms": normalize_ms,
                    "hyperfeel_wrong_batches": wrong_batches,
                    "hyperfeel_wrong_samples": wrong_samples,
                    "client_e2e_round_ms": (time.perf_counter() - client_started) * 1000.0,
                    "system_granularity": "internal_stage_breakdown",
                }
            )
        return payload, next_state

    def server_step(self, payloads: list[dict[str, Any]]) -> None:
        self._last_server_stage_metrics = {
            "server_delta_stack_ms": 0.0,
            "server_delta_sum_ms": 0.0,
            "server_apply_global_ms": 0.0,
        }
        if not payloads:
            self.global_delta.zero_()
            return
        sum_started = time.perf_counter()
        aggregated_delta = torch.zeros_like(payloads[0]["delta"])
        for payload in payloads:
            aggregated_delta.add_(payload["delta"])
        self.global_delta = aggregated_delta
        self._last_server_stage_metrics["server_delta_sum_ms"] = (time.perf_counter() - sum_started) * 1000.0
        apply_started = time.perf_counter()
        self.global_prototypes.weight.add_(self.global_delta)
        self.global_prototypes.normalize_()
        self._last_server_stage_metrics["server_apply_global_ms"] = (time.perf_counter() - apply_started) * 1000.0
        if self.debug:
            print(
                f"[debug][hyperfeel][round={self._round + 1}][server] "
                f"num_payloads={len(payloads)} global_delta_norm={torch.linalg.norm(self.global_delta).item():.4f} "
                f"global_proto_norm={torch.linalg.norm(self.global_prototypes.weight).item():.4f}"
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
        stage_metrics = dict(self._last_server_stage_metrics)
        system_rows: list[dict[str, Any]] = []
        for row in self._current_round_system_rows:
            item = dict(row)
            item["server_step_ms_total"] = float(server_step_ms)
            item["amortized_server_ms"] = amortized_server_ms
            item["client_e2e_round_ms"] = float(item["client_e2e_round_ms"]) + amortized_server_ms
            for key, value in stage_metrics.items():
                item[key] = float(value)
                item[f"amortized_{key}"] = float(value) / max(len(selected_indices), 1)
            system_rows.append(item)
        self._current_round_system_rows = []
        return {"system_metrics": system_rows}

    def evaluate(self, clients: list[ClientData], states: list[ClientState]) -> dict[str, float]:
        personalized_accs = []
        local_train_accs = []
        for client, state in zip(clients, states):
            x_hv = self.encoder.encode(client.x_test)
            pred = similarity_scores(x_hv, state.memory, self.metric).argmax(dim=1)
            acc = (pred.cpu() == client.y_test.cpu()).float().mean().item()
            personalized_accs.append(acc)

            x_train_hv = self.encoder.encode(client.x_train)
            local_train_pred = similarity_scores(x_train_hv, state.memory, self.metric).argmax(dim=1)
            local_train_acc = (local_train_pred.cpu() == client.y_train.cpu()).float().mean().item()
            local_train_accs.append(local_train_acc)
        return {
            "mean_personalized_accuracy": sum(personalized_accs) / max(len(personalized_accs), 1),
            "mean_local_test_accuracy": sum(personalized_accs) / max(len(personalized_accs), 1),
            "mean_local_train_accuracy": sum(local_train_accs) / max(len(local_train_accs), 1),
        }
