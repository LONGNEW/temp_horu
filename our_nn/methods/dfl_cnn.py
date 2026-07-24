from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import time
import torch

from our_hd.data import ClientData
from our_nn.federated import NNClientState, NNFederatedMethod
from our_nn.models import DFLFEMNISTCNN
from our_nn.train import average_state_dicts, detached_state_dict, evaluate_model, train_dfl_epoch


@dataclass
class DFLCNNMethod(NNFederatedMethod):
    num_classes: int
    cnn_hidden_dim: int
    local_epochs: int
    batch_size: int
    lr: float
    align_weight: float = 1.0
    disentangle_weight: float = 0.1
    lr_decay: float = 1.0
    optimizer_name: str = "sgd"
    momentum: float = 0.0
    weight_decay: float = 0.0
    dropout: float = 0.0
    input_preprocessing: str = "none"
    device: torch.device | str = "cpu"
    state_device: torch.device | str = "cpu"
    debug: bool = False
    enable_system_profiling: bool = False
    trace_rounds: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self.state_device = torch.device(self.state_device)
        self.global_model = self._new_model()
        self.current_lr = float(self.lr)
        self._round = 0
        self._current_round_system_rows: list[dict[str, Any]] = []
        self._last_server_stage_metrics: dict[str, float] = {
            "server_average_branch_ms": 0.0,
            "server_apply_branch_ms": 0.0,
        }

    def _should_trace_round(self, round_index: int) -> bool:
        trace_rounds = {int(value) for value in self.trace_rounds}
        if not trace_rounds:
            return True
        return int(round_index) in trace_rounds

    def _new_model(self) -> DFLFEMNISTCNN:
        return DFLFEMNISTCNN(
            self.num_classes,
            hidden_dim=self.cnn_hidden_dim,
            dropout=self.dropout,
            input_preprocessing=self.input_preprocessing,
        ).to(self.device)

    def init_client_state(self, client: ClientData) -> NNClientState:
        del client
        return NNClientState()

    def profiled_init_client_state(self, client: ClientData) -> tuple[NNClientState, dict[str, Any]]:
        started = time.perf_counter()
        state = self.init_client_state(client)
        return state, {"init_client_state_ms": (time.perf_counter() - started) * 1000.0}

    def client_step(self, client: ClientData, state: NNClientState) -> tuple[dict[str, Any], NNClientState]:
        round_index = int(self._round) + 1
        trace_round = self._should_trace_round(round_index)
        client_started = time.perf_counter()

        materialize_started = time.perf_counter()
        local_model = self._new_model()
        if state.personalized_state is not None:
            local_model.load_state_dict(state.personalized_state)
        else:
            local_model.load_state_dict(self.global_model.state_dict())
        personal_model_materialize_ms = (time.perf_counter() - materialize_started) * 1000.0

        sync_started = time.perf_counter()
        local_model.global_branch.load_state_dict(self.global_model.global_branch.state_dict())
        sync_global_branch_ms = (time.perf_counter() - sync_started) * 1000.0

        reference_started = time.perf_counter()
        global_invariant_state = detached_state_dict(self.global_model.global_branch, device=self.state_device)
        capture_global_reference_ms = (time.perf_counter() - reference_started) * 1000.0
        train_metrics = {"accuracy": 0.0}
        local_update_started = time.perf_counter()
        for _ in range(self.local_epochs):
            train_metrics = train_dfl_epoch(
                local_model,
                client.x_train,
                client.y_train,
                lr=self.current_lr,
                batch_size=self.batch_size,
                global_invariant_state=global_invariant_state,
                align_weight=self.align_weight,
                disentangle_weight=self.disentangle_weight,
                optimizer_name=self.optimizer_name,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        local_update_ms = (time.perf_counter() - local_update_started) * 1000.0

        if self.debug:
            print(
                f"[debug][dfl_cnn][round={self._round + 1}][client={client.client_id}] "
                f"personal_train_acc={train_metrics['accuracy']:.4f}"
            )

        payload_pack_started = time.perf_counter()
        payload = {
            "global_branch_state": detached_state_dict(local_model.global_branch, device=self.state_device),
            "num_samples": float(len(client.y_train)),
        }
        capture_personal_state_started = time.perf_counter()
        next_state = NNClientState(personalized_state=detached_state_dict(local_model, device=self.state_device))
        capture_personal_state_ms = (time.perf_counter() - capture_personal_state_started) * 1000.0
        payload_pack_ms = (time.perf_counter() - payload_pack_started) * 1000.0
        if self.enable_system_profiling and trace_round:
            payload_bytes = float(sum(t.numel() * t.element_size() for t in payload["global_branch_state"].values()))
            self._current_round_system_rows.append(
                {
                    "round": round_index,
                    "client_id": client.client_id,
                    "materialize_pending_update_ms": 0.0,
                    "encode_train_ms": 0.0,
                    "local_update_ms": local_update_ms,
                    "payload_pack_ms": payload_pack_ms,
                    "eval_ms": 0.0,
                    "train_samples": int(client.y_train.numel()),
                    "test_samples": int(client.y_test.numel()),
                    "round_comm_bytes": payload_bytes,
                    "personal_model_materialize_ms": personal_model_materialize_ms,
                    "sync_global_branch_ms": sync_global_branch_ms,
                    "capture_global_reference_ms": capture_global_reference_ms,
                    "dfl_local_update_ms": local_update_ms,
                    "capture_personal_state_ms": capture_personal_state_ms,
                    "client_e2e_round_ms": (time.perf_counter() - client_started) * 1000.0,
                    "system_granularity": "internal_stage_breakdown",
                }
            )
        return payload, next_state

    def server_step(self, payloads: list[dict[str, Any]]) -> None:
        self._last_server_stage_metrics = {
            "server_average_branch_ms": 0.0,
            "server_apply_branch_ms": 0.0,
        }
        if payloads:
            branch_states = [payload["global_branch_state"] for payload in payloads]
            weights = [payload["num_samples"] for payload in payloads]
            average_started = time.perf_counter()
            averaged_branch_state = average_state_dicts(branch_states, weights)
            self._last_server_stage_metrics["server_average_branch_ms"] = (time.perf_counter() - average_started) * 1000.0
            apply_started = time.perf_counter()
            self.global_model.global_branch.load_state_dict(averaged_branch_state)
            self._last_server_stage_metrics["server_apply_branch_ms"] = (time.perf_counter() - apply_started) * 1000.0
        self.current_lr *= float(self.lr_decay)
        self._round += 1

    def collect_round_artifacts(
        self,
        clients: list[ClientData],
        states: list[NNClientState],
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

    def evaluate(self, clients: list[ClientData], states: list[NNClientState]) -> dict[str, float]:
        personalized_test_accs = []
        personalized_train_accs = []
        for client, state in zip(clients, states):
            personal_model = self._new_model()
            if state.personalized_state is not None:
                personal_model.load_state_dict(state.personalized_state)
            else:
                personal_model.load_state_dict(self.global_model.state_dict())
            personal_model.global_branch.load_state_dict(self.global_model.global_branch.state_dict())
            personalized_test_accs.append(evaluate_model(personal_model, client.x_test, client.y_test, batch_size=self.batch_size))
            personalized_train_accs.append(evaluate_model(personal_model, client.x_train, client.y_train, batch_size=self.batch_size))
        return {
            "mean_personalized_accuracy": sum(personalized_test_accs) / max(len(personalized_test_accs), 1),
            "mean_personalized_train_accuracy": sum(personalized_train_accs) / max(len(personalized_train_accs), 1),
            "min_personalized_accuracy": min(personalized_test_accs) if personalized_test_accs else 0.0,
            "max_personalized_accuracy": max(personalized_test_accs) if personalized_test_accs else 0.0,
        }
