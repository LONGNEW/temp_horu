from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..data import ClientData
from ..encoder import BaseHDEncoder
from ..federated import ClientState, FederatedMethod
from ..local_update import LocalHDUpdater
from ..memory import ClassMemory
from ..similarity import SimilarityMetric, similarity_scores


@dataclass
class LocalHDMethod(FederatedMethod):
    encoder: BaseHDEncoder
    updater: LocalHDUpdater
    num_classes: int
    metric: SimilarityMetric = "cos"
    debug: bool = False

    def __post_init__(self) -> None:
        self._round = 0

    def init_client_state(self, client: ClientData) -> ClientState:
        x_hv = self.encoder.encode(client.x_train)
        local_memory = ClassMemory.from_encoded(
            x_hv,
            client.y_train.to(x_hv.device),
            self.num_classes,
        ).normalize_()
        return ClientState(memory=local_memory.weight.detach().clone(), extras={})

    def client_step(self, client: ClientData, state: ClientState) -> tuple[dict[str, Any], ClientState]:
        x_hv = self.encoder.encode(client.x_train)
        base_memory = ClassMemory(weight=state.memory.clone())
        updated_memory = self.updater.step(base_memory, x_hv, client.y_train.to(x_hv.device))
        delta = updated_memory.weight - base_memory.weight
        if self.debug:
            train_pred = similarity_scores(x_hv, updated_memory.weight, self.metric).argmax(dim=1)
            train_acc = (train_pred.cpu() == client.y_train.cpu()).float().mean().item()
            print(
                f"[debug][local][round={self._round + 1}][client={client.client_id}] "
                f"delta_norm={torch.linalg.norm(delta).item():.4f} "
                f"changed={int(torch.count_nonzero(delta).item())} "
                f"train_acc={train_acc:.4f}"
            )
        return {}, ClientState(memory=updated_memory.weight.detach().clone(), extras={})

    def server_step(self, payloads: list[dict[str, Any]]) -> None:
        self._round += 1

    def evaluate(self, clients: list[ClientData], states: list[ClientState]) -> dict[str, float]:
        test_accs = []
        train_accs = []
        for client, state in zip(clients, states):
            x_test_hv = self.encoder.encode(client.x_test)
            test_pred = similarity_scores(x_test_hv, state.memory, self.metric).argmax(dim=1)
            test_accs.append((test_pred.cpu() == client.y_test.cpu()).float().mean().item())

            x_train_hv = self.encoder.encode(client.x_train)
            train_pred = similarity_scores(x_train_hv, state.memory, self.metric).argmax(dim=1)
            train_accs.append((train_pred.cpu() == client.y_train.cpu()).float().mean().item())

        return {
            "mean_local_test_accuracy": sum(test_accs) / max(len(test_accs), 1),
            "mean_local_train_accuracy": sum(train_accs) / max(len(train_accs), 1),
            "min_local_test_accuracy": min(test_accs) if test_accs else 0.0,
            "max_local_test_accuracy": max(test_accs) if test_accs else 0.0,
            "min_local_train_accuracy": min(train_accs) if train_accs else 0.0,
            "max_local_train_accuracy": max(train_accs) if train_accs else 0.0,
        }
