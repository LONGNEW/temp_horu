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
class CentralizedHDMethod(FederatedMethod):
    encoder: BaseHDEncoder
    updater: LocalHDUpdater
    num_classes: int
    metric: SimilarityMetric = "cos"
    debug: bool = False

    def __post_init__(self) -> None:
        self.global_memory: ClassMemory | None = None
        self._pooled_train_hv: torch.Tensor | None = None
        self._pooled_train_y: torch.Tensor | None = None
        self._round = 0

    def init_client_state(self, client: ClientData) -> ClientState:
        return ClientState(memory=None, extras={})

    def bootstrap(self, clients: list[ClientData], states: list[ClientState]) -> list[ClientState]:
        if not clients:
            raise ValueError("centralized_hd requires at least one client")

        pooled_x_train = torch.cat([client.x_train for client in clients], dim=0)
        pooled_y_train = torch.cat([client.y_train for client in clients], dim=0).to(torch.long)
        pooled_train_hv = self.encoder.encode(pooled_x_train)
        self._pooled_train_hv = pooled_train_hv
        self._pooled_train_y = pooled_y_train.to(pooled_train_hv.device)
        self.global_memory = ClassMemory.from_encoded(
            pooled_train_hv,
            self._pooled_train_y,
            self.num_classes,
        ).normalize_()
        return states

    def client_step(self, client: ClientData, state: ClientState) -> tuple[dict[str, Any], ClientState]:
        _ = client
        return {}, state

    def server_step(self, payloads: list[dict[str, Any]]) -> None:
        _ = payloads
        assert self.global_memory is not None
        assert self._pooled_train_hv is not None
        assert self._pooled_train_y is not None

        base_memory = self.global_memory.clone()
        updated_memory = self.updater.step(base_memory, self._pooled_train_hv, self._pooled_train_y)
        self.global_memory = updated_memory

        if self.debug:
            pooled_pred = similarity_scores(self._pooled_train_hv, self.global_memory.weight, self.metric).argmax(dim=1)
            pooled_acc = (pooled_pred == self._pooled_train_y).float().mean().item()
            print(
                f"[debug][centralized_hd][round={self._round + 1}] "
                f"pooled_train_acc={pooled_acc:.4f}"
            )
        self._round += 1

    def evaluate(self, clients: list[ClientData], states: list[ClientState]) -> dict[str, float]:
        _ = states
        assert self.global_memory is not None

        test_accs = []
        train_accs = []
        for client in clients:
            x_test_hv = self.encoder.encode(client.x_test)
            test_pred = similarity_scores(x_test_hv, self.global_memory.weight, self.metric).argmax(dim=1)
            test_accs.append((test_pred.cpu() == client.y_test.cpu()).float().mean().item())

            x_train_hv = self.encoder.encode(client.x_train)
            train_pred = similarity_scores(x_train_hv, self.global_memory.weight, self.metric).argmax(dim=1)
            train_accs.append((train_pred.cpu() == client.y_train.cpu()).float().mean().item())

        return {
            "mean_global_accuracy": sum(test_accs) / max(len(test_accs), 1),
            "mean_global_train_accuracy": sum(train_accs) / max(len(train_accs), 1),
            "min_global_accuracy": min(test_accs) if test_accs else 0.0,
            "max_global_accuracy": max(test_accs) if test_accs else 0.0,
        }
