from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from our_hd.data import ClientData
from our_nn.federated import NNClientState, NNFederatedMethod
from our_nn.models import MLP
from our_nn.train import average_state_dicts, detached_state_dict, evaluate_model, train_supervised_epoch


@dataclass
class FedAvgMLPMethod(NNFederatedMethod):
    input_dim: int
    num_classes: int
    hidden_dim: int
    num_hidden_layers: int
    local_epochs: int
    batch_size: int
    lr: float
    lr_decay: float = 1.0
    momentum: float = 0.0
    weight_decay: float = 0.0
    optimizer_name: str = "sgd"
    dropout: float = 0.0
    device: torch.device | str = "cpu"
    state_device: torch.device | str = "cpu"
    debug: bool = False

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self.state_device = torch.device(self.state_device)
        self.global_model = self._new_model()
        self.current_lr = float(self.lr)
        self._round = 0

    def _new_model(self) -> MLP:
        return MLP(
            self.input_dim,
            self.hidden_dim,
            self.num_classes,
            num_hidden_layers=self.num_hidden_layers,
            dropout=self.dropout,
        ).to(self.device)

    def init_client_state(self, client: ClientData) -> NNClientState:
        return NNClientState()

    def client_step(self, client: ClientData, state: NNClientState) -> tuple[dict[str, Any], NNClientState]:
        local_model = self._new_model()
        local_model.load_state_dict(self.global_model.state_dict())
        train_metrics = {"accuracy": 0.0}
        for _ in range(self.local_epochs):
            train_metrics = train_supervised_epoch(
                local_model,
                client.x_train,
                client.y_train,
                lr=self.current_lr,
                batch_size=self.batch_size,
                optimizer_name=self.optimizer_name,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        if self.debug:
            print(
                f"[debug][fedavg_mlp][round={self._round + 1}][client={client.client_id}] "
                f"train_acc={train_metrics['accuracy']:.4f}"
            )
        payload = {
            "model_state": detached_state_dict(local_model, device=self.state_device),
            "num_samples": float(len(client.y_train)),
        }
        return payload, state

    def server_step(self, payloads: list[dict[str, Any]]) -> None:
        if payloads:
            states = [payload["model_state"] for payload in payloads]
            weights = [payload["num_samples"] for payload in payloads]
            averaged_state = average_state_dicts(states, weights)
            self.global_model.load_state_dict(averaged_state)
        self.current_lr *= float(self.lr_decay)
        self._round += 1

    def evaluate(self, clients: list[ClientData], states: list[NNClientState]) -> dict[str, float]:
        test_accs = [evaluate_model(self.global_model, client.x_test, client.y_test, batch_size=self.batch_size) for client in clients]
        train_accs = [evaluate_model(self.global_model, client.x_train, client.y_train, batch_size=self.batch_size) for client in clients]
        return {
            "mean_global_accuracy": sum(test_accs) / max(len(test_accs), 1),
            "mean_global_train_accuracy": sum(train_accs) / max(len(train_accs), 1),
            "min_global_accuracy": min(test_accs) if test_accs else 0.0,
            "max_global_accuracy": max(test_accs) if test_accs else 0.0,
        }
