from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from our_hd.data import ClientData
from our_nn.federated import NNClientState, NNFederatedMethod
from our_nn.models import MLP
from our_nn.train import average_state_dicts, detached_state_dict, evaluate_model, train_supervised_epoch


@dataclass
class DittoMLPMethod(NNFederatedMethod):
    input_dim: int
    num_classes: int
    hidden_dim: int
    num_hidden_layers: int
    local_epochs: int
    batch_size: int
    global_lr: float
    personal_lr: float
    lambda_reg: float
    global_lr_decay: float = 1.0
    personal_lr_decay: float = 1.0
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
        self.current_global_lr = float(self.global_lr)
        self.current_personal_lr = float(self.personal_lr)
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
        global_state_device = detached_state_dict(self.global_model, device=self.device)

        global_local_model = self._new_model()
        global_local_model.load_state_dict(self.global_model.state_dict())
        global_metrics = {"accuracy": 0.0}
        for _ in range(self.local_epochs):
            global_metrics = train_supervised_epoch(
                global_local_model,
                client.x_train,
                client.y_train,
                lr=self.current_global_lr,
                batch_size=self.batch_size,
                optimizer_name=self.optimizer_name,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )

        personal_model = self._new_model()
        if state.personalized_state is not None:
            personal_model.load_state_dict(state.personalized_state)
        else:
            personal_model.load_state_dict(self.global_model.state_dict())
        personal_metrics = {"accuracy": 0.0}
        for _ in range(self.local_epochs):
            personal_metrics = train_supervised_epoch(
                personal_model,
                client.x_train,
                client.y_train,
                lr=self.current_personal_lr,
                batch_size=self.batch_size,
                optimizer_name=self.optimizer_name,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                prox_state=global_state_device,
                prox_mu=self.lambda_reg,
            )

        if self.debug:
            print(
                f"[debug][ditto_mlp][round={self._round + 1}][client={client.client_id}] "
                f"global_train_acc={global_metrics['accuracy']:.4f} "
                f"personal_train_acc={personal_metrics['accuracy']:.4f}"
            )

        payload = {
            "model_state": detached_state_dict(global_local_model, device=self.state_device),
            "num_samples": float(len(client.y_train)),
        }
        next_state = NNClientState(personalized_state=detached_state_dict(personal_model, device=self.state_device))
        return payload, next_state

    def server_step(self, payloads: list[dict[str, Any]]) -> None:
        if payloads:
            states = [payload["model_state"] for payload in payloads]
            weights = [payload["num_samples"] for payload in payloads]
            averaged_state = average_state_dicts(states, weights)
            self.global_model.load_state_dict(averaged_state)
        self.current_global_lr *= float(self.global_lr_decay)
        self.current_personal_lr *= float(self.personal_lr_decay)
        self._round += 1

    def evaluate(self, clients: list[ClientData], states: list[NNClientState]) -> dict[str, float]:
        global_test_accs = []
        personalized_test_accs = []
        personalized_train_accs = []
        for client, state in zip(clients, states):
            global_test_accs.append(evaluate_model(self.global_model, client.x_test, client.y_test, batch_size=self.batch_size))
            personal_model = self._new_model()
            if state.personalized_state is not None:
                personal_model.load_state_dict(state.personalized_state)
            else:
                personal_model.load_state_dict(self.global_model.state_dict())
            personalized_test_accs.append(evaluate_model(personal_model, client.x_test, client.y_test, batch_size=self.batch_size))
            personalized_train_accs.append(evaluate_model(personal_model, client.x_train, client.y_train, batch_size=self.batch_size))

        return {
            "mean_global_accuracy": sum(global_test_accs) / max(len(global_test_accs), 1),
            "mean_personalized_accuracy": sum(personalized_test_accs) / max(len(personalized_test_accs), 1),
            "mean_personalized_train_accuracy": sum(personalized_train_accs) / max(len(personalized_train_accs), 1),
            "min_personalized_accuracy": min(personalized_test_accs) if personalized_test_accs else 0.0,
            "max_personalized_accuracy": max(personalized_test_accs) if personalized_test_accs else 0.0,
        }
