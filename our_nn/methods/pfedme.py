from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from our_hd.data import ClientData
from our_nn.federated import NNClientState, NNFederatedMethod
from our_nn.models import MLP
from our_nn.train import average_state_dicts, blend_state_dicts, detached_state_dict, evaluate_model, train_supervised_epoch


@dataclass
class PFedMeMLPMethod(NNFederatedMethod):
    input_dim: int
    num_classes: int
    hidden_dim: int
    num_hidden_layers: int
    local_epochs: int
    batch_size: int
    personal_lr: float
    reference_lr: float
    lambda_reg: float
    beta: float
    personal_steps: int = 1
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
        reference_state = detached_state_dict(self.global_model, device=self.state_device)
        personal_model = self._new_model()
        if state.personalized_state is not None:
            personal_model.load_state_dict(state.personalized_state)
        else:
            personal_model.load_state_dict(reference_state)

        train_metrics = {"accuracy": 0.0}
        for _ in range(self.local_epochs):
            for _ in range(max(1, self.personal_steps)):
                train_metrics = train_supervised_epoch(
                    personal_model,
                    client.x_train,
                    client.y_train,
                    lr=self.current_personal_lr,
                    batch_size=self.batch_size,
                    optimizer_name=self.optimizer_name,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay,
                    prox_state=reference_state,
                    prox_mu=self.lambda_reg,
                )
            personalized_state = detached_state_dict(personal_model, device=self.state_device)
            reference_state = blend_state_dicts(
                reference_state,
                personalized_state,
                alpha=self.reference_lr * self.lambda_reg,
            )

        if self.debug:
            print(
                f"[debug][pfedme_mlp][round={self._round + 1}][client={client.client_id}] "
                f"personal_train_acc={train_metrics['accuracy']:.4f}"
            )

        payload = {
            "reference_state": reference_state,
            "num_samples": float(len(client.y_train)),
        }
        next_state = NNClientState(personalized_state=detached_state_dict(personal_model, device=self.state_device))
        return payload, next_state

    def server_step(self, payloads: list[dict[str, Any]]) -> None:
        if payloads:
            reference_states = [payload["reference_state"] for payload in payloads]
            weights = [payload["num_samples"] for payload in payloads]
            averaged_reference = average_state_dicts(reference_states, weights)
            global_state = detached_state_dict(self.global_model, device=self.state_device)
            next_global_state = blend_state_dicts(global_state, averaged_reference, alpha=self.beta)
            self.global_model.load_state_dict(next_global_state)
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
