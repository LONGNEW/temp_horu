from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from our_hd.data import ClientData
from our_nn.federated import NNClientState, NNFederatedMethod
from our_nn.models import DFLNet
from our_nn.train import average_state_dicts, detached_state_dict, evaluate_model, train_dfl_epoch


@dataclass
class DFLMLPMethod(NNFederatedMethod):
    input_dim: int
    num_classes: int
    hidden_dim: int
    branch_layers: int
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
    device: torch.device | str = "cpu"
    state_device: torch.device | str = "cpu"
    debug: bool = False

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self.state_device = torch.device(self.state_device)
        self.global_model = self._new_model()
        self.current_lr = float(self.lr)
        self._round = 0

    def _new_model(self) -> DFLNet:
        return DFLNet(
            self.input_dim,
            self.hidden_dim,
            self.num_classes,
            branch_layers=self.branch_layers,
            dropout=self.dropout,
        ).to(self.device)

    def init_client_state(self, client: ClientData) -> NNClientState:
        return NNClientState()

    def client_step(self, client: ClientData, state: NNClientState) -> tuple[dict[str, Any], NNClientState]:
        local_model = self._new_model()
        if state.personalized_state is not None:
            local_model.load_state_dict(state.personalized_state)
        else:
            local_model.load_state_dict(self.global_model.state_dict())
        local_model.global_branch.load_state_dict(self.global_model.global_branch.state_dict())

        global_invariant_state = detached_state_dict(self.global_model.global_branch, device=self.state_device)
        train_metrics = {"accuracy": 0.0}
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

        if self.debug:
            print(
                f"[debug][dfl_mlp][round={self._round + 1}][client={client.client_id}] "
                f"personal_train_acc={train_metrics['accuracy']:.4f}"
            )

        payload = {
            "global_branch_state": detached_state_dict(local_model.global_branch, device=self.state_device),
            "num_samples": float(len(client.y_train)),
        }
        next_state = NNClientState(personalized_state=detached_state_dict(local_model, device=self.state_device))
        return payload, next_state

    def server_step(self, payloads: list[dict[str, Any]]) -> None:
        if payloads:
            branch_states = [payload["global_branch_state"] for payload in payloads]
            weights = [payload["num_samples"] for payload in payloads]
            averaged_branch_state = average_state_dicts(branch_states, weights)
            self.global_model.global_branch.load_state_dict(averaged_branch_state)
        self.current_lr *= float(self.lr_decay)
        self._round += 1

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
