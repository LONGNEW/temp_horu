from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from our_hd.data import ClientData


@dataclass
class NNClientState:
    personalized_state: dict[str, Any] | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class NNFederatedMethod(ABC):
    def bootstrap(self, clients: list[ClientData], states: list[NNClientState]) -> list[NNClientState]:
        return states

    @abstractmethod
    def init_client_state(self, client: ClientData) -> NNClientState:
        raise NotImplementedError

    @abstractmethod
    def client_step(self, client: ClientData, state: NNClientState) -> tuple[dict[str, Any], NNClientState]:
        raise NotImplementedError

    @abstractmethod
    def server_step(self, payloads: list[dict[str, Any]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, clients: list[ClientData], states: list[NNClientState]) -> dict[str, float]:
        raise NotImplementedError


class NNFederatedRunner:
    def __init__(
        self,
        method: NNFederatedMethod,
        rounds: int,
        *,
        client_participation: float = 1.0,
        seed: int = 13,
    ) -> None:
        self.method = method
        self.rounds = int(rounds)
        self.client_participation = float(client_participation)
        self.rng = np.random.default_rng(seed)

    def _sample_client_indices(self, num_clients: int) -> list[int]:
        if num_clients <= 0:
            return []
        if self.client_participation >= 1.0:
            return list(range(num_clients))
        sample_size = min(num_clients, max(1, int(math.ceil(self.client_participation * num_clients))))
        return sorted(self.rng.choice(num_clients, size=sample_size, replace=False).tolist())

    def run(self, clients: list[ClientData]) -> dict[str, Any]:
        states = [self.method.init_client_state(client) for client in clients]
        states = self.method.bootstrap(clients, states)
        history: list[dict[str, float]] = []

        for _ in range(self.rounds):
            payloads: list[dict[str, Any]] = []
            next_states = list(states)
            selected_indices = self._sample_client_indices(len(clients))
            for idx in selected_indices:
                payload, next_state = self.method.client_step(clients[idx], states[idx])
                payloads.append(payload)
                next_states[idx] = next_state
            self.method.server_step(payloads)
            states = next_states
            history.append(self.method.evaluate(clients, states))

        return {"history": history, "states": states}
