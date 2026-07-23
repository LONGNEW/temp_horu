from __future__ import annotations

import math
import time
import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .data import ClientData


@dataclass
class ClientState:
    memory: torch.Tensor | None = None
    extras: dict[str, Any] | None = None


class FederatedMethod(ABC):
    """Base interface for HD federated methods."""

    def bootstrap(self, clients: list[ClientData], states: list[ClientState]) -> list[ClientState]:
        return states

    def profiled_init_client_state(
        self,
        client: ClientData,
    ) -> tuple[ClientState, dict[str, Any]]:
        started = time.perf_counter()
        state = self.init_client_state(client)
        return state, {"init_client_state_ms": (time.perf_counter() - started) * 1000.0}

    def profiled_bootstrap(
        self,
        clients: list[ClientData],
        states: list[ClientState],
    ) -> tuple[list[ClientState], dict[str, Any]]:
        started = time.perf_counter()
        next_states = self.bootstrap(clients, states)
        return next_states, {"total_bootstrap_ms": (time.perf_counter() - started) * 1000.0}

    @abstractmethod
    def init_client_state(self, client: ClientData) -> ClientState:
        raise NotImplementedError

    @abstractmethod
    def client_step(self, client: ClientData, state: ClientState) -> tuple[dict[str, Any], ClientState]:
        raise NotImplementedError

    @abstractmethod
    def server_step(self, payloads: list[dict[str, Any]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, clients: list[ClientData], states: list[ClientState]) -> dict[str, float]:
        raise NotImplementedError

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
        del clients, states, payloads, round_index, selected_indices, server_step_ms, round_runtime_sec
        return None


class FederatedRunner:
    """Small round runner so each method only implements its paper logic."""

    def __init__(
        self,
        method: FederatedMethod,
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

    def _clone_state(self, state: ClientState) -> ClientState:
        def _clone_value(value: object) -> object:
            if isinstance(value, torch.Tensor):
                return value.detach().clone()
            if isinstance(value, dict):
                return {key: _clone_value(item) for key, item in value.items()}
            if isinstance(value, list):
                return [_clone_value(item) for item in value]
            if isinstance(value, tuple):
                return tuple(_clone_value(item) for item in value)
            return copy.deepcopy(value)

        return ClientState(
            memory=None if state.memory is None else state.memory.detach().clone(),
            extras=None if state.extras is None else {key: _clone_value(value) for key, value in state.extras.items()},
        )

    def run(self, clients: list[ClientData], snapshot_rounds: set[int] | None = None) -> dict[str, Any]:
        snapshot_rounds = set(snapshot_rounds) if snapshot_rounds is not None else set()
        init_profiles: list[dict[str, Any]] = []
        states: list[ClientState] = []
        method_states: dict[str, Any] = {}
        for client in clients:
            state, profile = self.method.profiled_init_client_state(client)
            states.append(state)
            init_profiles.append({"client_id": client.client_id, **profile})
        states, bootstrap_artifacts = self.method.profiled_bootstrap(clients, states)
        history: list[dict[str, float]] = []
        round_artifacts: list[dict[str, Any]] = []
        round_states: dict[str, list[ClientState]] = {}

        for round_index in range(self.rounds):
            round_started = time.perf_counter()
            payloads: list[dict[str, Any]] = []
            next_states = list(states)
            selected_indices = self._sample_client_indices(len(clients))
            for idx in selected_indices:
                client_step_started = time.perf_counter()
                payload, next_state = self.method.client_step(clients[idx], states[idx])
                # Artifact collectors may opt into this timing.  The reserved
                # key is ignored by method server implementations.
                payload["_client_step_ms"] = (time.perf_counter() - client_step_started) * 1000.0
                payloads.append(payload)
                next_states[idx] = next_state
            server_started = time.perf_counter()
            self.method.server_step(payloads)
            server_step_ms = (time.perf_counter() - server_started) * 1000.0
            states = next_states
            history.append(self.method.evaluate(clients, states))
            round_runtime_sec = time.perf_counter() - round_started
            artifacts = self.method.collect_round_artifacts(
                clients,
                states,
                payloads,
                round_index=round_index + 1,
                selected_indices=selected_indices,
                server_step_ms=server_step_ms,
                round_runtime_sec=round_runtime_sec,
            )
            round_artifacts.append(
                {
                    "round": round_index + 1,
                    "selected_client_ids": [clients[idx].client_id for idx in selected_indices],
                    "server_step_ms": server_step_ms,
                    "round_runtime_sec": round_runtime_sec,
                    **({} if artifacts is None else artifacts),
                }
            )
            if (round_index + 1) in snapshot_rounds:
                round_states[str(round_index + 1)] = [self._clone_state(state) for state in states]
                try:
                    method_states[str(round_index + 1)] = copy.deepcopy(self.method)
                except Exception:
                    method_states[str(round_index + 1)] = None

        return {
            "history": history,
            "states": states,
            "init_profiles": init_profiles,
            "bootstrap_artifacts": bootstrap_artifacts,
            "round_artifacts": round_artifacts,
            "round_states": round_states,
            "method_states": method_states,
        }
