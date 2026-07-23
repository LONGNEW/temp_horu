from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..data import ClientData
from ..encoder import (
    BaseHDEncoder,
    GroupLearnedCosineProjectionEncoder,
    PackedAdditiveCosineEncoder,
    ResidualPackedCosineEncoder,
)
from ..federated import ClientState, FederatedMethod
from ..local_update import LocalHDUpdater
from ..memory import ClassMemory
from ..similarity import SimilarityMetric, similarity_scores


@dataclass
class MaskingAntiCollapseHDMethod(FederatedMethod):
    """Centralized HD with a masking-based sparse-subspace encoder."""

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
            raise ValueError("masking_anticollapse_hd requires at least one client")
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
                f"[debug][masking_anticollapse_hd][round={self._round + 1}] "
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


@dataclass
class MLPGroupLearnedHDMethod(FederatedMethod):
    """Centralized HD where encoder groups are learned offline via a lightweight MLP."""

    encoder: GroupLearnedCosineProjectionEncoder
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
            raise ValueError("mlp_group_learned_hd requires at least one client")
        pooled_x_train = torch.cat([client.x_train for client in clients], dim=0)
        pooled_y_train = torch.cat([client.y_train for client in clients], dim=0).to(torch.long)

        self.encoder.fit(pooled_x_train, pooled_y_train, self.num_classes)
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
                f"[debug][mlp_group_learned_hd][round={self._round + 1}] "
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


@dataclass
class PackedAdditiveHDMethod(FederatedMethod):
    """Centralized HD with a packed additive grouped encoder."""

    encoder: PackedAdditiveCosineEncoder
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
            raise ValueError("packed additive HD requires at least one client")
        pooled_x_train = torch.cat([client.x_train for client in clients], dim=0)
        pooled_y_train = torch.cat([client.y_train for client in clients], dim=0).to(torch.long)

        self.encoder.fit(pooled_x_train, pooled_y_train, self.num_classes)
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
                f"[debug][packed_additive_hd][round={self._round + 1}] "
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


@dataclass
class NaiveGroupEnsembleHDMethod(FederatedMethod):
    """Centralized HD with explicit per-group memories (diagnostic oracle, no packing)."""

    encoder: PackedAdditiveCosineEncoder
    updater: LocalHDUpdater
    num_classes: int
    metric: SimilarityMetric = "cos"
    debug: bool = False

    def __post_init__(self) -> None:
        self.group_memories: list[ClassMemory] = []
        self.group_weights: torch.Tensor | None = None
        self._pooled_train_hv_by_group: list[torch.Tensor] = []
        self._pooled_train_y: torch.Tensor | None = None
        self._round = 0

    def init_client_state(self, client: ClientData) -> ClientState:
        return ClientState(memory=None, extras={})

    def bootstrap(self, clients: list[ClientData], states: list[ClientState]) -> list[ClientState]:
        if not clients:
            raise ValueError("naive_group_ensemble requires at least one client")
        pooled_x_train = torch.cat([client.x_train for client in clients], dim=0)
        pooled_y_train = torch.cat([client.y_train for client in clients], dim=0).to(torch.long)
        self.encoder.fit(pooled_x_train, pooled_y_train, self.num_classes)
        group_hv = self.encoder.encode_group_features(pooled_x_train)
        self._pooled_train_hv_by_group = group_hv
        self._pooled_train_y = pooled_y_train.to(group_hv[0].device)
        self.group_weights = self.encoder.group_weights.detach().to(group_hv[0].device).clone()
        self.group_memories = [
            ClassMemory.from_encoded(hv, self._pooled_train_y, self.num_classes).normalize_()
            for hv in group_hv
        ]
        return states

    def client_step(self, client: ClientData, state: ClientState) -> tuple[dict[str, Any], ClientState]:
        _ = client
        return {}, state

    def server_step(self, payloads: list[dict[str, Any]]) -> None:
        _ = payloads
        assert self.group_memories
        assert self._pooled_train_hv_by_group
        assert self._pooled_train_y is not None

        next_memories: list[ClassMemory] = []
        for memory, hv in zip(self.group_memories, self._pooled_train_hv_by_group):
            next_memories.append(self.updater.step(memory.clone(), hv, self._pooled_train_y))
        self.group_memories = next_memories
        self._round += 1

    def _fused_scores(self, group_hv: list[torch.Tensor]) -> torch.Tensor:
        assert self.group_weights is not None
        assert self.group_memories
        scores = torch.zeros(
            group_hv[0].shape[0],
            self.num_classes,
            device=group_hv[0].device,
            dtype=group_hv[0].dtype,
        )
        for g, hv in enumerate(group_hv):
            scores = scores + (
                float(self.group_weights[g].item())
                * similarity_scores(hv, self.group_memories[g].weight, self.metric)
            )
        return scores

    def evaluate(self, clients: list[ClientData], states: list[ClientState]) -> dict[str, float]:
        _ = states
        assert self.group_memories
        assert self.group_weights is not None

        test_accs = []
        train_accs = []
        for client in clients:
            test_group_hv = self.encoder.encode_group_features(client.x_test)
            test_scores = self._fused_scores(test_group_hv)
            test_pred = test_scores.argmax(dim=1)
            test_accs.append((test_pred.cpu() == client.y_test.cpu()).float().mean().item())

            train_group_hv = self.encoder.encode_group_features(client.x_train)
            train_scores = self._fused_scores(train_group_hv)
            train_pred = train_scores.argmax(dim=1)
            train_accs.append((train_pred.cpu() == client.y_train.cpu()).float().mean().item())

        return {
            "mean_global_accuracy": sum(test_accs) / max(len(test_accs), 1),
            "mean_global_train_accuracy": sum(train_accs) / max(len(train_accs), 1),
            "min_global_accuracy": min(test_accs) if test_accs else 0.0,
            "max_global_accuracy": max(test_accs) if test_accs else 0.0,
        }


@dataclass
class ResidualPackedHDMethod(FederatedMethod):
    """Centralized HD with budget-split residual(full + packed-group) encoder."""

    encoder: ResidualPackedCosineEncoder
    updater: LocalHDUpdater
    num_classes: int
    metric: SimilarityMetric = "cos"
    residual_eta_mode: str = "fixed"
    residual_eta_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    residual_eta_beta: float = 0.25
    debug: bool = False

    def __post_init__(self) -> None:
        mode = str(self.residual_eta_mode).lower()
        if mode not in {"fixed", "auto_margin_var"}:
            raise ValueError(f"Unsupported residual_eta_mode: {self.residual_eta_mode}")
        self.residual_eta_mode = mode
        self.residual_eta_grid = tuple(float(value) for value in self.residual_eta_grid)
        self.residual_eta_beta = float(self.residual_eta_beta)
        self.global_memory: ClassMemory | None = None
        self._pooled_train_hv: torch.Tensor | None = None
        self._pooled_train_y: torch.Tensor | None = None
        self._selected_eta: float | None = None
        self._eta_objective_by_candidate: dict[str, float] = {}
        self._round = 0

    def init_client_state(self, client: ClientData) -> ClientState:
        return ClientState(memory=None, extras={})

    def _margin_objective(
        self,
        encoded: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[float, float, float]:
        labels = labels.to(encoded.device).long()
        memory = ClassMemory.from_encoded(encoded, labels, self.num_classes).normalize_()
        scores = similarity_scores(encoded, memory.weight, self.metric)
        true_scores = scores.gather(1, labels.unsqueeze(1)).squeeze(1)
        if self.num_classes > 1:
            masked_scores = scores.clone()
            masked_scores.scatter_(1, labels.unsqueeze(1), float("-inf"))
            best_other = masked_scores.max(dim=1).values
        else:
            best_other = torch.zeros_like(true_scores)
        margins = true_scores - best_other
        mean_margin = float(margins.mean().item())
        std_margin = float(margins.std(unbiased=False).item())
        objective = mean_margin - (self.residual_eta_beta * std_margin)
        return objective, mean_margin, std_margin

    def _select_eta_auto_margin_var(
        self,
        pooled_x_train: torch.Tensor,
        pooled_y_train: torch.Tensor,
    ) -> float:
        candidates = [float(value) for value in self.residual_eta_grid]
        if not candidates:
            return float(self.encoder.eta)

        best_eta = float(self.encoder.eta)
        best_objective = float("-inf")
        self._eta_objective_by_candidate = {}
        for eta in candidates:
            self.encoder.set_eta(eta)
            self.encoder.fit(pooled_x_train, pooled_y_train, self.num_classes)
            encoded = self.encoder.encode(pooled_x_train)
            objective, mean_margin, std_margin = self._margin_objective(encoded, pooled_y_train)
            candidate_key = f"{float(eta):.6g}"
            self._eta_objective_by_candidate[candidate_key] = objective
            if objective > best_objective:
                best_objective = objective
                best_eta = float(eta)
            if self.debug:
                print(
                    "[debug][residual_packed_group_hd][eta_search] "
                    f"eta={eta:.4f} J={objective:.6f} margin_mean={mean_margin:.6f} "
                    f"margin_std={std_margin:.6f}"
                )
        return float(best_eta)

    def _attach_eta_diagnostics(self, states: list[ClientState]) -> list[ClientState]:
        if self._selected_eta is None:
            return states
        updated_states: list[ClientState] = []
        for state in states:
            extras = {} if state.extras is None else dict(state.extras)
            extras["_selected_eta"] = float(self._selected_eta)
            if self._eta_objective_by_candidate:
                extras["_eta_objective_by_candidate"] = dict(self._eta_objective_by_candidate)
            updated_states.append(ClientState(memory=state.memory, extras=extras))
        return updated_states

    def bootstrap(self, clients: list[ClientData], states: list[ClientState]) -> list[ClientState]:
        if not clients:
            raise ValueError("residual_packed_group_hd requires at least one client")
        pooled_x_train = torch.cat([client.x_train for client in clients], dim=0)
        pooled_y_train = torch.cat([client.y_train for client in clients], dim=0).to(torch.long)

        if self.residual_eta_mode == "auto_margin_var":
            self._selected_eta = self._select_eta_auto_margin_var(pooled_x_train, pooled_y_train)
        else:
            self._selected_eta = float(self.encoder.eta)
            self._eta_objective_by_candidate = {}
        self.encoder.set_eta(float(self._selected_eta))
        self.encoder.fit(pooled_x_train, pooled_y_train, self.num_classes)
        pooled_train_hv = self.encoder.encode(pooled_x_train)
        self._pooled_train_hv = pooled_train_hv
        self._pooled_train_y = pooled_y_train.to(pooled_train_hv.device)
        self.global_memory = ClassMemory.from_encoded(
            pooled_train_hv,
            self._pooled_train_y,
            self.num_classes,
        ).normalize_()
        return self._attach_eta_diagnostics(states)

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
            "_selected_eta": float(self.encoder.eta if self._selected_eta is None else self._selected_eta),
        }
