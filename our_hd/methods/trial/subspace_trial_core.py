from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from ...data import ClientData
from ...encoder import BaseHDEncoder
from ...federated import ClientState, FederatedMethod


def mean_prototypes(x_hv: torch.Tensor, y: torch.Tensor, num_classes: int) -> torch.Tensor:
    sums = torch.zeros(num_classes, x_hv.shape[1], device=x_hv.device, dtype=x_hv.dtype)
    counts = torch.zeros(num_classes, device=x_hv.device, dtype=x_hv.dtype)
    sums.index_add_(0, y.long(), x_hv)
    counts.index_add_(0, y.long(), torch.ones_like(y, dtype=x_hv.dtype))
    prototypes = sums / counts.clamp_min(1.0).unsqueeze(1)
    prototypes[counts <= 0] = 0.0
    return prototypes


def row_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / torch.linalg.norm(x, dim=1, keepdim=True).clamp_min(eps)


def pairwise_cosine(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x_norm = torch.linalg.norm(x, dim=1).clamp_min(eps)
    y_norm = torch.linalg.norm(y, dim=1).clamp_min(eps)
    return (x * y).sum(dim=1) / (x_norm * y_norm)


def split_train_validation(y: torch.Tensor, val_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    num_samples = int(y.numel())
    if num_samples <= 2 or float(val_fraction) <= 0.0:
        indices = torch.arange(num_samples, device=y.device)
        return indices, indices[:0]

    num_val = int(round(float(val_fraction) * num_samples))
    num_val = max(1, min(num_samples - 1, num_val))
    generator = torch.Generator(device=y.device if y.device.type != "cpu" else "cpu")
    generator.manual_seed(int(seed))
    perm = torch.randperm(num_samples, generator=generator, device=y.device)
    val_idx = perm[:num_val]
    train_idx = perm[num_val:]
    return train_idx, val_idx


def covariance_from_memories(memories: list[torch.Tensor]) -> torch.Tensor:
    hd_dim = memories[0].shape[1]
    covariance = torch.zeros(hd_dim, hd_dim, device=memories[0].device, dtype=memories[0].dtype)
    for memory in memories:
        covariance.add_(memory.T @ memory)
    return covariance


def orthonormalize_columns(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x.numel() == 0:
        return x
    q, _ = torch.linalg.qr(x, mode="reduced")
    keep: list[torch.Tensor] = []
    for col in range(q.shape[1]):
        vector = q[:, col]
        if float(torch.linalg.norm(vector).item()) > eps:
            keep.append(vector)
    if not keep:
        return x[:, :0]
    return torch.stack(keep, dim=1)


def complete_basis(
    hd_dim: int,
    rank: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    reference: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    resolved_rank = max(1, int(rank))
    reference_basis = reference if reference is not None and reference.numel() > 0 else None
    basis_vectors: list[torch.Tensor] = []

    def project_out(vector: torch.Tensor) -> torch.Tensor:
        result = vector
        if reference_basis is not None:
            result = result - reference_basis @ (reference_basis.T @ result)
        for existing in basis_vectors:
            result = result - existing * torch.dot(existing, result)
        return result

    for idx in range(hd_dim):
        candidate = torch.zeros(hd_dim, device=device, dtype=dtype)
        candidate[idx] = 1.0
        candidate = project_out(candidate)
        norm = torch.linalg.norm(candidate)
        if float(norm.item()) <= eps:
            continue
        basis_vectors.append(candidate / norm)
        if len(basis_vectors) >= resolved_rank:
            return torch.stack(basis_vectors, dim=1)

    generator = torch.Generator(device=device if device.type != "cpu" else "cpu")
    generator.manual_seed(13)
    attempts = 0
    while len(basis_vectors) < resolved_rank and attempts < (resolved_rank * 16):
        candidate = torch.randn(hd_dim, generator=generator, device=device, dtype=dtype)
        candidate = project_out(candidate)
        norm = torch.linalg.norm(candidate)
        if float(norm.item()) > eps:
            basis_vectors.append(candidate / norm)
        attempts += 1

    if not basis_vectors:
        return torch.zeros(hd_dim, resolved_rank, device=device, dtype=dtype)
    stacked = torch.stack(basis_vectors[:resolved_rank], dim=1)
    if stacked.shape[1] < resolved_rank:
        padding = torch.zeros(hd_dim, resolved_rank - stacked.shape[1], device=device, dtype=dtype)
        stacked = torch.cat([stacked, padding], dim=1)
    return stacked


def orthogonalize_against(basis: torch.Tensor, reference: torch.Tensor | None, target_rank: int) -> torch.Tensor:
    if target_rank <= 0:
        return basis[:, :0]
    if reference is not None and reference.numel() > 0:
        basis = basis - reference @ (reference.T @ basis)
    basis = orthonormalize_columns(basis)
    if basis.shape[1] >= target_rank:
        return basis[:, :target_rank]
    extra = complete_basis(
        hd_dim=basis.shape[0],
        rank=target_rank - basis.shape[1],
        device=basis.device,
        dtype=basis.dtype,
        reference=reference
        if basis.shape[1] == 0
        else torch.cat([reference, basis], dim=1)
        if reference is not None and reference.numel() > 0
        else basis,
    )
    if basis.shape[1] == 0:
        return extra[:, :target_rank]
    return torch.cat([basis, extra], dim=1)[:, :target_rank]


def top_basis_from_covariance(
    covariance: torch.Tensor,
    rank: int,
    *,
    reference: torch.Tensor | None = None,
    complete_degenerate_basis: bool = False,
    oversample_factor: int = 1,
) -> torch.Tensor:
    resolved_rank = max(1, min(int(rank), covariance.shape[0]))
    if float(torch.linalg.norm(covariance).item()) <= 1e-12:
        if not complete_degenerate_basis:
            return torch.zeros(covariance.shape[0], resolved_rank, device=covariance.device, dtype=covariance.dtype)
        return complete_basis(
            hd_dim=covariance.shape[0],
            rank=resolved_rank,
            device=covariance.device,
            dtype=covariance.dtype,
            reference=reference,
        )
    _, eigenvectors = torch.linalg.eigh(covariance)
    candidate_count = max(resolved_rank, int(oversample_factor) * resolved_rank)
    candidates = torch.flip(eigenvectors, dims=[1])[:, :candidate_count]
    if reference is None and not complete_degenerate_basis and int(oversample_factor) <= 1:
        return candidates[:, :resolved_rank]
    return orthogonalize_against(candidates, reference, resolved_rank)


def simple_personal_basis(memory: torch.Tensor, shared_basis: torch.Tensor, rank: int) -> torch.Tensor:
    projected = (memory @ shared_basis) @ shared_basis.T
    residual = memory - projected
    covariance = residual.T @ residual
    return top_basis_from_covariance(covariance, rank)


def residual_personal_basis_and_coords(
    memory: torch.Tensor,
    shared_basis: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    projected = (memory @ shared_basis) @ shared_basis.T
    residual = memory - projected
    covariance = residual.T @ residual
    personal_basis = top_basis_from_covariance(
        covariance,
        rank,
        reference=shared_basis,
        complete_degenerate_basis=True,
        oversample_factor=2,
    )
    personal_coords = residual @ personal_basis
    return personal_basis, personal_coords


@dataclass
class BaseSubspaceTrialMethod(FederatedMethod):
    encoder: BaseHDEncoder
    num_classes: int
    shared_rank: int = 16
    personal_rank: int = 16
    local_epochs: int = 3
    batch_size: int = 32
    global_lr: float = 1.0
    personal_lr: float = 1.0
    val_fraction: float = 0.2
    alpha_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    enable_inround_checkpoints: bool = False
    enable_subspace_diagnostics: bool = False
    enable_system_profiling: bool = False
    trace_rounds: tuple[int, ...] = ()
    runtime_seed: int = 13
    debug: bool = False

    def __post_init__(self) -> None:
        self.shared_basis: torch.Tensor | None = None
        self._round = 0
        self._init_runtime_state()

    def _init_runtime_state(self) -> None:
        pass

    def _client_seed(self, client: ClientData) -> int:
        return 13 + sum(ord(ch) for ch in client.client_id)

    def _empirical_memory(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_hv = self.encoder.encode(x)
        return self._empirical_memory_from_encoded(x_hv, y)

    def _empirical_memory_from_encoded(self, x_hv: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        prototypes = mean_prototypes(x_hv, y.to(x_hv.device).long(), self.num_classes)
        return row_normalize(prototypes)

    def _build_init_extras(
        self,
        client: ClientData,
        *,
        train_idx: torch.Tensor,
        val_idx: torch.Tensor,
        train_x_hv: torch.Tensor,
        train_y: torch.Tensor,
        full_memory: torch.Tensor,
    ) -> dict[str, Any]:
        return {}

    def init_client_state(self, client: ClientData) -> ClientState:
        seed = self._client_seed(client)
        train_idx, val_idx = split_train_validation(client.y_train, self.val_fraction, seed)
        train_x = client.x_train.index_select(0, train_idx)
        train_y = client.y_train.index_select(0, train_idx).to(self.encoder.device).long()
        train_x_hv = self.encoder.encode(train_x)
        full_memory = self._empirical_memory_from_encoded(train_x_hv, train_y)
        extras = {
            "train_idx": train_idx.detach().clone(),
            "val_idx": val_idx.detach().clone(),
        }
        extras.update(
            self._build_init_extras(
                client,
                train_idx=train_idx,
                val_idx=val_idx,
                train_x_hv=train_x_hv,
                train_y=train_y,
                full_memory=full_memory,
            )
        )
        return ClientState(memory=full_memory.detach().clone(), extras=extras)

    def bootstrap(self, clients: list[ClientData], states: list[ClientState]) -> list[ClientState]:
        self.shared_basis = self._shared_basis_from_states(states)
        return [
            self._project_client_state(client, state.memory.to(self.encoder.device), state.extras)
            for client, state in zip(clients, states)
        ]

    def _shared_basis_from_states(self, states: list[ClientState]) -> torch.Tensor:
        memories = [state.memory.to(self.encoder.device) for state in states]
        return self._shared_basis_from_memories(memories)

    def _shared_basis_from_memories(self, memories: list[torch.Tensor]) -> torch.Tensor:
        return self._shared_basis_from_covariance(covariance_from_memories(memories))

    @abstractmethod
    def _shared_basis_from_covariance(self, covariance: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _decompose_memory(self, full_memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _bootstrap_alpha(
        self,
        client: ClientData,
        *,
        extras: dict[str, Any],
        personal_basis: torch.Tensor,
        shared_coords: torch.Tensor,
        personal_coords: torch.Tensor,
    ) -> float:
        return self._learn_alpha(
            client,
            val_idx=extras.get("val_idx"),
            personal_basis=personal_basis,
            shared_coords=shared_coords,
            personal_coords=personal_coords,
        )

    def _bootstrap_extra_updates(
        self,
        *,
        full_memory: torch.Tensor,
        personal_basis: torch.Tensor,
        shared_coords: torch.Tensor,
        personal_coords: torch.Tensor,
        alpha: float,
    ) -> dict[str, Any]:
        return {
            "full_memory": full_memory.detach().clone(),
            "shared_coords": shared_coords.detach().clone(),
            "personal_coords": personal_coords.detach().clone(),
            "personal_basis": personal_basis.detach().clone(),
            "alpha": float(alpha),
        }

    def _bootstrap_state_memory(
        self,
        *,
        full_memory: torch.Tensor,
        reconstructed_memory: torch.Tensor,
    ) -> torch.Tensor:
        return reconstructed_memory

    def _project_client_state(
        self,
        client: ClientData,
        full_memory: torch.Tensor,
        extras: dict[str, Any] | None,
    ) -> ClientState:
        extras = {} if extras is None else dict(extras)
        personal_basis, shared_coords, personal_coords = self._decompose_memory(full_memory)
        alpha = self._bootstrap_alpha(
            client,
            extras=extras,
            personal_basis=personal_basis,
            shared_coords=shared_coords,
            personal_coords=personal_coords,
        )
        reconstructed_memory = self._reconstruct_memory(shared_coords, personal_coords, personal_basis)
        extras.update(
            self._bootstrap_extra_updates(
                full_memory=full_memory,
                personal_basis=personal_basis,
                shared_coords=shared_coords,
                personal_coords=personal_coords,
                alpha=alpha,
            )
        )
        state_memory = self._bootstrap_state_memory(
            full_memory=full_memory,
            reconstructed_memory=reconstructed_memory,
        )
        return ClientState(memory=state_memory.detach().clone(), extras=extras)

    def _reconstruct_memory(
        self,
        shared_coords: torch.Tensor,
        personal_coords: torch.Tensor,
        personal_basis: torch.Tensor,
    ) -> torch.Tensor:
        assert self.shared_basis is not None
        reconstructed = shared_coords @ self.shared_basis.T
        reconstructed = reconstructed + (personal_coords @ personal_basis.T)
        return row_normalize(reconstructed)

    @abstractmethod
    def _scores(
        self,
        x_hv: torch.Tensor,
        *,
        personal_basis: torch.Tensor,
        shared_coords: torch.Tensor,
        personal_coords: torch.Tensor,
        alpha: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _learn_alpha(
        self,
        client: ClientData,
        *,
        val_idx: torch.Tensor | None,
        personal_basis: torch.Tensor,
        shared_coords: torch.Tensor,
        personal_coords: torch.Tensor,
    ) -> float:
        if val_idx is None or int(val_idx.numel()) == 0:
            return 0.5
        x_val = client.x_train.index_select(0, val_idx)
        y_val = client.y_train.index_select(0, val_idx)
        x_val_hv = self.encoder.encode(x_val)
        best_alpha = 0.5
        best_acc = -1.0
        for alpha in self.alpha_grid:
            fused_scores, _, _ = self._scores(
                x_val_hv,
                personal_basis=personal_basis,
                shared_coords=shared_coords,
                personal_coords=personal_coords,
                alpha=float(alpha),
            )
            pred = fused_scores.argmax(dim=1)
            acc = float((pred.cpu() == y_val.cpu()).float().mean().item())
            if acc > best_acc:
                best_acc = acc
                best_alpha = float(alpha)
        return best_alpha

    def _materialize_state(self, client: ClientData, state: ClientState, *, consume: bool) -> ClientState:
        extras = {} if state.extras is None else dict(state.extras)
        memory = None if state.memory is None else state.memory.detach().clone()
        return ClientState(memory=memory, extras=extras)

    def _state_components(self, state: ClientState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        assert state.extras is not None
        personal_basis = state.extras["personal_basis"].to(self.encoder.device)
        shared_coords = state.extras["shared_coords"].to(self.encoder.device)
        personal_coords = state.extras["personal_coords"].to(self.encoder.device)
        alpha = float(state.extras["alpha"])
        return personal_basis, shared_coords, personal_coords, alpha

    def _next_state_extras(
        self,
        state: ClientState,
        *,
        full_memory: torch.Tensor,
        shared_coords: torch.Tensor,
        personal_coords: torch.Tensor,
        personal_basis: torch.Tensor,
        alpha: float,
    ) -> dict[str, Any]:
        assert state.extras is not None
        return {
            "train_idx": state.extras["train_idx"].detach().clone(),
            "val_idx": state.extras["val_idx"].detach().clone(),
            "full_memory": full_memory.detach().clone(),
            "shared_coords": shared_coords.detach().clone(),
            "personal_coords": personal_coords.detach().clone(),
            "personal_basis": personal_basis.detach().clone(),
            "alpha": float(alpha),
        }

    def _include_mean_local_test_accuracy(self) -> bool:
        return False

    def evaluate(self, clients: list[ClientData], states: list[ClientState]) -> dict[str, float]:
        personalized_accs = []
        shared_branch_accs = []
        personal_branch_accs = []
        alpha_values = []

        for client, state in zip(clients, states):
            effective_state = self._materialize_state(client, state, consume=False)
            personal_basis, shared_coords, personal_coords, alpha = self._state_components(effective_state)
            alpha_values.append(alpha)

            x_test_hv = self.encoder.encode(client.x_test)
            fused_scores, shared_scores, personal_scores = self._scores(
                x_test_hv,
                personal_basis=personal_basis,
                shared_coords=shared_coords,
                personal_coords=personal_coords,
                alpha=alpha,
            )
            y_test = client.y_test.cpu()
            personalized_accs.append(float((fused_scores.argmax(dim=1).cpu() == y_test).float().mean().item()))
            shared_branch_accs.append(float((shared_scores.argmax(dim=1).cpu() == y_test).float().mean().item()))
            personal_branch_accs.append(float((personal_scores.argmax(dim=1).cpu() == y_test).float().mean().item()))

        mean_personalized_accuracy = sum(personalized_accs) / max(len(personalized_accs), 1)
        metrics = {
            "mean_personalized_accuracy": mean_personalized_accuracy,
            "mean_shared_branch_accuracy": sum(shared_branch_accs) / max(len(shared_branch_accs), 1),
            "mean_personal_branch_accuracy": sum(personal_branch_accs) / max(len(personal_branch_accs), 1),
            "mean_alpha": sum(alpha_values) / max(len(alpha_values), 1),
        }
        if self._include_mean_local_test_accuracy():
            metrics["mean_local_test_accuracy"] = mean_personalized_accuracy
        return metrics
