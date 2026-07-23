"""Deterministic basis construction for HoRU bootstrap."""
from __future__ import annotations
import hashlib
import torch


def tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().to("cpu").contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def canonicalize_signs(basis: torch.Tensor) -> torch.Tensor:
    """Choose a deterministic sign per column; first max-absolute index wins ties."""
    result = basis.clone()
    for column in range(result.shape[1]):
        index = int(torch.argmax(result[:, column].abs()).item())
        if result[index, column] < 0:
            result[:, column].neg_()
    return result


def shared_basis(prototypes: list[torch.Tensor], common_rank: int, global_rank: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Return canonical common/global bases from sum_i M_i^T M_i."""
    dimension = prototypes[0].shape[1]
    total_rank = common_rank + global_rank
    if total_rank > dimension:
        raise ValueError("shared rank exceeds HD dimension")
    covariance = sum((item.T @ item for item in prototypes), torch.zeros((dimension, dimension), device=prototypes[0].device, dtype=prototypes[0].dtype))
    values, vectors = torch.linalg.eigh(covariance)
    order = torch.argsort(values, descending=True)
    values, vectors = values[order], canonicalize_signs(vectors[:, order])
    basis = vectors[:, :total_rank]
    energy = values.clamp_min(0).sum()
    selected = values[:total_rank]
    metrics = {"covariance_sha256": tensor_sha256(covariance), "selected_eigenvalues": selected.detach().cpu().tolist(), "explained_energy_ratio": float((selected.clamp_min(0).sum() / energy).item()) if energy > 0 else 0.0, "raw_basis_sha256": tensor_sha256(basis), "projector_sha256": tensor_sha256(basis @ basis.T), "orthogonality_max_abs_error": float((basis.T @ basis - torch.eye(total_rank, device=basis.device)).abs().max().item()) if total_rank else 0.0}
    return basis[:, :common_rank], basis[:, common_rank:], metrics


def personal_basis(residual: torch.Tensor, rank: int, policy: str) -> tuple[torch.Tensor, dict]:
    """Build a deterministic local basis, exposing the full-SVD completion policy."""
    full = policy == "full_svd"
    _, singular, vh = torch.linalg.svd(residual, full_matrices=full)
    available = vh.shape[0]
    if rank > available:
        raise ValueError(f"{policy} provides only {available} right singular vectors")
    basis = canonicalize_signs(vh[:rank].T)
    tol = torch.finfo(residual.dtype).eps * max(residual.shape) * (float(singular[0]) if singular.numel() else 0.0)
    effective_rank = int((singular > tol).sum().item())
    return basis, {"personal_basis_policy": policy, "personal_singular_values": singular.detach().cpu().tolist(), "effective_numerical_rank": effective_rank, "zero_singular_directions_selected": max(0, rank - effective_rank), "completion_provenance": "USER_SPECIFIED_NUMERICAL_COMPLETION" if full else "TASK_T004_REDUCED_SVD", "personal_projector_sha256": tensor_sha256(basis @ basis.T)}
