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


def _lowrank_right_vectors(matrix: torch.Tensor, rank: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top right singular vectors via deterministic CPU low-rank SVD."""
    if rank <= 0:
        return torch.empty(0, dtype=matrix.dtype, device=matrix.device), torch.empty((matrix.shape[1], 0), dtype=matrix.dtype, device=matrix.device)
    source_device = matrix.device
    cpu = matrix.detach().to("cpu", dtype=torch.float32).contiguous()
    q = min(min(cpu.shape), rank + min(8, max(0, min(cpu.shape) - rank)))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        _, singular, right = torch.svd_lowrank(cpu, q=q, niter=2)
    order = torch.argsort(singular, descending=True)
    singular = singular[order][:rank]
    basis = canonicalize_signs(right[:, order][:, :rank]).to(device=source_device, dtype=matrix.dtype)
    return singular.to(device=source_device, dtype=matrix.dtype), basis


def _complete_basis(existing: torch.Tensor, rank: int, seed: int) -> torch.Tensor:
    """Deterministically complete an orthonormal basis without full-matrix SVD."""
    if existing.shape[1] >= rank:
        return existing[:, :rank]
    dimension = existing.shape[0]
    completed = existing
    attempt = 0
    tolerance = torch.finfo(existing.dtype).eps * max(1, dimension)
    while completed.shape[1] < rank:
        need = rank - completed.shape[1]
        width = min(dimension, max(need + 8, need * 2))
        with torch.random.fork_rng(devices=[]):
            generator = torch.Generator(device="cpu").manual_seed(seed + attempt)
            candidates = torch.randn((dimension, width), generator=generator, dtype=torch.float32)
        projected = candidates.to(device=completed.device, dtype=completed.dtype)
        if completed.numel():
            projected = projected - completed @ (completed.T @ projected)
        keep = torch.linalg.vector_norm(projected, dim=0) > tolerance
        if not bool(torch.any(keep)):
            attempt += 1
            continue
        fresh = torch.linalg.qr(projected[:, keep], mode="reduced").Q
        fresh = canonicalize_signs(fresh)
        if completed.numel():
            fresh = fresh - completed @ (completed.T @ fresh)
            keep = torch.linalg.vector_norm(fresh, dim=0) > tolerance
            fresh = torch.linalg.qr(fresh[:, keep], mode="reduced").Q if bool(torch.any(keep)) else fresh[:, :0]
        if fresh.shape[1] == 0:
            attempt += 1
            continue
        completed = torch.cat([completed, fresh[:, :need]], dim=1)
        attempt += 1
    return completed[:, :rank]


def shared_basis(prototypes: list[torch.Tensor], common_rank: int, global_rank: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Return canonical common/global bases from a deterministic low-rank solve."""
    dimension = prototypes[0].shape[1]
    total_rank = common_rank + global_rank
    if total_rank > dimension:
        raise ValueError("shared rank exceeds HD dimension")
    stacked = torch.cat(prototypes, dim=0)
    singular, basis = _lowrank_right_vectors(stacked, total_rank, seed=104729 + dimension * 17 + total_rank)
    total_energy = torch.linalg.matrix_norm(stacked, ord="fro").pow(2)
    selected = singular.pow(2)
    metrics = {
        "basis_solver": "svd_lowrank_right_vectors",
        "selected_eigenvalues": selected.detach().cpu().tolist(),
        "explained_energy_ratio": float((selected.clamp_min(0).sum() / total_energy).item()) if float(total_energy.item()) > 0 else 0.0,
    }
    return basis[:, :common_rank], basis[:, common_rank:], metrics


def personal_basis(residual: torch.Tensor, rank: int, policy: str) -> tuple[torch.Tensor, dict]:
    """Build a deterministic local basis, exposing the completion policy."""
    if policy == "reduced_svd":
        _, singular, vh = torch.linalg.svd(residual, full_matrices=False)
        available = vh.shape[0]
        if rank > available:
            raise ValueError(f"{policy} provides only {available} right singular vectors")
        basis = canonicalize_signs(vh[:rank].T)
        completion = "TASK_T004_REDUCED_SVD"
    else:
        available = min(residual.shape)
        singular, selected = _lowrank_right_vectors(residual, min(rank, available), seed=130363 + residual.shape[0] * 31 + residual.shape[1])
        basis = _complete_basis(selected, rank, seed=161803 + residual.shape[0] * 43 + residual.shape[1])
        completion = "USER_SPECIFIED_NUMERICAL_COMPLETION"
    tol = torch.finfo(residual.dtype).eps * max(residual.shape) * (float(singular[0]) if singular.numel() else 0.0)
    effective_rank = int((singular > tol).sum().item())
    return basis, {
        "personal_basis_policy": policy,
        "personal_basis_solver": "svd_lowrank_with_completion" if policy == "full_svd" else "svd_reduced",
        "personal_singular_values": singular.detach().cpu().tolist(),
        "effective_numerical_rank": effective_rank,
        "zero_singular_directions_selected": max(0, rank - effective_rank),
        "completion_provenance": completion,
        "personal_projector_sha256": tensor_sha256(basis @ basis.T),
    }
