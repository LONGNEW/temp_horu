"""Direct coefficient-space HoRU prediction."""
from __future__ import annotations
import torch


def coefficient_gram(common_basis: torch.Tensor, global_basis: torch.Tensor, personal_basis: torch.Tensor) -> torch.Tensor:
    """Return B.T @ B for B=[B_c,B_g,B_p]; this retains basis-overlap scale."""
    basis = torch.cat([common_basis, global_basis, personal_basis], dim=1)
    return basis.T @ basis


def scores(z_c: torch.Tensor, z_g: torch.Tensor, z_p: torch.Tensor, common: torch.Tensor, global_coefficients: torch.Tensor, delta: torch.Tensor, personal: torch.Tensor, gram: torch.Tensor) -> torch.Tensor:
    """Direct dot products between one query and all class coefficients."""
    query = torch.cat([z_c, z_g, z_p])
    rows = torch.cat([common + delta, global_coefficients, personal], dim=1)
    return rows @ query


def batch_scores(z_c: torch.Tensor, z_g: torch.Tensor, z_p: torch.Tensor, common: torch.Tensor, global_coefficients: torch.Tensor, delta: torch.Tensor, personal: torch.Tensor, gram: torch.Tensor) -> torch.Tensor:
    """Vectorized direct dot products for a batch of coefficient queries."""
    queries = torch.cat([z_c, z_g, z_p], dim=1)
    rows = torch.cat([common + delta, global_coefficients, personal], dim=1)
    return queries @ rows.T


def predict_batch(z_c: torch.Tensor, z_g: torch.Tensor, z_p: torch.Tensor, common: torch.Tensor, global_coefficients: torch.Tensor, delta: torch.Tensor, personal: torch.Tensor, gram: torch.Tensor) -> torch.Tensor:
    return torch.argmax(
        batch_scores(z_c, z_g, z_p, common, global_coefficients, delta, personal, gram),
        dim=1,
    )


def predict(z_c: torch.Tensor, z_g: torch.Tensor, z_p: torch.Tensor, common: torch.Tensor, global_coefficients: torch.Tensor, delta: torch.Tensor, personal: torch.Tensor, gram: torch.Tensor) -> int:
    return int(torch.argmax(scores(z_c, z_g, z_p, common, global_coefficients, delta, personal, gram)).item())
