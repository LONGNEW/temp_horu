"""One-time HoRU shared/local decomposition and coefficient-cache bootstrap."""
from __future__ import annotations
import time
import torch
import torch.nn.functional as F
from .basis import personal_basis, shared_basis, tensor_sha256
from .state import ClientBootstrapState


def class_prototype(encoded: torch.Tensor, labels: torch.Tensor, num_classes: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    counts = torch.bincount(labels, minlength=num_classes).to(torch.long)
    prototype = torch.zeros((num_classes, encoded.shape[1]), dtype=encoded.dtype, device=encoded.device)
    for label in range(num_classes):
        rows = encoded[labels == label]
        if rows.numel():
            prototype[label] = rows.mean(dim=0)
    norms = torch.linalg.vector_norm(prototype, dim=1, keepdim=True)
    prototype = torch.where(norms > 0, prototype / norms.clamp_min(torch.finfo(prototype.dtype).eps), prototype)
    return prototype, counts


def _cache(encoded: torch.Tensor, common: torch.Tensor, global_basis: torch.Tensor, personal: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"z_c": encoded @ common, "z_g": encoded @ global_basis, "z_p": encoded @ personal}


def bootstrap_horu(clients: dict[int, dict[str, torch.Tensor]], common_rank: int, global_rank: int, personal_rank: int, personal_policy: str, num_classes: int = 6) -> tuple[dict[int, ClientBootstrapState], torch.Tensor, torch.Tensor, dict, list[dict], list[dict]]:
    """Bootstrap only; no coefficient retraining, aggregation rounds, or personalization update."""
    prototype_ms: dict[int, float] = {}; prototypes = {}; counts = {}
    for cid, client in clients.items():
        begun = time.perf_counter_ns(); prototypes[cid], counts[cid] = class_prototype(client["train_h"], client["train_y"], num_classes); prototype_ms[cid] = (time.perf_counter_ns() - begun) / 1e6
    begun = time.perf_counter_ns(); common, global_basis, basis_info = shared_basis(list(prototypes.values()), common_rank, global_rank); basis_ms = (time.perf_counter_ns() - begun) / 1e6
    begun = time.perf_counter_ns()
    common_totals = {cid: prototypes[cid] @ common for cid in clients}
    global_totals = {cid: prototypes[cid] @ global_basis for cid in clients}
    projection_ms = (time.perf_counter_ns() - begun) / 1e6
    begun = time.perf_counter_ns()
    denominator = torch.stack(list(counts.values())).sum(dim=0)

    def weighted_consensus(values: dict[int, torch.Tensor]) -> torch.Tensor:
        sample = next(iter(values.values()))
        numerator = sum(
            (counts[cid].to(prototypes[cid].dtype)[:, None] * values[cid] for cid in clients),
            torch.zeros_like(sample),
        )
        return torch.where(
            denominator[:, None] > 0,
            numerator / denominator[:, None].clamp_min(1),
            numerator,
        )

    consensus = weighted_consensus(common_totals)
    global_consensus = weighted_consensus(global_totals)
    coefficients_ms = (time.perf_counter_ns() - begun) / 1e6
    states: dict[int, ClientBootstrapState] = {}; client_rows = []; recon_rows = []
    for cid, client in clients.items():
        begun = time.perf_counter_ns()
        residual = prototypes[cid] - (
            (common_totals[cid] @ common.T)
            + (global_consensus @ global_basis.T)
        )
        residual_ms = (time.perf_counter_ns() - begun) / 1e6
        begun = time.perf_counter_ns(); personal, personal_info = personal_basis(residual, personal_rank, personal_policy); svd_ms = (time.perf_counter_ns() - begun) / 1e6
        begun = time.perf_counter_ns(); coefficients = residual @ personal; coeff_ms = (time.perf_counter_ns() - begun) / 1e6
        begun = time.perf_counter_ns(); train_cache = _cache(client["train_h"], common, global_basis, personal); test_cache = _cache(client["test_h"], common, global_basis, personal); cache_ms = (time.perf_counter_ns() - begun) / 1e6
        state = ClientBootstrapState(cid, prototypes[cid], counts[cid], consensus.clone(), global_consensus.clone(), torch.zeros((num_classes, common_rank), device=common.device), personal, coefficients, train_cache, test_cache, client["train_y"], client["test_y"]); states[cid] = state
        client_total = residual_ms + svd_ms + coeff_ms + cache_ms
        client_rows.append({"timing_scope": "table_i.client", "client_id": cid, "pre_table_i_prototype_ms": prototype_ms[cid], "residual_construction_ms": residual_ms, "personal_basis_svd_ms": svd_ms, "residual_coefficient_projection_ms": coeff_ms, "query_coefficient_cache_ms": cache_ms, "client_bootstrap_ms": client_total, "class_counts": counts[cid].cpu().tolist(), "residual_common_orthogonality_max_abs": float((residual @ common).abs().max().item()), "residual_global_mismatch_max_abs": float((residual @ global_basis).abs().max().item()) if global_rank else 0.0, **personal_info})
        # Reconstruction quality is diagnostic-only and is not included in
        # client_bootstrap_ms or any aggregate latency.
        reconstructed = consensus @ common.T + global_consensus @ global_basis.T + coefficients @ personal.T
        reconstructed = F.normalize(reconstructed, p=2, dim=1, eps=torch.finfo(reconstructed.dtype).eps)
        nonempty = counts[cid] > 0; cosine = F.cosine_similarity(reconstructed[nonempty], prototypes[cid][nonempty], dim=1)
        recon_rows.append({"client_id": cid, "finite": bool(torch.isfinite(reconstructed).all().item()), "nonempty_row_norm_min": float(torch.linalg.vector_norm(reconstructed[nonempty], dim=1).min().item()), "prototype_reconstruction_cosine_mean": float(cosine.mean().item()), "prototype_reconstruction_error_mean": float(torch.linalg.vector_norm(reconstructed[nonempty] - prototypes[cid][nonempty], dim=1).mean().item()), "train_cache_shapes": str({k: list(v.shape) for k,v in train_cache.items()}), "test_cache_shapes": str({k: list(v.shape) for k,v in test_cache.items()})})
    client_sum, client_max = sum(x["client_bootstrap_ms"] for x in client_rows), max(x["client_bootstrap_ms"] for x in client_rows)
    server_total = basis_ms + projection_ms + coefficients_ms
    raw_basis = torch.cat([common, global_basis], dim=1)
    metrics = {**basis_info, "timing_scope": "table_i.server", "raw_basis_sha256": tensor_sha256(raw_basis), "projector_sha256": tensor_sha256(raw_basis @ raw_basis.T), "orthogonality_max_abs_error": float((raw_basis.T @ raw_basis - torch.eye(common_rank + global_rank, device=raw_basis.device, dtype=raw_basis.dtype)).abs().max().item()) if common_rank + global_rank else 0.0, "common_orthogonality_max_abs_error": float((common.T @ common - torch.eye(common_rank, device=common.device)).abs().max().item()), "global_orthogonality_max_abs_error": float((global_basis.T @ global_basis - torch.eye(global_rank, device=common.device)).abs().max().item()) if global_rank else 0.0, "cross_orthogonality_max_abs_error": float((common.T @ global_basis).abs().max().item()) if global_rank else 0.0, "client_prototype_hashes": {str(cid): tensor_sha256(state.prototype) for cid,state in states.items()}, "common_consensus_sha256": tensor_sha256(consensus), "global_consensus_sha256": tensor_sha256(global_consensus), "client_bootstrap_sum_ms": client_sum, "client_bootstrap_max_ms": client_max, "server_common_global_basis_ms": basis_ms, "server_client_hv_projection_ms": projection_ms, "server_common_global_coefficients_ms": coefficients_ms, "server_bootstrap_total_ms": server_total}
    return states, common, global_basis, metrics, client_rows, recon_rows
