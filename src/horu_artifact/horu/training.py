"""Sample-wise coefficient updates and local HoRU diagnostics."""
from __future__ import annotations
import time
import torch
from .inference import predict_batch
from .state import ClientBootstrapState


def train_client(state: ClientBootstrapState, epochs: int, batch_size: int, eta_shared: float, eta_personal: float, gram: torch.Tensor, seed: int, round_id: int) -> tuple[int, dict[str, float]]:
    """Apply the paper's additive coefficient push-pull once per stale batch."""
    cache, labels, updates = state.train_cache, state.train_labels, 0
    similarity_ms = update_ms = 0.0
    for epoch in range(epochs):
        generator = torch.Generator(device="cpu").manual_seed(seed + state.client_id * 1009 + round_id * 100_003 + epoch)
        order = torch.randperm(labels.numel(), generator=generator).to(labels.device)
        for begin in range(0, order.numel(), batch_size):
            indices = order[begin:begin + batch_size]
            started = time.perf_counter_ns()
            predictions = predict_batch(
                cache["z_c"][indices], cache["z_g"][indices], cache["z_p"][indices],
                state.common, state.global_coefficients, state.delta, state.personal, gram,
            )
            similarity_ms += (time.perf_counter_ns() - started) / 1e6
            started = time.perf_counter_ns()
            targets = labels[indices]
            wrong = predictions != targets
            wrong_count = int(wrong.sum().item())
            if wrong_count:
                wrong_indices = indices[wrong]
                wrong_targets = targets[wrong]
                wrong_predictions = predictions[wrong]
                common_update = torch.zeros_like(state.common)
                global_update = torch.zeros_like(state.global_coefficients)
                personal_update = torch.zeros_like(state.personal)
                common_update.index_add_(0, wrong_targets, cache["z_c"][wrong_indices])
                common_update.index_add_(0, wrong_predictions, -cache["z_c"][wrong_indices])
                global_update.index_add_(0, wrong_targets, cache["z_g"][wrong_indices])
                global_update.index_add_(0, wrong_predictions, -cache["z_g"][wrong_indices])
                personal_update.index_add_(0, wrong_targets, cache["z_p"][wrong_indices])
                personal_update.index_add_(0, wrong_predictions, -cache["z_p"][wrong_indices])
                updates += wrong_count
                state.common.add_(common_update, alpha=eta_shared)
                state.global_coefficients.add_(global_update, alpha=eta_shared)
                state.delta.add_(common_update, alpha=eta_personal)
                state.personal.add_(personal_update, alpha=eta_personal)
            update_ms += (time.perf_counter_ns() - started) / 1e6
    return updates, {"coefficient_similarity_ms": similarity_ms, "coefficient_update_ms": update_ms}


def error_statistics(state: ClientBootstrapState, gram: torch.Tensor, num_classes: int = 6, batch_size: int = 32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    labels = state.train_labels
    started = time.perf_counter_ns()
    predictions = torch.cat([
        predict_batch(
            state.train_cache["z_c"][begin:begin + batch_size],
            state.train_cache["z_g"][begin:begin + batch_size],
            state.train_cache["z_p"][begin:begin + batch_size],
            state.common, state.global_coefficients, state.delta, state.personal, gram,
        )
        for begin in range(0, labels.numel(), batch_size)
    ])
    final_prediction_ms = (time.perf_counter_ns() - started) / 1e6
    started = time.perf_counter_ns()
    counts = torch.bincount(labels, minlength=num_classes)
    errors = torch.bincount(labels[predictions != labels], minlength=num_classes)
    ratios = errors.to(torch.float32) / counts.clamp_min(1)
    statistics_ms = (time.perf_counter_ns() - started) / 1e6
    return counts, errors, ratios, final_prediction_ms, statistics_ms


def norm_diagnostics(state: ClientBootstrapState) -> dict[str, float]:
    cache = state.train_cache
    query = torch.cat([cache["z_c"], cache["z_g"], cache["z_p"]], dim=1)
    return {"z_c_norm_mean": float(torch.linalg.vector_norm(cache["z_c"], dim=1).mean()), "z_g_norm_mean": float(torch.linalg.vector_norm(cache["z_g"], dim=1).mean()), "z_p_norm_mean": float(torch.linalg.vector_norm(cache["z_p"], dim=1).mean()), "z_all_norm_mean": float(torch.linalg.vector_norm(query, dim=1).mean()), "common_delta_norm_mean": float(torch.linalg.vector_norm(state.common + state.delta, dim=1).mean()), "global_norm_mean": float(torch.linalg.vector_norm(state.global_coefficients, dim=1).mean()), "personal_norm_mean": float(torch.linalg.vector_norm(state.personal, dim=1).mean())}
