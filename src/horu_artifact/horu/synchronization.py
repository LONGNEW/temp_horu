"""Shared-only HoRU aggregation and error-ratio absorption."""
from __future__ import annotations
import time
import torch
from .state import ClientBootstrapState


def aggregate_shared(states: list[ClientBootstrapState]) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    started = time.perf_counter_ns(); counts = torch.stack([state.class_counts for state in states]).to(states[0].common.dtype)
    denominator = counts.sum(dim=0)
    common = sum((counts[i, :, None] * state.common for i, state in enumerate(states)), torch.zeros_like(states[0].common))
    common = torch.where(denominator[:, None] > 0, common / denominator[:, None].clamp_min(1), torch.zeros_like(common)); common_ms = (time.perf_counter_ns() - started) / 1e6
    started = time.perf_counter_ns(); global_coefficients = sum((counts[i, :, None] * state.global_coefficients for i, state in enumerate(states)), torch.zeros_like(states[0].global_coefficients))
    global_coefficients = torch.where(denominator[:, None] > 0, global_coefficients / denominator[:, None].clamp_min(1), torch.zeros_like(global_coefficients)); global_ms = (time.perf_counter_ns() - started) / 1e6
    return common, global_coefficients, {"common_aggregation_ms": common_ms, "global_aggregation_ms": global_ms, "server_aggregation_total_ms": common_ms + global_ms}


def follow_ratio(class_total_counts: torch.Tensor, class_wrong_counts: torch.Tensor, gate_alpha: float = 1.0, gate_min: float = 0.1, gate_max: float = 0.9) -> torch.Tensor:
    error_ratio = class_wrong_counts.to(torch.float32) / class_total_counts.to(torch.float32).clamp_min(1.0)
    rollback_gate = (1.0 - (gate_alpha * error_ratio)).clamp(min=gate_min, max=gate_max)
    rollback_gate = torch.where(class_total_counts > 0, rollback_gate, torch.zeros_like(rollback_gate))
    return torch.where(class_total_counts > 0, 1.0 - rollback_gate, torch.zeros_like(rollback_gate))


def absorb_shared(state: ClientBootstrapState, common: torch.Tensor, global_coefficients: torch.Tensor, class_total_counts: torch.Tensor, class_wrong_counts: torch.Tensor, eta_global: float, gate_alpha: float = 1.0, gate_min: float = 0.1, gate_max: float = 0.9) -> float:
    started = time.perf_counter_ns()
    follow = follow_ratio(class_total_counts, class_wrong_counts, gate_alpha, gate_min, gate_max)[:, None].to(state.common.dtype)
    state.common.add_(eta_global * follow * (common - state.common))
    state.global_coefficients.add_(eta_global * follow * (global_coefficients - state.global_coefficients))
    return (time.perf_counter_ns() - started) / 1e6
