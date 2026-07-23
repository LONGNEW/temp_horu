"""Explicit HoRU shared and local state containers."""
from dataclasses import dataclass
import torch


@dataclass
class ClientBootstrapState:
    client_id: int
    prototype: torch.Tensor
    class_counts: torch.Tensor
    common: torch.Tensor
    global_coefficients: torch.Tensor
    delta: torch.Tensor
    personal_basis: torch.Tensor
    personal: torch.Tensor
    train_cache: dict[str, torch.Tensor]
    test_cache: dict[str, torch.Tensor]
    train_labels: torch.Tensor
    test_labels: torch.Tensor
