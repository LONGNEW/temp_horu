from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import nn


def detached_state_dict(module_or_state, *, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    state = module_or_state.state_dict() if hasattr(module_or_state, "state_dict") else module_or_state
    target_device = torch.device(device)
    return {name: tensor.detach().to(target_device).clone() for name, tensor in state.items()}


def average_state_dicts(states: list[dict[str, torch.Tensor]], weights: Iterable[float] | None = None) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("Cannot average an empty list of state dicts.")
    if weights is None:
        weights = [1.0] * len(states)

    weight_tensor = torch.tensor(list(weights), dtype=torch.float64)
    weight_tensor = weight_tensor / weight_tensor.sum().clamp_min(1e-12)

    averaged: dict[str, torch.Tensor] = {}
    for name in states[0].keys():
        first = states[0][name].detach()
        accumulated = first.mul(weight_tensor[0].to(device=first.device, dtype=first.dtype))
        for state, weight in zip(states[1:], weight_tensor[1:]):
            accumulated.add_(state[name].detach(), alpha=float(weight))
        averaged[name] = accumulated
    return averaged


def blend_state_dicts(
    base_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    *,
    alpha: float,
) -> dict[str, torch.Tensor]:
    mixed: dict[str, torch.Tensor] = {}
    for name, tensor in base_state.items():
        mixed[name] = (1.0 - float(alpha)) * tensor.detach() + float(alpha) * target_state[name].detach().to(tensor.device)
    return mixed


def build_optimizer(
    params,
    *,
    lr: float,
    optimizer_name: str = "sgd",
    momentum: float = 0.0,
    weight_decay: float = 0.0,
):
    if optimizer_name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def minibatches(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool = True,
):
    if x.shape[0] == 0:
        return
    if shuffle:
        indices = torch.randperm(x.shape[0], device=x.device)
    else:
        indices = torch.arange(x.shape[0], device=x.device)
    for start in range(0, x.shape[0], batch_size):
        batch_idx = indices[start:start + batch_size]
        yield x[batch_idx], y[batch_idx]


def _prepare_prox_state(prox_state: Mapping[str, torch.Tensor] | None, device: torch.device) -> dict[str, torch.Tensor] | None:
    if prox_state is None:
        return None
    return {name: tensor.detach().to(device) for name, tensor in prox_state.items()}


def _prox_penalty(model: nn.Module, prox_state: Mapping[str, torch.Tensor]) -> torch.Tensor:
    penalty = torch.zeros((), device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        penalty = penalty + torch.sum((param - prox_state[name]) ** 2)
    return penalty


def train_supervised_epoch(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    lr: float,
    batch_size: int,
    optimizer_name: str = "sgd",
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    prox_state: Mapping[str, torch.Tensor] | None = None,
    prox_mu: float = 0.0,
) -> dict[str, float]:
    model.train()
    device = next(model.parameters()).device
    optimizer = build_optimizer(
        model.parameters(),
        lr=float(lr),
        optimizer_name=optimizer_name,
        momentum=float(momentum),
        weight_decay=float(weight_decay),
    )
    prox_state_device = _prepare_prox_state(prox_state, next(model.parameters()).device)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for xb, yb in minibatches(x, y, batch_size=batch_size, shuffle=True):
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        if prox_state_device is not None and prox_mu > 0.0:
            loss = loss + 0.5 * float(prox_mu) * _prox_penalty(model, prox_state_device)
        loss.backward()
        optimizer.step()

        total_examples += int(yb.shape[0])
        total_loss += float(loss.item()) * int(yb.shape[0])
        total_correct += int((logits.argmax(dim=1) == yb).sum().item())

    if total_examples == 0:
        return {"loss": 0.0, "accuracy": 0.0}
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


@torch.no_grad()
def evaluate_model(model: nn.Module, x: torch.Tensor, y: torch.Tensor, *, batch_size: int = 256) -> float:
    model.eval()
    device = next(model.parameters()).device
    total_correct = 0
    total_examples = 0
    for xb, yb in minibatches(x, y, batch_size=batch_size, shuffle=False):
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        total_examples += int(yb.shape[0])
        total_correct += int((logits.argmax(dim=1) == yb).sum().item())
    if total_examples == 0:
        return 0.0
    return total_correct / total_examples


def set_module_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def _dfl_disentangle_penalty(invariant: torch.Tensor, specific: torch.Tensor) -> torch.Tensor:
    invariant_z = F.normalize(invariant, dim=1)
    specific_z = F.normalize(specific, dim=1)
    return torch.mean((invariant_z * specific_z).sum(dim=1) ** 2)


def train_dfl_epoch(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    lr: float,
    batch_size: int,
    global_invariant_state: Mapping[str, torch.Tensor],
    align_weight: float,
    disentangle_weight: float,
    optimizer_name: str = "sgd",
    momentum: float = 0.0,
    weight_decay: float = 0.0,
) -> dict[str, float]:
    device = next(model.parameters()).device
    reference_branch = copy.deepcopy(model.global_branch)
    reference_branch.load_state_dict(global_invariant_state)
    reference_branch.to(device)
    reference_branch.eval()

    model.train()

    set_module_requires_grad(model.global_branch, False)
    set_module_requires_grad(model.local_branch, True)
    set_module_requires_grad(model.head, True)
    specific_optimizer = build_optimizer(
        list(model.local_branch.parameters()) + list(model.head.parameters()),
        lr=float(lr),
        optimizer_name=optimizer_name,
        momentum=float(momentum),
        weight_decay=float(weight_decay),
    )
    for xb, yb in minibatches(x, y, batch_size=batch_size, shuffle=True):
        xb = xb.to(device)
        yb = yb.to(device)
        specific_optimizer.zero_grad()
        logits, invariant, specific = model.forward_branches(xb)
        loss = F.cross_entropy(logits, yb) + float(disentangle_weight) * _dfl_disentangle_penalty(invariant, specific)
        loss.backward()
        specific_optimizer.step()

    set_module_requires_grad(model.global_branch, True)
    set_module_requires_grad(model.local_branch, False)
    set_module_requires_grad(model.head, True)
    invariant_optimizer = build_optimizer(
        list(model.global_branch.parameters()) + list(model.head.parameters()),
        lr=float(lr),
        optimizer_name=optimizer_name,
        momentum=float(momentum),
        weight_decay=float(weight_decay),
    )
    for xb, yb in minibatches(x, y, batch_size=batch_size, shuffle=True):
        xb = xb.to(device)
        yb = yb.to(device)
        invariant_optimizer.zero_grad()
        logits, invariant, specific = model.forward_branches(xb)
        with torch.no_grad():
            if hasattr(model, "forward_global_branch"):
                reference_invariant = model.forward_global_branch(xb, branch=reference_branch)
            else:
                reference_invariant = reference_branch(xb)
        loss = (
            F.cross_entropy(logits, yb)
            + float(align_weight) * F.mse_loss(invariant, reference_invariant)
            + float(disentangle_weight) * _dfl_disentangle_penalty(invariant, specific)
        )
        loss.backward()
        invariant_optimizer.step()

    set_module_requires_grad(model.global_branch, True)
    set_module_requires_grad(model.local_branch, True)
    set_module_requires_grad(model.head, True)
    return {"accuracy": evaluate_model(model, x, y, batch_size=batch_size)}
