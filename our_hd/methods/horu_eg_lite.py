from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..data import ClientData
from ..encoder import BaseHDEncoder, CosineProjectionEncoder
from ..federated import ClientState, FederatedMethod
from .trial.subspace_trial_core import row_normalize
from .trial.subspace_trial_rowgate_v3_bootstrap_ablation import HoRUCoreMethod


def _clone_value(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {str(key): _clone_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    return value


def _pack_state(state: ClientState) -> dict[str, object]:
    return {
        "memory": _clone_value(state.memory),
        "extras": _clone_value(state.extras),
    }


def _unpack_state(payload: dict[str, object]) -> ClientState:
    return ClientState(
        memory=_clone_value(payload.get("memory")),
        extras=_clone_value(payload.get("extras")) if isinstance(payload.get("extras"), dict) else None,
    )


def _allocate_budget(total: int, weights: list[float], *, min_per_slot: int) -> list[int]:
    count = len(weights)
    if count == 0:
        return []
    total = max(int(total), 0)
    min_per_slot = max(int(min_per_slot), 0)
    if total == 0:
        return [0 for _ in range(count)]

    allocation = [0 for _ in range(count)]
    if min_per_slot > 0:
        required = min_per_slot * count
        if total >= required:
            allocation = [min_per_slot for _ in range(count)]
            total -= required
        else:
            order = sorted(range(count), key=lambda idx: weights[idx], reverse=True)
            for idx in order[:total]:
                allocation[idx] += 1
            return allocation

    positive = [max(float(w), 0.0) for w in weights]
    total_positive = sum(positive)
    if total_positive <= 0.0:
        normalized = [1.0 / float(count) for _ in range(count)]
    else:
        normalized = [value / total_positive for value in positive]

    raw = [value * float(total) for value in normalized]
    floors = [int(value) for value in raw]
    allocation = [base + extra for base, extra in zip(allocation, floors)]
    remainder = total - int(sum(floors))
    if remainder > 0:
        fractions = [value - float(floor) for value, floor in zip(raw, floors)]
        order = sorted(range(count), key=lambda idx: fractions[idx], reverse=True)
        for idx in order[:remainder]:
            allocation[idx] += 1
    return allocation


def _normalized_entropy(weights: torch.Tensor, *, eps: float = 1e-8) -> float:
    if int(weights.numel()) <= 1:
        return 0.0
    safe = weights.clamp_min(eps)
    entropy = float((-(safe * safe.log()).sum()).item())
    return entropy / float(torch.log(torch.tensor(float(weights.numel()), device=weights.device)).item())


@dataclass
class _EvidenceGroupSpec:
    name: str
    feature_indices: tuple[int, ...]
    hd_dim: int
    shared_rank: int
    personal_rank: int
    intersection_rank: int


@dataclass
class HoRUEGLiteMethod(FederatedMethod):
    """Evidence-Grouped HoRU Lite (matched-budget grouped decomposition + score fusion)."""

    encoder: BaseHDEncoder
    num_classes: int
    shared_rank: int = 32
    personal_rank: int = 64
    local_epochs: int = 3
    batch_size: int = 32
    global_lr: float = 0.035
    personal_lr: float = 0.035
    val_fraction: float = 0.0
    gate_alpha: float = 1.0
    gate_min: float = 0.1
    gate_max: float = 0.9
    intersection_rank: int = 24
    intersection_ratio: float | None = None
    debug: bool = False
    eg_group_preset: str = "auto"
    eg_num_groups: int = 4
    eg_weight_temperature: float = 0.35
    eg_weight_prior_blend: float = 0.5
    eg_weight_update_momentum: float = 0.35
    eg_enable_interactions: bool = True
    eg_interaction_weight: float = 0.10
    eg_interaction_pairs: tuple[str, ...] = ("hand+ankle", "chest+ankle")

    def __post_init__(self) -> None:
        self._group_specs: list[_EvidenceGroupSpec] = []
        self._group_methods: dict[str, HoRUCoreMethod] = {}
        self._group_base_weights = torch.empty(0, device=self.encoder.device, dtype=torch.float32)
        self._group_index_by_name: dict[str, int] = {}
        self._interaction_pairs: list[tuple[int, int]] = []
        self._client_group_cache: dict[str, dict[str, ClientData]] = {}
        self._input_dim: int | None = None

    def _merge_groups_for_shared_budget(
        self,
        groups: list[tuple[str, list[int]]],
    ) -> list[tuple[str, list[int]]]:
        max_groups = max(1, int(self.shared_rank))
        merged = [(name, list(indices)) for name, indices in groups]
        while len(merged) > max_groups and len(merged) > 1:
            left_name, left_indices = merged[0]
            right_name, right_indices = merged[1]
            merged[0] = (f"{left_name}+{right_name}", left_indices + right_indices)
            del merged[1]
        return merged

    def _resolve_raw_groups(self, input_dim: int) -> list[tuple[str, list[int]]]:
        preset = str(self.eg_group_preset).lower()
        if preset in {"auto", "pamap2", "pamap2_default"} and int(input_dim) == 52:
            return [
                ("hr", [0]),
                ("hand", list(range(1, 18))),
                ("chest", list(range(18, 35))),
                ("ankle", list(range(35, 52))),
            ]

        requested = max(1, min(int(self.eg_num_groups), int(input_dim)))
        requested = min(requested, max(1, int(self.shared_rank)))
        chunks = torch.tensor_split(torch.arange(input_dim, dtype=torch.long), requested)
        groups: list[tuple[str, list[int]]] = []
        for idx, chunk in enumerate(chunks):
            indices = [int(value) for value in chunk.tolist()]
            if not indices:
                continue
            groups.append((f"group_{idx + 1}", indices))
        if not groups:
            groups.append(("group_1", [int(v) for v in range(input_dim)]))
        return groups

    def _resolve_intersection_ratio(self) -> float:
        if self.intersection_ratio is not None:
            return min(max(float(self.intersection_ratio), 0.0), 1.0)
        shared_rank = max(int(self.shared_rank), 1)
        ratio = float(self.intersection_rank) / float(shared_rank)
        return min(max(ratio, 0.0), 1.0)

    def _resolve_interaction_pairs(self) -> list[tuple[int, int]]:
        if not bool(self.eg_enable_interactions) or len(self._group_specs) <= 1:
            return []
        name_to_index = self._group_index_by_name
        pairs: list[tuple[int, int]] = []
        for token in self.eg_interaction_pairs:
            text = str(token).strip().lower()
            if "+" not in text:
                continue
            left_name, right_name = [piece.strip() for piece in text.split("+", 1)]
            left = name_to_index.get(left_name)
            right = name_to_index.get(right_name)
            if left is None or right is None or left == right:
                continue
            pair = (min(left, right), max(left, right))
            if pair not in pairs:
                pairs.append(pair)
        if pairs:
            return pairs

        # Fallback: choose up to two pairs from highest-prior groups.
        order = torch.argsort(self._group_base_weights, descending=True).tolist()
        fallback: list[tuple[int, int]] = []
        for left_pos in range(len(order)):
            for right_pos in range(left_pos + 1, len(order)):
                pair = (int(order[left_pos]), int(order[right_pos]))
                fallback.append(pair)
                if len(fallback) >= 2:
                    return fallback
        return fallback

    def _build_group_runtime(self, input_dim: int) -> None:
        if self._input_dim is not None:
            if int(input_dim) != int(self._input_dim):
                raise ValueError(f"Input dimension changed from {self._input_dim} to {input_dim}")
            return

        if int(self.shared_rank) <= 0:
            raise ValueError("HoRUEGLiteMethod requires shared_rank > 0.")
        raw_groups = self._resolve_raw_groups(int(input_dim))
        groups = self._merge_groups_for_shared_budget(raw_groups)
        group_sizes = [len(indices) for _, indices in groups]
        if not group_sizes:
            raise ValueError("No evidence groups were resolved.")

        hd_total = int(self.encoder.hd_dim)
        hd_dims = _allocate_budget(hd_total, [float(size) for size in group_sizes], min_per_slot=1)
        shared_ranks = _allocate_budget(int(self.shared_rank), [float(size) for size in group_sizes], min_per_slot=1)
        personal_ranks = _allocate_budget(int(self.personal_rank), [float(size) for size in group_sizes], min_per_slot=0)
        overlap_ratio = self._resolve_intersection_ratio()

        self._group_specs = []
        self._group_methods = {}
        self._group_index_by_name = {}
        random_phase = bool(getattr(self.encoder, "random_phase", False))
        binary = bool(getattr(self.encoder, "binary", False))
        for idx, ((name, indices), hd_dim, shared, personal) in enumerate(
            zip(groups, hd_dims, shared_ranks, personal_ranks)
        ):
            normalized_name = str(name).strip().lower()
            intersection = int(round(float(shared) * overlap_ratio))
            intersection = min(max(intersection, 0), int(shared))
            spec = _EvidenceGroupSpec(
                name=normalized_name,
                feature_indices=tuple(int(value) for value in indices),
                hd_dim=int(hd_dim),
                shared_rank=int(shared),
                personal_rank=int(personal),
                intersection_rank=int(intersection),
            )
            self._group_specs.append(spec)
            self._group_index_by_name[normalized_name] = idx
            self._group_methods[normalized_name] = HoRUCoreMethod(
                encoder=CosineProjectionEncoder(
                    input_dim=len(spec.feature_indices),
                    hd_dim=spec.hd_dim,
                    binary=binary,
                    random_phase=random_phase,
                    device=self.encoder.device,
                ),
                num_classes=int(self.num_classes),
                shared_rank=spec.shared_rank,
                personal_rank=spec.personal_rank,
                local_epochs=int(self.local_epochs),
                batch_size=int(self.batch_size),
                global_lr=float(self.global_lr),
                personal_lr=float(self.personal_lr),
                val_fraction=float(self.val_fraction),
                gate_alpha=float(self.gate_alpha),
                gate_min=float(self.gate_min),
                gate_max=float(self.gate_max),
                intersection_rank=spec.intersection_rank,
                intersection_ratio=None,
                refresh_interval=0,
                debug=bool(self.debug),
            )
        self._group_base_weights = torch.tensor(
            group_sizes,
            device=self.encoder.device,
            dtype=torch.float32,
        )
        self._group_base_weights = self._group_base_weights / self._group_base_weights.sum().clamp_min(1e-8)
        self._interaction_pairs = self._resolve_interaction_pairs()
        self._input_dim = int(input_dim)

    def _ensure_runtime(self, client: ClientData) -> None:
        self._build_group_runtime(int(client.x_train.shape[1]))

    def _group_client_view(self, client: ClientData, spec: _EvidenceGroupSpec) -> ClientData:
        per_client = self._client_group_cache.setdefault(client.client_id, {})
        cached = per_client.get(spec.name)
        if cached is not None:
            return cached
        feature_index = torch.tensor(
            spec.feature_indices,
            device=client.x_train.device,
            dtype=torch.long,
        )
        view = ClientData(
            client_id=client.client_id,
            x_train=client.x_train.index_select(1, feature_index),
            y_train=client.y_train,
            x_test=client.x_test.index_select(1, feature_index),
            y_test=client.y_test,
        )
        per_client[spec.name] = view
        return view

    def _weights_from_extras(self, extras: dict[str, Any] | None) -> torch.Tensor:
        if extras is None:
            return self._group_base_weights.detach().clone()
        weights = extras.get("eg_group_weights")
        if not isinstance(weights, torch.Tensor):
            return self._group_base_weights.detach().clone()
        weights = weights.to(self.encoder.device, dtype=torch.float32)
        if int(weights.numel()) != len(self._group_specs):
            return self._group_base_weights.detach().clone()
        total = float(weights.sum().item())
        if total <= 1e-12:
            return self._group_base_weights.detach().clone()
        return weights / weights.sum().clamp_min(1e-8)

    def _compose_memory(
        self,
        group_states: dict[str, dict[str, object]],
        weights: torch.Tensor,
    ) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        for idx, spec in enumerate(self._group_specs):
            packed = group_states.get(spec.name)
            if not isinstance(packed, dict):
                continue
            memory = packed.get("memory")
            if not isinstance(memory, torch.Tensor):
                continue
            scaled = torch.sqrt(weights[idx].clamp_min(0.0)) * memory.to(self.encoder.device)
            chunks.append(scaled)
        if not chunks:
            return torch.zeros(
                int(self.num_classes),
                int(self.encoder.hd_dim),
                device=self.encoder.device,
                dtype=torch.float32,
            )
        return row_normalize(torch.cat(chunks, dim=1))

    def _score_components_for_group(
        self,
        *,
        method: HoRUCoreMethod,
        group_client: ClientData,
        group_state: ClientState,
        split: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        effective_state = method._materialize_state(group_client, group_state, consume=False)
        common_coords, common_delta_coords, global_only_coords, personal_coords, personal_basis = (
            method._state_components(effective_state)
        )
        if split == "train":
            x_hv, y = method._cached_local_train_tensors(group_client, effective_state)
        else:
            x_hv = method.encoder.encode(group_client.x_test)
            y = group_client.y_test.to(method.encoder.device).long()
        full_scores, shared_scores, personal_scores, _ = method._predict_scores_with_context(
            x_hv,
            extras=effective_state.extras,
            common_coords=common_coords,
            common_delta_coords=common_delta_coords,
            global_only_coords=global_only_coords,
            personal_coords=personal_coords,
            personal_basis=personal_basis,
        )
        mean_common_delta_norm = float(torch.linalg.norm(common_delta_coords, dim=1).mean().item())
        return full_scores, shared_scores, personal_scores, y, mean_common_delta_norm

    def _fused_scores(
        self,
        *,
        full_scores_by_group: list[torch.Tensor],
        shared_scores_by_group: list[torch.Tensor],
        personal_scores_by_group: list[torch.Tensor],
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
        fused = torch.zeros_like(full_scores_by_group[0], device=self.encoder.device)
        shared = torch.zeros_like(shared_scores_by_group[0], device=self.encoder.device)
        personal = torch.zeros_like(personal_scores_by_group[0], device=self.encoder.device)
        for idx in range(len(full_scores_by_group)):
            fused = fused + (weights[idx] * full_scores_by_group[idx])
            shared = shared + (weights[idx] * shared_scores_by_group[idx])
            personal = personal + (weights[idx] * personal_scores_by_group[idx])
        if bool(self.eg_enable_interactions) and float(self.eg_interaction_weight) > 0.0:
            for left, right in self._interaction_pairs:
                pair_weight = float(self.eg_interaction_weight) * float((weights[left] * weights[right]).item())
                fused = fused + (pair_weight * (full_scores_by_group[left] * full_scores_by_group[right]))
        return fused, shared, personal, _normalized_entropy(weights), float(weights.max().item())

    def _aggregate_client_scores(
        self,
        *,
        client: ClientData,
        state: ClientState,
        split: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float, float]:
        assert state.extras is not None
        group_states = state.extras.get("eg_group_states")
        if not isinstance(group_states, dict):
            raise ValueError("Missing eg_group_states in HoRUEGLite state.")

        full_scores_by_group: list[torch.Tensor] = []
        shared_scores_by_group: list[torch.Tensor] = []
        personal_scores_by_group: list[torch.Tensor] = []
        target: torch.Tensor | None = None
        common_delta_norms: list[float] = []

        for spec in self._group_specs:
            method = self._group_methods[spec.name]
            packed = group_states.get(spec.name)
            if not isinstance(packed, dict):
                raise ValueError(f"Missing state for evidence group: {spec.name}")
            group_state = _unpack_state(packed)
            group_client = self._group_client_view(client, spec)
            full_scores, shared_scores, personal_scores, y, mean_common_delta_norm = self._score_components_for_group(
                method=method,
                group_client=group_client,
                group_state=group_state,
                split=split,
            )
            full_scores_by_group.append(full_scores)
            shared_scores_by_group.append(shared_scores)
            personal_scores_by_group.append(personal_scores)
            common_delta_norms.append(mean_common_delta_norm)
            if target is None:
                target = y

        if target is None:
            raise RuntimeError("No group scores available for EG-HoRU aggregation.")
        weights = self._weights_from_extras(state.extras)
        fused, shared, personal, attention_entropy, attention_max = self._fused_scores(
            full_scores_by_group=full_scores_by_group,
            shared_scores_by_group=shared_scores_by_group,
            personal_scores_by_group=personal_scores_by_group,
            weights=weights,
        )
        mean_common_delta_norm = float(sum(common_delta_norms) / len(common_delta_norms)) if common_delta_norms else 0.0
        return fused, shared, personal, target, mean_common_delta_norm, attention_entropy, attention_max

    def _estimate_client_weights(
        self,
        *,
        client: ClientData,
        group_states: dict[str, dict[str, object]],
    ) -> torch.Tensor:
        accuracies: list[float] = []
        for spec in self._group_specs:
            packed = group_states.get(spec.name)
            if not isinstance(packed, dict):
                accuracies.append(0.0)
                continue
            method = self._group_methods[spec.name]
            group_state = _unpack_state(packed)
            group_client = self._group_client_view(client, spec)
            full_scores, _, _, y, _ = self._score_components_for_group(
                method=method,
                group_client=group_client,
                group_state=group_state,
                split="train",
            )
            predictions = full_scores.argmax(dim=1)
            accuracy = float((predictions == y).float().mean().item())
            accuracies.append(accuracy)

        raw = torch.tensor(accuracies, device=self.encoder.device, dtype=torch.float32)
        if int(raw.numel()) == 0:
            return self._group_base_weights.detach().clone()
        if float((raw.max() - raw.min()).item()) <= 1e-8:
            learned = self._group_base_weights.detach().clone()
        else:
            temperature = max(float(self.eg_weight_temperature), 1e-4)
            learned = torch.softmax(raw / temperature, dim=0)

        blend = min(max(float(self.eg_weight_prior_blend), 0.0), 1.0)
        fused = ((1.0 - blend) * learned) + (blend * self._group_base_weights)
        return fused / fused.sum().clamp_min(1e-8)

    def _weights_from_quality(self, quality: torch.Tensor) -> torch.Tensor:
        raw = quality.to(self.encoder.device, dtype=torch.float32)
        if int(raw.numel()) == 0:
            return self._group_base_weights.detach().clone()
        if float((raw.max() - raw.min()).item()) <= 1e-8:
            learned = self._group_base_weights.detach().clone()
        else:
            temperature = max(float(self.eg_weight_temperature), 1e-4)
            learned = torch.softmax(raw / temperature, dim=0)
        blend = min(max(float(self.eg_weight_prior_blend), 0.0), 1.0)
        fused = ((1.0 - blend) * learned) + (blend * self._group_base_weights)
        return fused / fused.sum().clamp_min(1e-8)

    def _updated_client_weights(
        self,
        *,
        previous_weights: torch.Tensor,
        estimated_weights: torch.Tensor,
    ) -> torch.Tensor:
        momentum = min(max(float(self.eg_weight_update_momentum), 0.0), 1.0)
        updated = ((1.0 - momentum) * previous_weights) + (momentum * estimated_weights)
        return updated / updated.sum().clamp_min(1e-8)

    def _packed_group_states(self, extras: dict[str, Any]) -> dict[str, dict[str, object]]:
        group_states = extras.get("eg_group_states")
        if not isinstance(group_states, dict):
            raise ValueError("Missing eg_group_states in client state.")
        packed: dict[str, dict[str, object]] = {}
        for spec in self._group_specs:
            state_payload = group_states.get(spec.name)
            if not isinstance(state_payload, dict):
                raise ValueError(f"Missing group payload for {spec.name}.")
            packed[spec.name] = {
                "memory": _clone_value(state_payload.get("memory")),
                "extras": _clone_value(state_payload.get("extras")),
            }
        return packed

    def init_client_state(self, client: ClientData) -> ClientState:
        self._ensure_runtime(client)
        group_states: dict[str, dict[str, object]] = {}
        for spec in self._group_specs:
            method = self._group_methods[spec.name]
            group_client = self._group_client_view(client, spec)
            group_states[spec.name] = _pack_state(method.init_client_state(group_client))
        weights = self._group_base_weights.detach().clone()
        return ClientState(
            memory=self._compose_memory(group_states, weights),
            extras={
                "eg_group_states": group_states,
                "eg_group_weights": weights.detach().clone(),
            },
        )

    def bootstrap(self, clients: list[ClientData], states: list[ClientState]) -> list[ClientState]:
        if not clients:
            return states
        self._ensure_runtime(clients[0])
        by_group_state_lists: dict[str, list[ClientState]] = {}
        for spec in self._group_specs:
            grouped_states: list[ClientState] = []
            for state in states:
                assert state.extras is not None
                packed = self._packed_group_states(state.extras)[spec.name]
                grouped_states.append(_unpack_state(packed))
            by_group_state_lists[spec.name] = grouped_states

        bootstrapped_by_group: dict[str, list[ClientState]] = {}
        for spec in self._group_specs:
            method = self._group_methods[spec.name]
            group_clients = [self._group_client_view(client, spec) for client in clients]
            bootstrapped_by_group[spec.name] = method.bootstrap(group_clients, by_group_state_lists[spec.name])

        next_states: list[ClientState] = []
        for idx, client in enumerate(clients):
            packed_group_states: dict[str, dict[str, object]] = {}
            for spec in self._group_specs:
                packed_group_states[spec.name] = _pack_state(bootstrapped_by_group[spec.name][idx])
            weights = self._estimate_client_weights(client=client, group_states=packed_group_states)
            next_states.append(
                ClientState(
                    memory=self._compose_memory(packed_group_states, weights),
                    extras={
                        "eg_group_states": packed_group_states,
                        "eg_group_weights": weights.detach().clone(),
                    },
                )
            )
        return next_states

    def client_step(self, client: ClientData, state: ClientState) -> tuple[dict[str, Any], ClientState]:
        self._ensure_runtime(client)
        assert state.extras is not None
        packed_group_states = self._packed_group_states(state.extras)
        next_group_states: dict[str, dict[str, object]] = {}
        payloads: dict[str, dict[str, object]] = {}
        group_quality: list[float] = []
        for spec in self._group_specs:
            method = self._group_methods[spec.name]
            group_client = self._group_client_view(client, spec)
            group_state = _unpack_state(packed_group_states[spec.name])
            payload, next_state = method.client_step(group_client, group_state)
            payloads[spec.name] = payload
            next_group_states[spec.name] = _pack_state(next_state)
            wrong = payload.get("class_wrong_counts")
            counts = None
            if isinstance(next_state.extras, dict):
                counts = next_state.extras.get("train_class_counts")
            if isinstance(wrong, torch.Tensor) and isinstance(counts, torch.Tensor):
                wrong_total = float(wrong.to(self.encoder.device).sum().item())
                count_total = float(counts.to(self.encoder.device).sum().item())
                if count_total > 0.0:
                    group_quality.append(max(0.0, 1.0 - (wrong_total / count_total)))
                    continue
            group_quality.append(0.0)

        previous_weights = self._weights_from_extras(state.extras)
        estimated_weights = self._weights_from_quality(
            torch.tensor(group_quality, device=self.encoder.device, dtype=torch.float32)
        )
        weights = self._updated_client_weights(
            previous_weights=previous_weights,
            estimated_weights=estimated_weights,
        )
        next_client_state = ClientState(
            memory=self._compose_memory(next_group_states, weights),
            extras={
                "eg_group_states": next_group_states,
                "eg_group_weights": weights.detach().clone(),
            },
        )
        return {
            "client_id": client.client_id,
            "eg_group_payloads": payloads,
        }, next_client_state

    def server_step(self, payloads: list[dict[str, Any]]) -> None:
        for spec in self._group_specs:
            method = self._group_methods[spec.name]
            grouped_payloads: list[dict[str, object]] = []
            for payload in payloads:
                group_payloads = payload.get("eg_group_payloads")
                if not isinstance(group_payloads, dict):
                    continue
                group_payload = group_payloads.get(spec.name)
                if isinstance(group_payload, dict):
                    grouped_payloads.append(group_payload)
            method.server_step(grouped_payloads)

    def evaluate(self, clients: list[ClientData], states: list[ClientState]) -> dict[str, float]:
        personalized_accs: list[float] = []
        shared_accs: list[float] = []
        personal_accs: list[float] = []
        common_delta_norms: list[float] = []
        weight_entropies: list[float] = []
        weight_maxima: list[float] = []

        for client, state in zip(clients, states):
            fused_scores, shared_scores, personal_scores, target, mean_common_delta_norm, attention_entropy, attention_max = self._aggregate_client_scores(
                client=client,
                state=state,
                split="test",
            )
            personalized_accs.append(float((fused_scores.argmax(dim=1) == target).float().mean().item()))
            shared_accs.append(float((shared_scores.argmax(dim=1) == target).float().mean().item()))
            personal_accs.append(float((personal_scores.argmax(dim=1) == target).float().mean().item()))
            common_delta_norms.append(float(mean_common_delta_norm))
            weights = self._weights_from_extras(state.extras)
            weight_entropies.append(_normalized_entropy(weights))
            weight_maxima.append(float(weights.max().item()))
        def _mean(values: list[float]) -> float:
            if not values:
                return 0.0
            return float(sum(values) / len(values))

        mean_sync_gate = _mean(
            [
                float(getattr(method, "_last_sync_metrics", {}).get("mean_shared_sync_gate", 0.0))
                for method in self._group_methods.values()
            ]
        )
        mean_sync_before = _mean(
            [
                float(getattr(method, "_last_sync_metrics", {}).get("mean_shared_delta_before", 0.0))
                for method in self._group_methods.values()
            ]
        )
        mean_sync_after = _mean(
            [
                float(getattr(method, "_last_sync_metrics", {}).get("mean_shared_delta_after", 0.0))
                for method in self._group_methods.values()
            ]
        )

        total_intersection = float(sum(spec.intersection_rank for spec in self._group_specs))
        total_shared = float(sum(spec.shared_rank for spec in self._group_specs))
        total_global_only = float(sum(max(spec.shared_rank - spec.intersection_rank, 0) for spec in self._group_specs))

        return {
            "mean_personalized_accuracy": _mean(personalized_accs),
            "mean_local_test_accuracy": _mean(personalized_accs),
            "mean_shared_branch_accuracy": _mean(shared_accs),
            "mean_personal_branch_accuracy": _mean(personal_accs),
            "mean_alpha": 0.0,
            "mean_common_delta_norm": _mean(common_delta_norms),
            "mean_shared_sync_gate": mean_sync_gate,
            "mean_shared_delta_before": mean_sync_before,
            "mean_shared_delta_after": mean_sync_after,
            "global_only_basis_refresh_applied": 0.0,
            "global_only_basis_refresh_mean_drift": 0.0,
            "intersection_rank": total_intersection,
            "intersection_ratio": 0.0 if total_shared <= 0.0 else (total_intersection / total_shared),
            "global_only_rank": total_global_only,
            "wasserstein_sync_applied": 0.0,
            "wasserstein_sync_eligible_rows": 0.0,
            "wasserstein_sync_class_coverage": 0.0,
            "eg_group_count": float(len(self._group_specs)),
            "eg_interaction_count": float(len(self._interaction_pairs)),
            "eg_weight_entropy_mean": _mean(weight_entropies),
            "eg_weight_max_mean": _mean(weight_maxima),
            "eg_hd_dim_total": float(sum(spec.hd_dim for spec in self._group_specs)),
            "eg_shared_rank_total": total_shared,
            "eg_personal_rank_total": float(sum(spec.personal_rank for spec in self._group_specs)),
        }
