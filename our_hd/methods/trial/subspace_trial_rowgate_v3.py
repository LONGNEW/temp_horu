from __future__ import annotations

from dataclasses import dataclass

import torch

from ...data import ClientData
from ...federated import ClientState
from ..wasserstein import kmeans_centers_labels_sse, normalize_vec
from .subspace_trial_rowgate_basis_refresh_v3 import SubspaceTrialRowGateBasisRefreshV3Method


@dataclass
class SubspaceTrialRowGateCommonBasisMethod(SubspaceTrialRowGateBasisRefreshV3Method):
    """RowGate with fixed common/global-only/personal-only bases after bootstrap."""

    refresh_interval: int = 0
    enable_wasserstein_sync: bool = False
    wasserstein_atoms: int = 3
    wasserstein_beta: float = 0.0
    wasserstein_max_iters: int = 20
    wasserstein_interval: int = 1

    def _init_runtime_state(self) -> None:
        super()._init_runtime_state()
        self._server_class_counts: dict[str, torch.Tensor] = {}
        self._last_wasserstein_sync_applied: float = 0.0
        self._last_wasserstein_sync_eligible_rows: float = 0.0
        self._last_wasserstein_sync_class_coverage: float = 0.0

    def bootstrap(self, clients: list[ClientData], states: list[ClientState]) -> list[ClientState]:
        states = super().bootstrap(clients, states)
        self._server_class_counts = {}
        for client, state in zip(clients, states):
            assert state.extras is not None
            self._server_class_counts[client.client_id] = state.extras["train_class_counts"].detach().clone()
        return states

    def _server_gate_payload(self, client_id: str, payload: dict[str, object]) -> dict[str, torch.Tensor]:
        class_wrong_counts = payload["class_wrong_counts"]
        assert isinstance(class_wrong_counts, torch.Tensor)
        return {
            "class_total_counts": self._server_class_counts[client_id].to(self.encoder.device),
            "class_wrong_counts": class_wrong_counts.to(self.encoder.device),
        }

    def _local_gate_payload(self, extras: dict[str, object]) -> dict[str, torch.Tensor]:
        class_total_counts = extras["train_class_counts"]
        assert isinstance(class_total_counts, torch.Tensor)
        class_wrong_counts = extras.get("last_class_wrong_counts")
        if not isinstance(class_wrong_counts, torch.Tensor):
            class_wrong_counts = torch.zeros_like(class_total_counts)
        return {
            "class_total_counts": class_total_counts.to(self.encoder.device),
            "class_wrong_counts": class_wrong_counts.to(self.encoder.device),
        }

    def _apply_broadcast_sync(self, extras: dict[str, object], pending: dict[str, object]) -> None:
        global_common = pending.get("broadcast_common_coords")
        global_global_only = pending.get("broadcast_global_only_coords")
        if not isinstance(global_common, torch.Tensor) or not isinstance(global_global_only, torch.Tensor):
            return

        local_common = extras.get("common_coords")
        local_global_only = extras.get("global_only_coords")
        if not isinstance(local_common, torch.Tensor) or not isinstance(local_global_only, torch.Tensor):
            return

        local_upload = torch.cat([local_common.to(self.encoder.device), local_global_only.to(self.encoder.device)], dim=1)
        global_upload = torch.cat([global_common.to(self.encoder.device), global_global_only.to(self.encoder.device)], dim=1)
        _, follow_ratio = self._row_gate_terms(self._local_gate_payload(extras))
        synced_upload = local_upload + (float(self.global_lr) * follow_ratio).unsqueeze(1) * (global_upload - local_upload)
        wasserstein_centers = pending.get("broadcast_wasserstein_centers")
        wasserstein_beta = pending.get("broadcast_wasserstein_beta")
        class_counts = extras.get("train_class_counts")
        if (
            isinstance(wasserstein_centers, torch.Tensor)
            and isinstance(wasserstein_beta, (int, float))
            and float(wasserstein_beta) > 0.0
            and isinstance(class_counts, torch.Tensor)
        ):
            synced_upload = self._apply_wasserstein_pull(
                synced_upload,
                class_counts=class_counts.to(self.encoder.device),
                centers=wasserstein_centers.to(self.encoder.device),
                beta=float(wasserstein_beta),
            )
        common_rank = int(global_common.shape[1])
        extras["common_coords"] = synced_upload[:, :common_rank].detach().clone()
        extras["global_only_coords"] = synced_upload[:, common_rank:].detach().clone()

    def _should_apply_wasserstein_sync(self, round_index: int) -> bool:
        if not bool(self.enable_wasserstein_sync):
            return False
        if float(self.wasserstein_beta) <= 0.0:
            return False
        interval = max(int(self.wasserstein_interval), 1)
        return (int(round_index) % interval) == 0

    def _build_wasserstein_centers(
        self,
        *,
        synced_upload_by_client: dict[str, torch.Tensor],
        class_counts_by_client: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor | None, int, int]:
        if not synced_upload_by_client:
            return None, 0, 0

        sample = next(iter(synced_upload_by_client.values()))
        num_classes = int(sample.shape[0])
        upload_dim = int(sample.shape[1])
        atoms = max(int(self.wasserstein_atoms), 1)
        max_iters = max(int(self.wasserstein_max_iters), 1)
        centers = torch.zeros(num_classes, atoms, upload_dim, device=self.encoder.device, dtype=sample.dtype)

        eligible_rows = 0
        covered_classes = 0
        for class_idx in range(num_classes):
            points: list[torch.Tensor] = []
            for client_id, upload in synced_upload_by_client.items():
                class_counts = class_counts_by_client.get(client_id)
                if class_counts is None or class_idx >= int(class_counts.numel()):
                    continue
                if float(class_counts[class_idx].item()) <= 0.0:
                    continue
                row = upload[class_idx]
                row_norm = torch.linalg.norm(row)
                if float(row_norm.item()) <= 1e-8:
                    continue
                points.append((row / row_norm.clamp_min(1e-8)).to(torch.float32))

            if not points:
                continue
            covered_classes += 1
            eligible_rows += len(points)
            class_points = torch.stack(points, dim=0)
            n_clusters = min(atoms, int(class_points.shape[0]))
            class_centers, _, _ = kmeans_centers_labels_sse(
                class_points,
                n_clusters=n_clusters,
                max_iters=max_iters,
            )
            if int(class_centers.shape[0]) < atoms:
                pad = class_centers[-1:].repeat(atoms - int(class_centers.shape[0]), 1)
                class_centers = torch.cat([class_centers, pad], dim=0)
            centers[class_idx] = class_centers[:atoms].to(device=self.encoder.device, dtype=sample.dtype)

        return centers, eligible_rows, covered_classes

    def _apply_wasserstein_pull(
        self,
        upload: torch.Tensor,
        *,
        class_counts: torch.Tensor,
        centers: torch.Tensor,
        beta: float,
    ) -> torch.Tensor:
        if float(beta) <= 0.0:
            return upload
        if int(upload.shape[0]) == 0 or int(upload.shape[1]) == 0:
            return upload
        if centers.ndim != 3:
            return upload

        num_classes = min(int(upload.shape[0]), int(centers.shape[0]), int(class_counts.numel()))
        for class_idx in range(num_classes):
            if float(class_counts[class_idx].item()) <= 0.0:
                continue
            row = upload[class_idx]
            row_norm = torch.linalg.norm(row)
            if float(row_norm.item()) <= 1e-8:
                continue
            row_unit = row / row_norm.clamp_min(1e-8)
            class_centers = centers[class_idx]
            center_norms = torch.linalg.norm(class_centers, dim=1)
            valid = center_norms > 1e-8
            if not bool(valid.any()):
                continue
            valid_centers = class_centers[valid]
            distances = torch.cdist(
                row_unit.unsqueeze(0).to(torch.float32),
                valid_centers.to(torch.float32),
                p=2.0,
            ).squeeze(0)
            nearest = valid_centers[int(torch.argmin(distances).item())].to(row.dtype)
            pulled = normalize_vec(((1.0 - float(beta)) * row_unit) + (float(beta) * nearest))
            upload[class_idx] = pulled * row_norm
        return upload

    def _materialize_state(self, client: ClientData, state: ClientState, *, consume: bool) -> ClientState:
        extras = {} if state.extras is None else dict(state.extras)
        pending = self._pending_state_updates.get(client.client_id)
        if pending is not None:
            self._apply_broadcast_sync(extras, pending)
            for key, value in pending.items():
                if key in {
                    "broadcast_common_coords",
                    "broadcast_global_only_coords",
                    "broadcast_wasserstein_centers",
                    "broadcast_wasserstein_beta",
                }:
                    continue
                extras[key] = value.detach().clone() if isinstance(value, torch.Tensor) else value

        full_memory = self._full_memory(*self._state_components(ClientState(extras=extras)))
        extras["full_memory"] = full_memory.detach().clone()

        if consume and pending is not None:
            del self._pending_state_updates[client.client_id]
        return ClientState(memory=full_memory.detach().clone(), extras=extras)

    def client_step(self, client: ClientData, state: ClientState) -> tuple[dict[str, object], ClientState]:
        payload, next_state = super().client_step(client, state)
        assert next_state.extras is not None
        next_extras = dict(next_state.extras)
        class_wrong_counts = payload["class_wrong_counts"]
        assert isinstance(class_wrong_counts, torch.Tensor)
        next_extras["last_class_wrong_counts"] = class_wrong_counts.detach().clone()
        return {
            "client_id": payload["client_id"],
            "common_coords": payload["common_coords"],
            "global_only_coords": payload["global_only_coords"],
            "class_wrong_counts": class_wrong_counts.detach().clone(),
        }, ClientState(
            memory=next_state.memory.detach().clone() if isinstance(next_state.memory, torch.Tensor) else next_state.memory,
            extras=next_extras,
        )

    def server_step(self, payloads: list[dict[str, object]]) -> None:
        self._last_refresh_applied = 0.0
        self._last_refresh_mean_drift = 0.0
        self._last_wasserstein_sync_applied = 0.0
        self._last_wasserstein_sync_eligible_rows = 0.0
        self._last_wasserstein_sync_class_coverage = 0.0
        if not payloads:
            self._record_sync_metrics([], [], [])
            self._round += 1
            return

        global_common = self._aggregate_weighted_coords(
            {
                str(payload["client_id"]): payload["common_coords"].to(self.encoder.device)  # type: ignore[union-attr]
                for payload in payloads
            },
            self._server_class_counts,
        )
        global_global_only = self._aggregate_weighted_coords(
            {
                str(payload["client_id"]): payload["global_only_coords"].to(self.encoder.device)  # type: ignore[union-attr]
                for payload in payloads
            },
            self._server_class_counts,
        )

        global_upload = torch.cat([global_common, global_global_only], dim=1)

        pending_updates: dict[str, dict[str, object]] = {}
        synced_upload_by_client: dict[str, torch.Tensor] = {}
        gate_means: list[float] = []
        delta_before: list[float] = []
        delta_after: list[float] = []

        for payload in payloads:
            client_id = str(payload["client_id"])
            local_common = payload["common_coords"]
            local_global_only = payload["global_only_coords"]
            assert isinstance(local_common, torch.Tensor)
            assert isinstance(local_global_only, torch.Tensor)
            local_upload = torch.cat(
                [local_common.to(self.encoder.device), local_global_only.to(self.encoder.device)],
                dim=1,
            )
            _, follow_ratio = self._row_gate_terms(self._server_gate_payload(client_id, payload))
            pending_updates[client_id] = {
                "broadcast_common_coords": global_common.detach().clone(),
                "broadcast_global_only_coords": global_global_only.detach().clone(),
            }
            synced_upload = local_upload + (float(self.global_lr) * follow_ratio).unsqueeze(1) * (global_upload - local_upload)
            synced_upload_by_client[client_id] = synced_upload.detach().clone()
            gate_means.append(float(follow_ratio.mean().item()))
            delta_before.append(float(torch.linalg.norm(local_upload - global_upload, dim=1).mean().item()))
            delta_after.append(float(torch.linalg.norm(synced_upload - global_upload, dim=1).mean().item()))

        round_index = int(self._round) + 1
        if self._should_apply_wasserstein_sync(round_index):
            class_counts_by_client = {
                str(payload["client_id"]): self._server_class_counts[str(payload["client_id"])].to(self.encoder.device)
                for payload in payloads
            }
            centers, eligible_rows, covered_classes = self._build_wasserstein_centers(
                synced_upload_by_client=synced_upload_by_client,
                class_counts_by_client=class_counts_by_client,
            )
            if centers is not None:
                self._last_wasserstein_sync_applied = 1.0
                self._last_wasserstein_sync_eligible_rows = float(eligible_rows)
                self._last_wasserstein_sync_class_coverage = float(covered_classes)
                centers_payload = centers.detach().clone()
                beta_payload = float(self.wasserstein_beta)
                for client_id in pending_updates:
                    pending_updates[client_id]["broadcast_wasserstein_centers"] = centers_payload
                    pending_updates[client_id]["broadcast_wasserstein_beta"] = beta_payload

        self._pending_state_updates = pending_updates
        self._record_sync_metrics(gate_means, delta_before, delta_after)
        self._round += 1


SubspaceTrialRowGateV3Method = SubspaceTrialRowGateCommonBasisMethod
