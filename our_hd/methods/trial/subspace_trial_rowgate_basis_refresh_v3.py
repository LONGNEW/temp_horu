from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ...data import ClientData
from ...federated import ClientState
from ...similarity import similarity_scores
from .subspace_trial_core import row_normalize, top_basis_from_covariance
from .subspace_trial_rowgate import SubspaceTrialRowGateMethod


@dataclass
class SubspaceTrialRowGateBasisRefreshV3Method(SubspaceTrialRowGateMethod):
    """V3 overlap variant with common, global-only, and personal-only branches."""

    refresh_interval: int = 5
    intersection_rank: int = 8
    intersection_ratio: float | None = None

    def _init_runtime_state(self) -> None:
        super()._init_runtime_state()
        self.common_basis: torch.Tensor | None = None
        self.global_only_basis: torch.Tensor | None = None
        self._client_personal_cache: dict[str, dict[str, torch.Tensor]] = {}
        self._last_refresh_applied: float = 0.0
        self._last_refresh_mean_drift: float = 0.0
        self._last_wasserstein_sync_applied: float = 0.0
        self._last_wasserstein_sync_eligible_rows: float = 0.0
        self._last_wasserstein_sync_class_coverage: float = 0.0

    def _resolved_intersection_rank(self) -> int:
        if self.intersection_ratio is not None:
            ratio = min(max(float(self.intersection_ratio), 0.0), 1.0)
            return max(0, min(int(round(int(self.shared_rank) * ratio)), int(self.shared_rank)))
        return max(0, min(int(self.intersection_rank), int(self.shared_rank)))

    def _global_only_rank(self) -> int:
        return max(int(self.shared_rank) - self._resolved_intersection_rank(), 0)

    def _should_refresh(self) -> bool:
        interval = int(self.refresh_interval)
        if interval <= 0:
            return False
        current_round = int(self._round) + 1
        return (current_round % interval) == 0

    def _build_init_extras(
        self,
        client: ClientData,
        *,
        train_idx: torch.Tensor,
        val_idx: torch.Tensor,
        train_x_hv: torch.Tensor,
        train_y: torch.Tensor,
        full_memory: torch.Tensor,
    ) -> dict[str, Any]:
        extras = super()._build_init_extras(
            client,
            train_idx=train_idx,
            val_idx=val_idx,
            train_x_hv=train_x_hv,
            train_y=train_y,
            full_memory=full_memory,
        )
        extras.update(
            {
                "cached_train_hv": train_x_hv.detach().cpu().clone(),
                "cached_train_y": train_y.detach().cpu().clone(),
            }
        )
        return extras

    def _basis_from_covariance(
        self,
        covariance: torch.Tensor,
        rank: int,
        *,
        reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if int(rank) <= 0:
            return torch.zeros(
                covariance.shape[0],
                0,
                device=covariance.device,
                dtype=covariance.dtype,
            )
        return top_basis_from_covariance(
            covariance,
            rank,
            reference=reference,
            complete_degenerate_basis=True,
            oversample_factor=2,
        )

    def _set_basis_split(self, basis: torch.Tensor) -> None:
        common_rank = min(self._resolved_intersection_rank(), int(basis.shape[1]))
        self.common_basis = basis[:, :common_rank].detach().clone()
        self.global_only_basis = basis[:, common_rank:].detach().clone()
        self.shared_basis = torch.cat([self.common_basis, self.global_only_basis], dim=1)

    def _aggregate_weighted_coords(
        self,
        coords_by_client: dict[str, torch.Tensor],
        counts_by_client: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        sample = next(iter(coords_by_client.values()))
        weighted_sum = torch.zeros(
            self.num_classes,
            sample.shape[1],
            device=self.encoder.device,
            dtype=torch.float32,
        )
        total_counts = torch.zeros(self.num_classes, 1, device=self.encoder.device, dtype=torch.float32)
        for client_id, coords in coords_by_client.items():
            class_counts = counts_by_client[client_id].to(self.encoder.device).unsqueeze(1)
            weighted_sum.add_(coords.to(self.encoder.device) * class_counts)
            total_counts.add_(class_counts)
        aggregated = weighted_sum / total_counts.clamp_min(1.0)
        aggregated[total_counts.squeeze(1) <= 0] = 0.0
        return aggregated

    def _server_memory(
        self,
        common_coords: torch.Tensor,
        global_only_coords: torch.Tensor,
    ) -> torch.Tensor:
        assert self.common_basis is not None
        assert self.global_only_basis is not None
        memory = common_coords @ self.common_basis.T
        if int(self.global_only_basis.shape[1]) > 0:
            memory = memory + (global_only_coords @ self.global_only_basis.T)
        return memory

    def _personal_memory(
        self,
        common_delta_coords: torch.Tensor,
        personal_coords: torch.Tensor,
        personal_basis: torch.Tensor,
    ) -> torch.Tensor:
        assert self.common_basis is not None
        memory = common_delta_coords @ self.common_basis.T
        if int(personal_basis.shape[1]) > 0:
            memory = memory + (personal_coords @ personal_basis.T)
        return memory

    def _full_memory(
        self,
        common_coords: torch.Tensor,
        common_delta_coords: torch.Tensor,
        global_only_coords: torch.Tensor,
        personal_coords: torch.Tensor,
        personal_basis: torch.Tensor,
    ) -> torch.Tensor:
        return row_normalize(
            self._server_memory(common_coords, global_only_coords)
            + self._personal_memory(common_delta_coords, personal_coords, personal_basis)
        )

    def _personal_basis_from_memory(self, personal_memory: torch.Tensor) -> torch.Tensor:
        assert self.shared_basis is not None
        covariance = personal_memory.T @ personal_memory
        return self._basis_from_covariance(
            covariance,
            int(self.personal_rank),
            reference=self.shared_basis,
        )

    def _predict_scores(
        self,
        x_hv: torch.Tensor,
        *,
        common_coords: torch.Tensor,
        common_delta_coords: torch.Tensor,
        global_only_coords: torch.Tensor,
        personal_coords: torch.Tensor,
        personal_basis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shared_memory = self._server_memory(common_coords, global_only_coords)
        personal_memory = self._personal_memory(common_delta_coords, personal_coords, personal_basis)
        full_memory = row_normalize(shared_memory + personal_memory)
        return (
            similarity_scores(x_hv, full_memory, metric="cos"),
            similarity_scores(x_hv, shared_memory, metric="cos"),
            similarity_scores(x_hv, personal_memory, metric="cos"),
            full_memory,
        )

    def _state_components(
        self,
        state: ClientState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert state.extras is not None
        return (
            state.extras["common_coords"].to(self.encoder.device),
            state.extras["common_delta_coords"].to(self.encoder.device),
            state.extras["global_only_coords"].to(self.encoder.device),
            state.extras["personal_coords"].to(self.encoder.device),
            state.extras["personal_basis"].to(self.encoder.device),
        )

    def _set_predict_context(self, extras: dict[str, Any] | None) -> None:
        del extras

    def _clear_predict_context(self) -> None:
        return None

    def _predict_scores_with_context(
        self,
        x_hv: torch.Tensor,
        *,
        extras: dict[str, Any] | None,
        common_coords: torch.Tensor,
        common_delta_coords: torch.Tensor,
        global_only_coords: torch.Tensor,
        personal_coords: torch.Tensor,
        personal_basis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._set_predict_context(extras)
        try:
            return self._predict_scores(
                x_hv,
                common_coords=common_coords,
                common_delta_coords=common_delta_coords,
                global_only_coords=global_only_coords,
                personal_coords=personal_coords,
                personal_basis=personal_basis,
            )
        finally:
            self._clear_predict_context()

    def _next_state_extras(
        self,
        state: ClientState,
        *,
        full_memory: torch.Tensor,
        common_coords: torch.Tensor,
        common_delta_coords: torch.Tensor,
        global_only_coords: torch.Tensor,
        personal_coords: torch.Tensor,
        personal_basis: torch.Tensor,
    ) -> dict[str, Any]:
        assert state.extras is not None
        extras = {
            "train_idx": state.extras["train_idx"].detach().clone(),
            "val_idx": state.extras["val_idx"].detach().clone(),
            "train_class_counts": state.extras["train_class_counts"].detach().clone(),
            "full_memory": full_memory.detach().clone(),
            "common_coords": common_coords.detach().clone(),
            "common_delta_coords": common_delta_coords.detach().clone(),
            "global_only_coords": global_only_coords.detach().clone(),
            "personal_coords": personal_coords.detach().clone(),
            "personal_basis": personal_basis.detach().clone(),
            "alpha": 0.0,
        }
        if "cached_train_hv" in state.extras:
            extras["cached_train_hv"] = state.extras["cached_train_hv"].detach().clone()
        if "cached_train_y" in state.extras:
            extras["cached_train_y"] = state.extras["cached_train_y"].detach().clone()
        return extras

    def _cached_local_train_tensors(self, client: ClientData, state: ClientState) -> tuple[torch.Tensor, torch.Tensor]:
        assert state.extras is not None
        cached_train_hv = state.extras.get("cached_train_hv")
        cached_train_y = state.extras.get("cached_train_y")
        if isinstance(cached_train_hv, torch.Tensor) and isinstance(cached_train_y, torch.Tensor):
            return (
                cached_train_hv.to(self.encoder.device),
                cached_train_y.to(self.encoder.device).long(),
            )

        train_idx = state.extras["train_idx"]
        x_train = client.x_train.index_select(0, train_idx)
        y_train = client.y_train.index_select(0, train_idx).to(self.encoder.device).long()
        return self.encoder.encode(x_train), y_train

    def _materialize_state(self, client: ClientData, state: ClientState, *, consume: bool) -> ClientState:
        extras = {} if state.extras is None else dict(state.extras)
        pending = self._pending_state_updates.get(client.client_id)
        if pending is not None:
            for key, value in pending.items():
                extras[key] = value.detach().clone() if isinstance(value, torch.Tensor) else value

        full_memory = self._full_memory(*self._state_components(ClientState(extras=extras)))
        extras["full_memory"] = full_memory.detach().clone()

        if consume and pending is not None:
            del self._pending_state_updates[client.client_id]
        return ClientState(memory=full_memory.detach().clone(), extras=extras)

    def bootstrap(self, clients: list[ClientData], states: list[ClientState]) -> list[ClientState]:
        memories = [state.memory.to(self.encoder.device) for state in states]
        shared_basis = self._shared_basis_from_states(states)
        self._set_basis_split(shared_basis)
        self._pending_state_updates = {}
        self._client_personal_cache = {}

        counts_by_client: dict[str, torch.Tensor] = {}
        common_totals_by_client: dict[str, torch.Tensor] = {}
        global_totals_by_client: dict[str, torch.Tensor] = {}

        assert self.common_basis is not None
        assert self.global_only_basis is not None
        for client, state, memory in zip(clients, states, memories):
            assert state.extras is not None
            counts_by_client[client.client_id] = state.extras["train_class_counts"].detach().clone()
            common_totals_by_client[client.client_id] = memory @ self.common_basis
            global_totals_by_client[client.client_id] = memory @ self.global_only_basis

        common_consensus = self._aggregate_weighted_coords(common_totals_by_client, counts_by_client)
        global_consensus = self._aggregate_weighted_coords(global_totals_by_client, counts_by_client)

        bootstrapped: list[ClientState] = []
        for client, state, memory in zip(clients, states, memories):
            client_id = client.client_id
            common_total = common_totals_by_client[client_id]
            common_delta = common_total - common_consensus
            residual = memory - self._server_memory(common_total, global_consensus)
            personal_basis = self._personal_basis_from_memory(residual)
            personal_coords = residual @ personal_basis
            full_memory = self._full_memory(
                common_consensus,
                common_delta,
                global_consensus,
                personal_coords,
                personal_basis,
            )
            extras = {} if state.extras is None else dict(state.extras)
            extras.update(
                {
                    "full_memory": full_memory.detach().clone(),
                    "common_coords": common_consensus.detach().clone(),
                    "common_delta_coords": common_delta.detach().clone(),
                    "global_only_coords": global_consensus.detach().clone(),
                    "personal_coords": personal_coords.detach().clone(),
                    "personal_basis": personal_basis.detach().clone(),
                    "alpha": 0.0,
                }
            )
            self._client_personal_cache[client_id] = {
                "common_delta_coords": common_delta.detach().clone(),
                "personal_coords": personal_coords.detach().clone(),
                "personal_basis": personal_basis.detach().clone(),
            }
            bootstrapped.append(ClientState(memory=full_memory.detach().clone(), extras=extras))
        return bootstrapped

    def client_step(self, client: ClientData, state: ClientState) -> tuple[dict[str, Any], ClientState]:
        state = self._materialize_state(client, state, consume=True)
        assert state.extras is not None
        assert self.common_basis is not None
        assert self.global_only_basis is not None

        x_hv, y_train = self._cached_local_train_tensors(client, state)

        common_coords, common_delta_coords, global_only_coords, personal_coords, personal_basis = self._state_components(
            state
        )
        num_samples = int(x_hv.shape[0])
        num_classes = int(common_coords.shape[0])
        eye = torch.eye(num_classes, dtype=torch.bool, device=x_hv.device)

        for _ in range(int(self.local_epochs)):
            order = torch.randperm(num_samples, device=x_hv.device)
            for start in range(0, num_samples, self.batch_size):
                idx = order[start:start + self.batch_size]
                x_batch = x_hv.index_select(0, idx)
                y_batch = y_train.index_select(0, idx)
                pred = self._predict_scores_with_context(
                    x_batch,
                    extras=state.extras,
                    common_coords=common_coords,
                    common_delta_coords=common_delta_coords,
                    global_only_coords=global_only_coords,
                    personal_coords=personal_coords,
                    personal_basis=personal_basis,
                )[0].argmax(dim=1)
                wrong = pred != y_batch
                if not torch.any(wrong):
                    continue

                wrong_mask = wrong.repeat(num_classes, 1).T
                correct_update = wrong_mask & eye[y_batch]
                wrong_update = wrong_mask & eye[pred]
                update_mask = (correct_update.float() - wrong_update.float()).T

                common_update = update_mask @ (x_batch @ self.common_basis)
                global_update = update_mask @ (x_batch @ self.global_only_basis)
                personal_update = update_mask @ (x_batch @ personal_basis)

                common_coords.add_(common_update, alpha=float(self.global_lr))
                global_only_coords.add_(global_update, alpha=float(self.global_lr))
                common_delta_coords.add_(common_update, alpha=float(self.personal_lr))
                personal_coords.add_(personal_update, alpha=float(self.personal_lr))

        final_scores, _, _, full_memory = self._predict_scores_with_context(
            x_hv,
            extras=state.extras,
            common_coords=common_coords,
            common_delta_coords=common_delta_coords,
            global_only_coords=global_only_coords,
            personal_coords=personal_coords,
            personal_basis=personal_basis,
        )
        final_pred = final_scores.argmax(dim=1)
        class_total_counts, class_correct_counts, class_wrong_counts = self._final_class_prediction_stats(
            y_train,
            final_pred,
        )

        if self.debug:
            x_eval = self.encoder.encode(client.x_test)
            pred = self._predict_scores_with_context(
                x_eval,
                extras=state.extras,
                common_coords=common_coords,
                common_delta_coords=common_delta_coords,
                global_only_coords=global_only_coords,
                personal_coords=personal_coords,
                personal_basis=personal_basis,
            )[0].argmax(dim=1)
            acc = float((pred.cpu() == client.y_test.cpu()).float().mean().item())
            print(
                f"[debug][subspace_trial_v3][round={self._round + 1}][client={client.client_id}] "
                f"test_acc={acc:.4f}"
            )

        self._client_personal_cache[client.client_id] = {
            "common_delta_coords": common_delta_coords.detach().clone(),
            "personal_coords": personal_coords.detach().clone(),
            "personal_basis": personal_basis.detach().clone(),
        }

        payload = {
            "client_id": client.client_id,
            "common_coords": common_coords.detach().clone(),
            "global_only_coords": global_only_coords.detach().clone(),
            "class_counts": state.extras["train_class_counts"].to(self.encoder.device).detach().clone(),
            "class_total_counts": class_total_counts.detach().clone(),
            "class_correct_counts": class_correct_counts.detach().clone(),
            "class_wrong_counts": class_wrong_counts.detach().clone(),
        }
        return payload, ClientState(
            memory=full_memory.detach().clone(),
            extras=self._next_state_extras(
                state,
                full_memory=full_memory,
                common_coords=common_coords,
                common_delta_coords=common_delta_coords,
                global_only_coords=global_only_coords,
                personal_coords=personal_coords,
                personal_basis=personal_basis,
            ),
        )

    def _server_upload_sync(
        self,
        payload: dict[str, Any],
        local_upload: torch.Tensor,
        global_upload: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, follow_ratio = self._row_gate_terms(payload)
        updated = local_upload + (float(self.global_lr) * follow_ratio).unsqueeze(1) * (global_upload - local_upload)
        return updated, follow_ratio.to(torch.float32)

    def server_step(self, payloads: list[dict[str, Any]]) -> None:
        self._last_refresh_applied = 0.0
        self._last_refresh_mean_drift = 0.0
        if not payloads:
            self._record_sync_metrics([], [], [])
            self._round += 1
            return

        assert self.common_basis is not None
        assert self.global_only_basis is not None

        global_common = self._aggregate_weighted_coords(
            {str(payload["client_id"]): payload["common_coords"] for payload in payloads},
            {str(payload["client_id"]): payload["class_counts"] for payload in payloads},
        )
        global_global_only = self._aggregate_weighted_coords(
            {str(payload["client_id"]): payload["global_only_coords"] for payload in payloads},
            {str(payload["client_id"]): payload["class_counts"] for payload in payloads},
        )

        common_rank = int(global_common.shape[1])
        global_upload = torch.cat([global_common, global_global_only], dim=1)
        old_global_only_basis = self.global_only_basis.to(self.encoder.device)

        synced_common_by_client: dict[str, torch.Tensor] = {}
        synced_global_by_client: dict[str, torch.Tensor] = {}
        global_memory_by_client: dict[str, torch.Tensor] = {}
        gate_means: list[float] = []
        delta_before: list[float] = []
        delta_after: list[float] = []

        for payload in payloads:
            client_id = str(payload["client_id"])
            local_common = payload["common_coords"].to(self.encoder.device)
            local_global_only = payload["global_only_coords"].to(self.encoder.device)
            local_upload = torch.cat([local_common, local_global_only], dim=1)
            synced_upload, gate = self._server_upload_sync(payload, local_upload, global_upload)
            synced_common_by_client[client_id] = synced_upload[:, :common_rank].detach().clone()
            synced_global_by_client[client_id] = synced_upload[:, common_rank:].detach().clone()
            global_memory_by_client[client_id] = synced_global_by_client[client_id] @ old_global_only_basis.T

            gate_means.append(float(gate.mean().item()))
            delta_before.append(float(torch.linalg.norm(local_upload - global_upload, dim=1).mean().item()))
            delta_after.append(float(torch.linalg.norm(synced_upload - global_upload, dim=1).mean().item()))

        pending_updates: dict[str, dict[str, Any]] = {}
        if self._should_refresh() and int(old_global_only_basis.shape[1]) > 0:
            covariance = torch.zeros(
                old_global_only_basis.shape[0],
                old_global_only_basis.shape[0],
                device=self.encoder.device,
                dtype=torch.float32,
            )
            for global_memory in global_memory_by_client.values():
                covariance.add_(global_memory.T @ global_memory)

            self.global_only_basis = self._basis_from_covariance(
                covariance,
                int(old_global_only_basis.shape[1]),
                reference=self.common_basis,
            ).detach().clone()
            self.shared_basis = torch.cat([self.common_basis, self.global_only_basis], dim=1)
            self._last_refresh_applied = 1.0

            drifts: list[float] = []
            for client_id, global_memory in global_memory_by_client.items():
                refreshed_global_only = global_memory @ self.global_only_basis
                reconstructed = refreshed_global_only @ self.global_only_basis.T
                drifts.append(float(torch.linalg.norm(global_memory - reconstructed, dim=1).mean().item()))

                update_payload: dict[str, Any] = {
                    "common_coords": synced_common_by_client[client_id].detach().clone(),
                    "global_only_coords": refreshed_global_only.detach().clone(),
                }
                personal_cached = self._client_personal_cache.get(client_id)
                if personal_cached is not None:
                    personal_memory = (
                        personal_cached["personal_coords"].to(self.encoder.device)
                        @ personal_cached["personal_basis"].to(self.encoder.device).T
                    )
                    refreshed_personal_basis = self._personal_basis_from_memory(personal_memory)
                    refreshed_personal_coords = personal_memory @ refreshed_personal_basis
                    update_payload["personal_basis"] = refreshed_personal_basis.detach().clone()
                    update_payload["personal_coords"] = refreshed_personal_coords.detach().clone()

                pending_updates[client_id] = update_payload

            self._last_refresh_mean_drift = float(sum(drifts) / len(drifts)) if drifts else 0.0
        else:
            for client_id in synced_common_by_client:
                pending_updates[client_id] = {
                    "common_coords": synced_common_by_client[client_id].detach().clone(),
                    "global_only_coords": synced_global_by_client[client_id].detach().clone(),
                }

        self._pending_state_updates = pending_updates
        self._record_sync_metrics(gate_means, delta_before, delta_after)
        self._round += 1

    def evaluate(self, clients: list[ClientData], states: list[ClientState]) -> dict[str, float]:
        personalized_accs: list[float] = []
        shared_branch_accs: list[float] = []
        personal_branch_accs: list[float] = []
        common_delta_norms: list[float] = []

        for client, state in zip(clients, states):
            effective_state = self._materialize_state(client, state, consume=False)
            common_coords, common_delta_coords, global_only_coords, personal_coords, personal_basis = (
                self._state_components(effective_state)
            )
            x_test_hv = self.encoder.encode(client.x_test)
            fused_scores, shared_scores, personal_scores, _ = self._predict_scores_with_context(
                x_test_hv,
                extras=effective_state.extras,
                common_coords=common_coords,
                common_delta_coords=common_delta_coords,
                global_only_coords=global_only_coords,
                personal_coords=personal_coords,
                personal_basis=personal_basis,
            )
            y_test = client.y_test.cpu()
            personalized_accs.append(float((fused_scores.argmax(dim=1).cpu() == y_test).float().mean().item()))
            shared_branch_accs.append(float((shared_scores.argmax(dim=1).cpu() == y_test).float().mean().item()))
            personal_branch_accs.append(float((personal_scores.argmax(dim=1).cpu() == y_test).float().mean().item()))
            common_delta_norms.append(float(torch.linalg.norm(common_delta_coords, dim=1).mean().item()))

        mean_personalized_accuracy = sum(personalized_accs) / max(len(personalized_accs), 1)
        return {
            "mean_personalized_accuracy": mean_personalized_accuracy,
            "mean_local_test_accuracy": mean_personalized_accuracy,
            "mean_shared_branch_accuracy": sum(shared_branch_accs) / max(len(shared_branch_accs), 1),
            "mean_personal_branch_accuracy": sum(personal_branch_accs) / max(len(personal_branch_accs), 1),
            "mean_alpha": 0.0,
            "mean_common_delta_norm": sum(common_delta_norms) / max(len(common_delta_norms), 1),
            "mean_shared_sync_gate": float(self._last_sync_metrics["mean_shared_sync_gate"]),
            "mean_shared_delta_before": float(self._last_sync_metrics["mean_shared_delta_before"]),
            "mean_shared_delta_after": float(self._last_sync_metrics["mean_shared_delta_after"]),
            "global_only_basis_refresh_applied": float(self._last_refresh_applied),
            "global_only_basis_refresh_mean_drift": float(self._last_refresh_mean_drift),
            "intersection_rank": float(self._resolved_intersection_rank()),
            "intersection_ratio": float(self._resolved_intersection_rank() / max(int(self.shared_rank), 1)),
            "global_only_rank": float(self._global_only_rank()),
            "wasserstein_sync_applied": float(getattr(self, "_last_wasserstein_sync_applied", 0.0)),
            "wasserstein_sync_eligible_rows": float(getattr(self, "_last_wasserstein_sync_eligible_rows", 0.0)),
            "wasserstein_sync_class_coverage": float(getattr(self, "_last_wasserstein_sync_class_coverage", 0.0)),
        }
