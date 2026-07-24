from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ...data import ClientData
from ...federated import ClientState
from ...similarity import similarity_scores
from .subspace_trial_core import (
    BaseSubspaceTrialMethod,
    row_normalize,
    simple_personal_basis,
    top_basis_from_covariance,
)


@dataclass
class SubspaceTrialMethod(BaseSubspaceTrialMethod):
    def _init_runtime_state(self) -> None:
        self._pending_state_updates: dict[str, dict[str, Any]] = {}
        self._last_sync_metrics: dict[str, float] = {
            "mean_shared_sync_gate": 0.0,
            "mean_shared_delta_before": 0.0,
            "mean_shared_delta_after": 0.0,
        }

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
        del client, train_idx, val_idx, train_x_hv, full_memory
        class_counts = torch.bincount(train_y, minlength=self.num_classes).to(torch.float32)
        return {
            "train_class_counts": class_counts.detach().clone(),
        }

    def _shared_basis_from_covariance(self, covariance: torch.Tensor) -> torch.Tensor:
        return top_basis_from_covariance(covariance, self.shared_rank)

    def _decompose_memory(self, full_memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.shared_basis is not None
        personal_basis = simple_personal_basis(full_memory, self.shared_basis, self.personal_rank)
        shared_coords = row_normalize(full_memory @ self.shared_basis)
        personal_coords = row_normalize(full_memory @ personal_basis)
        return personal_basis, shared_coords, personal_coords

    def _scores(
        self,
        x_hv: torch.Tensor,
        *,
        personal_basis: torch.Tensor,
        shared_coords: torch.Tensor,
        personal_coords: torch.Tensor,
        alpha: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.shared_basis is not None
        z_g = row_normalize(x_hv @ self.shared_basis)
        z_p = row_normalize(x_hv @ personal_basis)
        shared_scores = similarity_scores(z_g, shared_coords, metric="cos")
        personal_scores = similarity_scores(z_p, personal_coords, metric="cos")
        fused_scores = (float(alpha) * shared_scores) + ((1.0 - float(alpha)) * personal_scores)
        return fused_scores, shared_scores, personal_scores

    def _materialize_state(self, client: ClientData, state: ClientState, *, consume: bool) -> ClientState:
        extras = {} if state.extras is None else dict(state.extras)
        pending = self._pending_state_updates.get(client.client_id)
        if pending is not None:
            for key, value in pending.items():
                extras[key] = value.detach().clone() if isinstance(value, torch.Tensor) else value

        personal_basis = extras["personal_basis"].to(self.encoder.device)
        shared_coords = extras["shared_coords"].to(self.encoder.device)
        personal_coords = extras["personal_coords"].to(self.encoder.device)
        full_memory = self._reconstruct_memory(shared_coords, personal_coords, personal_basis)
        extras["full_memory"] = full_memory.detach().clone()

        if consume and pending is not None:
            del self._pending_state_updates[client.client_id]
        return ClientState(memory=full_memory.detach().clone(), extras=extras)

    def _next_state_extras(
        self,
        state: ClientState,
        *,
        full_memory: torch.Tensor,
        shared_coords: torch.Tensor,
        personal_coords: torch.Tensor,
        personal_basis: torch.Tensor,
        alpha: float,
    ) -> dict[str, Any]:
        extras = super()._next_state_extras(
            state,
            full_memory=full_memory,
            shared_coords=shared_coords,
            personal_coords=personal_coords,
            personal_basis=personal_basis,
            alpha=alpha,
        )
        assert state.extras is not None
        extras["train_class_counts"] = state.extras["train_class_counts"].detach().clone()
        return extras

    def _aggregate_shared_coords(self, payloads: list[dict[str, Any]]) -> torch.Tensor:
        shared_dim = payloads[0]["shared_coords"].shape[1]
        weighted_sum = torch.zeros(
            self.num_classes,
            shared_dim,
            device=self.encoder.device,
            dtype=torch.float32,
        )
        total_counts = torch.zeros(self.num_classes, 1, device=self.encoder.device, dtype=torch.float32)
        for payload in payloads:
            shared_coords = payload["shared_coords"].to(self.encoder.device)
            class_counts = payload["class_counts"].to(self.encoder.device).unsqueeze(1)
            weighted_sum.add_(shared_coords * class_counts)
            total_counts.add_(class_counts)
        aggregated = weighted_sum / total_counts.clamp_min(1.0)
        aggregated[total_counts.squeeze(1) <= 0] = 0.0
        return row_normalize(aggregated)

    def _server_shared_coord_sync(
        self,
        payload: dict[str, Any],
        global_shared: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del payload
        gate = torch.zeros(self.num_classes, device=self.encoder.device, dtype=torch.float32)
        return global_shared.detach().clone(), gate

    def _record_sync_metrics(
        self,
        gate_means: list[float],
        delta_before: list[float],
        delta_after: list[float],
        follow_gate_values: list[torch.Tensor] | None = None,
    ) -> None:
        def _mean(values: list[float]) -> float:
            if not values:
                return 0.0
            return float(sum(values) / len(values))

        def _histogram(values: list[float], *, bins: int = 10) -> tuple[list[float], list[float]]:
            if bins <= 0:
                return [], []
            if not values:
                edges = [float(i) / float(bins) for i in range(bins + 1)]
                return [0.0 for _ in range(bins)], edges

            counts: list[int] = [0 for _ in range(bins)]
            for value in values:
                bounded = min(max(float(value), 0.0), 1.0)
                idx = min(int(bounded * bins), bins - 1)
                counts[idx] += 1

            total = float(sum(counts)) if sum(counts) > 0 else 1.0
            return [count / total for count in counts], [float(i) / float(bins) for i in range(bins + 1)]

        def _flatten(values: list[torch.Tensor] | None) -> list[float]:
            if not values:
                return []
            flat: list[float] = []
            for item in values:
                if item is None:
                    continue
                if isinstance(item, torch.Tensor):
                    flat.extend(float(v) for v in item.detach().to(torch.float32).reshape(-1).tolist())
                else:
                    flat.extend(float(v) for v in item)
            return flat

        gate_floor_value = 1.0 - min(max(float(getattr(self, "gate_min", 0.0)), 0.0), 1.0)
        gate_values = _flatten(follow_gate_values)
        if not gate_values:
            gate_values = gate_means

        self._last_sync_metrics = {
            "mean_shared_sync_gate": _mean(gate_means),
            "std_shared_sync_gate": float(np.std(np.asarray(gate_means, dtype=np.float64)) if gate_means else 0.0),
            "mean_shared_delta_before": _mean(delta_before),
            "mean_shared_delta_after": _mean(delta_after),
            "mean_shared_sync_gate_floor_fraction": float(
                sum(1.0 for value in gate_values if value >= (gate_floor_value - 1e-12)) / max(len(gate_values), 1)
            ),
            "mean_shared_sync_gate_histogram": _histogram(gate_values)[0],
            "mean_shared_sync_gate_histogram_bin_edges": _histogram(gate_values)[1],
        }

    def _final_class_prediction_stats(
        self,
        y_true: torch.Tensor,
        pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total = torch.bincount(y_true, minlength=self.num_classes).to(torch.float32)
        if int(y_true.numel()) == 0:
            zeros = torch.zeros(self.num_classes, device=self.encoder.device, dtype=torch.float32)
            return zeros, zeros, zeros
        correct_mask = pred == y_true
        if torch.any(correct_mask):
            correct = torch.bincount(y_true[correct_mask], minlength=self.num_classes).to(torch.float32)
        else:
            correct = torch.zeros(self.num_classes, device=self.encoder.device, dtype=torch.float32)
        wrong = total - correct
        return total, correct, wrong

    def client_step(self, client: ClientData, state: ClientState) -> tuple[dict[str, Any], ClientState]:
        state = self._materialize_state(client, state, consume=True)
        assert state.extras is not None
        train_idx = state.extras["train_idx"]
        x_train = client.x_train.index_select(0, train_idx)
        y_train = client.y_train.index_select(0, train_idx).to(self.encoder.device).long()
        x_hv = self.encoder.encode(x_train)

        personal_basis, shared_coords, personal_coords, alpha = self._state_components(state)

        num_samples = int(x_hv.shape[0])
        num_classes = int(shared_coords.shape[0])
        eye = torch.eye(num_classes, dtype=torch.bool, device=x_hv.device)
        for _ in range(int(self.local_epochs)):
            order = torch.randperm(num_samples, device=x_hv.device)
            for start in range(0, num_samples, self.batch_size):
                idx = order[start:start + self.batch_size]
                x_batch = x_hv.index_select(0, idx)
                y_batch = y_train.index_select(0, idx)
                z_g = row_normalize(x_batch @ self.shared_basis)
                z_p = row_normalize(x_batch @ personal_basis)
                fused_scores, _, _ = self._scores(
                    x_batch,
                    personal_basis=personal_basis,
                    shared_coords=shared_coords,
                    personal_coords=personal_coords,
                    alpha=alpha,
                )
                pred = fused_scores.argmax(dim=1)
                wrong = pred != y_batch
                if not torch.any(wrong):
                    continue

                wrong_mask = wrong.repeat(num_classes, 1).T
                correct_update = wrong_mask & eye[y_batch]
                wrong_update = wrong_mask & eye[pred]
                shared_updates = (correct_update.float() - wrong_update.float()).T @ z_g
                personal_updates = (correct_update.float() - wrong_update.float()).T @ z_p

                shared_coords.add_(shared_updates, alpha=float(self.global_lr))
                personal_coords.add_(personal_updates, alpha=float(self.personal_lr))
                shared_coords = row_normalize(shared_coords)
                personal_coords = row_normalize(personal_coords)

        alpha = self._learn_alpha(
            client,
            val_idx=state.extras["val_idx"],
            personal_basis=personal_basis,
            shared_coords=shared_coords,
            personal_coords=personal_coords,
        )
        with torch.no_grad():
            final_pred = self._scores(
                x_hv,
                personal_basis=personal_basis,
                shared_coords=shared_coords,
                personal_coords=personal_coords,
                alpha=alpha,
            )[0].argmax(dim=1)
            class_total_counts, class_correct_counts, class_wrong_counts = self._final_class_prediction_stats(
                y_train,
                final_pred,
            )
        fused_memory = self._reconstruct_memory(shared_coords, personal_coords, personal_basis)
        full_memory = fused_memory.detach().clone()

        if self.debug:
            x_eval = self.encoder.encode(client.x_test)
            pred = self._scores(
                x_eval,
                personal_basis=personal_basis,
                shared_coords=shared_coords,
                personal_coords=personal_coords,
                alpha=alpha,
            )[0].argmax(dim=1)
            acc = float((pred.cpu() == client.y_test.cpu()).float().mean().item())
            print(
                f"[debug][subspace_trial][round={self._round + 1}][client={client.client_id}] "
                f"alpha={alpha:.2f} test_acc={acc:.4f}"
            )

        payload = {
            "client_id": client.client_id,
            "shared_coords": shared_coords.detach().clone(),
            "class_counts": state.extras["train_class_counts"].to(self.encoder.device).detach().clone(),
            "class_total_counts": class_total_counts.detach().clone(),
            "class_correct_counts": class_correct_counts.detach().clone(),
            "class_wrong_counts": class_wrong_counts.detach().clone(),
            "full_memory": full_memory.detach().clone(),
        }
        return payload, ClientState(
            memory=fused_memory.detach().clone(),
            extras=self._next_state_extras(
                state,
                full_memory=full_memory,
                shared_coords=shared_coords,
                personal_coords=personal_coords,
                personal_basis=personal_basis,
                alpha=alpha,
            ),
        )

    def server_step(self, payloads: list[dict[str, Any]]) -> None:
        if not payloads:
            self._record_sync_metrics([], [], [])
            self._round += 1
            return
        global_shared = self._aggregate_shared_coords(payloads)
        gate_means: list[float] = []
        delta_before: list[float] = []
        delta_after: list[float] = []
        pending_updates: dict[str, dict[str, Any]] = {}
        for payload in payloads:
            local_shared = payload["shared_coords"].to(self.encoder.device)
            synced_shared, gate = self._server_shared_coord_sync(payload, global_shared)
            synced_shared = row_normalize(synced_shared)
            pending_updates[str(payload["client_id"])] = {
                "shared_coords": synced_shared.detach().clone(),
            }
            gate_means.append(float(gate.mean().item()))
            delta_before.append(float(torch.linalg.norm(local_shared - global_shared, dim=1).mean().item()))
            delta_after.append(float(torch.linalg.norm(synced_shared - global_shared, dim=1).mean().item()))
        self._pending_state_updates = pending_updates
        self._record_sync_metrics(gate_means, delta_before, delta_after)
        self._round += 1

    def evaluate(self, clients: list[ClientData], states: list[ClientState]) -> dict[str, float]:
        metrics = super().evaluate(clients, states)
        metrics.update(self._last_sync_metrics)
        return metrics
