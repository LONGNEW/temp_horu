from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from ...data import ClientData
from ...federated import ClientState
from ...similarity import similarity_scores
from .subspace_trial_rowgate_v3 import SubspaceTrialRowGateCommonBasisMethod


@dataclass
class HoRUCoreMethod(SubspaceTrialRowGateCommonBasisMethod):
    """Canonical HoRU core (common-basis + bootstrap common-delta zero)."""

    def collect_round_artifacts(
        self,
        clients: list[ClientData],
        states: list[ClientState],
        payloads: list[dict[str, object]],
        *,
        round_index: int,
        selected_indices: list[int],
        server_step_ms: float,
        round_runtime_sec: float,
    ) -> dict[str, object] | None:
        artifacts: dict[str, object] = {}
        if self.enable_subspace_diagnostics:
            common_shares: list[float] = []
            global_shares: list[float] = []
            personal_shares: list[float] = []
            personal_bases: list[torch.Tensor] = []
            common_personal_angles: list[float] = []
            coordinate_entropies: list[float] = []
            effective_ranks: list[float] = []
            assert self.common_basis is not None
            for client, state in zip(clients, states):
                effective = self._materialize_state(client, state, consume=False)
                common, delta, global_only, personal, personal_basis = self._state_components(effective)
                assert self.global_only_basis is not None
                branches = [
                    common @ self.common_basis.T,
                    global_only @ self.global_only_basis.T,
                    (delta @ self.common_basis.T) + (personal @ personal_basis.T),
                ]
                norms = torch.stack([torch.linalg.norm(branch) for branch in branches])
                shares = norms / norms.sum().clamp_min(1e-12)
                common_shares.append(float(shares[0].item()))
                global_shares.append(float(shares[1].item()))
                personal_shares.append(float(shares[2].item()))
                personal_bases.append(personal_basis)
                if personal_basis.numel() > 0 and self.common_basis.numel() > 0:
                    singular = torch.linalg.svdvals(self.common_basis.T @ personal_basis).clamp(-1.0, 1.0)
                    common_personal_angles.append(float(torch.rad2deg(torch.acos(singular)).mean().item()))
                coords = torch.cat([common.abs().flatten(), delta.abs().flatten(), global_only.abs().flatten(), personal.abs().flatten()])
                probs = coords / coords.sum().clamp_min(1e-12)
                coordinate_entropies.append(float((-(probs * probs.clamp_min(1e-12).log()).sum()).item()))
                full = self._full_memory(common, delta, global_only, personal, personal_basis)
                spectral = torch.linalg.svdvals(full)
                effective_ranks.append(float((spectral.square().sum() / spectral.max().square().clamp_min(1e-12)).item()))
            overlaps: list[float] = []
            for index, left in enumerate(personal_bases):
                for right in personal_bases[index + 1:]:
                    if left.numel() and right.numel():
                        singular = torch.linalg.svdvals(left.T @ right).clamp(-1.0, 1.0)
                        overlaps.append(float((90.0 - torch.rad2deg(torch.acos(singular)).mean()).item()))
            artifacts["geometry_metrics"] = {
                "mean_common_energy_share": float(sum(common_shares) / len(common_shares)),
                "mean_global_only_energy_share": float(sum(global_shares) / len(global_shares)),
                "mean_personal_energy_share": float(sum(personal_shares) / len(personal_shares)),
                "mean_client_personal_overlap_deg": float(sum(overlaps) / len(overlaps)) if overlaps else float("nan"),
                "mean_common_personal_angle_deg": float(sum(common_personal_angles) / len(common_personal_angles)) if common_personal_angles else float("nan"),
                "mean_coordinate_usage_entropy": float(sum(coordinate_entropies) / len(coordinate_entropies)),
                "mean_full_effective_rank": float(sum(effective_ranks) / len(effective_ranks)),
                "branch_collapse_indicator": 0.0,
            }
        if self.enable_system_profiling and int(round_index) in set(self.trace_rounds):
            per_client_server_ms = float(server_step_ms) / max(len(selected_indices), 1)
            rows: list[dict[str, object]] = []
            for payload in payloads:
                payload_bytes = sum(
                    value.numel() * value.element_size() for value in payload.values() if isinstance(value, torch.Tensor)
                )
                rows.append({
                    "round": int(round_index), "client_id": str(payload["client_id"]),
                    "client_step_ms": float(payload.get("_client_step_ms", 0.0)),
                    "payload_bytes": float(payload_bytes), "server_step_ms": per_client_server_ms,
                    "round_runtime_sec": float(round_runtime_sec), "selected_client_count": len(selected_indices),
                })
            artifacts["system_metrics"] = rows
        return artifacts or None

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
        # Core HoRU scoring path: compare in low-rank coefficient space.
        assert self.common_basis is not None
        assert self.global_only_basis is not None
        full_memory = self._full_memory(
            common_coords,
            common_delta_coords,
            global_only_coords,
            personal_coords,
            personal_basis,
        )

        common_query = x_hv @ self.common_basis
        global_query = x_hv @ self.global_only_basis
        personal_query = x_hv @ personal_basis

        full_query = torch.cat([common_query, global_query, personal_query], dim=1)
        full_coords = torch.cat(
            [common_coords + common_delta_coords, global_only_coords, personal_coords],
            dim=1,
        )
        shared_query = torch.cat([common_query, global_query], dim=1)
        shared_coords = torch.cat([common_coords, global_only_coords], dim=1)
        personal_query_coords = torch.cat([common_query, personal_query], dim=1)
        personal_branch_coords = torch.cat([common_delta_coords, personal_coords], dim=1)
        return (
            similarity_scores(full_query, full_coords, metric="cos"),
            similarity_scores(shared_query, shared_coords, metric="cos"),
            similarity_scores(personal_query_coords, personal_branch_coords, metric="cos"),
            full_memory,
        )

    def profiled_bootstrap(
        self,
        clients: list[ClientData],
        states: list[ClientState],
    ) -> tuple[list[ClientState], dict[str, object]]:
        """Bootstrap HoRU while retaining the C3 accounting contract.

        The decomposition is identical to :meth:`bootstrap`; the timings and
        cached projected queries make the one-time server/client work explicit
        for systems profiles.  The field names are retained for compatibility
        with the repository's existing profiling tests and reports.
        """
        bootstrap_started = time.perf_counter()
        memories = [state.memory.to(self.encoder.device) for state in states]

        server_basis_started = time.perf_counter()
        shared_basis = self._shared_basis_from_states(states)
        self._set_basis_split(shared_basis)
        bootstrap_server_basis_ms = (time.perf_counter() - server_basis_started) * 1000.0
        self._pending_state_updates = {}
        self._client_personal_cache = {}

        counts_by_client: dict[str, torch.Tensor] = {}
        common_totals_by_client: dict[str, torch.Tensor] = {}
        global_totals_by_client: dict[str, torch.Tensor] = {}

        assert self.common_basis is not None
        assert self.global_only_basis is not None
        server_projection_started = time.perf_counter()
        for client, state, memory in zip(clients, states, memories):
            assert state.extras is not None
            counts_by_client[client.client_id] = state.extras["train_class_counts"].detach().clone()
            common_totals_by_client[client.client_id] = memory @ self.common_basis
            global_totals_by_client[client.client_id] = memory @ self.global_only_basis
        bootstrap_server_projection_ms = (time.perf_counter() - server_projection_started) * 1000.0
        if hasattr(self, "_server_class_counts"):
            self._server_class_counts = {
                client_id: counts.detach().clone()
                for client_id, counts in counts_by_client.items()
            }

        server_consensus_started = time.perf_counter()
        common_consensus = self._aggregate_weighted_coords(common_totals_by_client, counts_by_client)
        global_consensus = self._aggregate_weighted_coords(global_totals_by_client, counts_by_client)
        zero_common_delta = torch.zeros_like(common_consensus)
        bootstrap_server_consensus_ms = (time.perf_counter() - server_consensus_started) * 1000.0
        per_client_basis_ms = bootstrap_server_basis_ms / max(len(clients), 1)
        per_client_projection_ms = bootstrap_server_projection_ms / max(len(clients), 1)
        per_client_consensus_ms = bootstrap_server_consensus_ms / max(len(clients), 1)

        bootstrapped: list[ClientState] = []
        client_profiles: list[dict[str, Any]] = []
        for client, state, memory in zip(clients, states, memories):
            client_started = time.perf_counter()
            client_id = client.client_id
            common_total = common_totals_by_client[client_id]

            # Even with zero bootstrap common-delta, the client-specific shared projection is
            # still needed to isolate the residual used to initialize the personal branch.
            residual_started = time.perf_counter()
            residual = memory - self._server_memory(common_total, global_consensus)
            bootstrap_residual_ms = (time.perf_counter() - residual_started) * 1000.0

            personal_basis_started = time.perf_counter()
            personal_basis = self._personal_basis_from_memory(residual)
            bootstrap_personal_basis_ms = (time.perf_counter() - personal_basis_started) * 1000.0

            personal_coords_started = time.perf_counter()
            personal_coords = residual @ personal_basis
            bootstrap_personal_coords_ms = (time.perf_counter() - personal_coords_started) * 1000.0

            full_memory_started = time.perf_counter()
            full_memory = self._full_memory(
                common_consensus,
                zero_common_delta,
                global_consensus,
                personal_coords,
                personal_basis,
            )
            bootstrap_full_memory_ms = (time.perf_counter() - full_memory_started) * 1000.0
            extras = {} if state.extras is None else dict(state.extras)
            extras.update(
                {
                    "full_memory": full_memory.detach().clone(),
                    "common_coords": common_consensus.detach().clone(),
                    "common_delta_coords": zero_common_delta.detach().clone(),
                    "global_only_coords": global_consensus.detach().clone(),
                    "personal_coords": personal_coords.detach().clone(),
                    "personal_basis": personal_basis.detach().clone(),
                    "alpha": 0.0,
                }
            )
            cached_train_hv = extras.get("cached_train_hv")
            if isinstance(cached_train_hv, torch.Tensor):
                train_hv = cached_train_hv.to(self.encoder.device)
                cached_common_query = train_hv @ self.common_basis
                cached_global_query = train_hv @ self.global_only_basis
                cached_personal_query = train_hv @ personal_basis
                cached_full_query = torch.cat(
                    [cached_common_query, cached_global_query, cached_personal_query], dim=1
                )
                extras.update(
                    {
                        "cached_common_query": cached_common_query.detach().clone(),
                        "cached_global_query": cached_global_query.detach().clone(),
                        "cached_personal_query": cached_personal_query.detach().clone(),
                        "cached_full_query": cached_full_query.detach().clone(),
                        "cached_full_query_norms": torch.linalg.norm(cached_full_query, dim=1).detach().clone(),
                    }
                )
            self._client_personal_cache[client_id] = {
                "common_delta_coords": zero_common_delta.detach().clone(),
                "personal_coords": personal_coords.detach().clone(),
                "personal_basis": personal_basis.detach().clone(),
            }
            bootstrapped.append(ClientState(memory=full_memory.detach().clone(), extras=extras))
            client_profiles.append(
                {
                    "client_id": client_id,
                    "bootstrap_projection_ms": (time.perf_counter() - client_started) * 1000.0,
                    "bootstrap_server_basis_ms": per_client_basis_ms,
                    "bootstrap_server_projection_ms": per_client_projection_ms,
                    "bootstrap_server_consensus_ms": per_client_consensus_ms,
                    "bootstrap_residual_ms": bootstrap_residual_ms,
                    "bootstrap_personal_basis_ms": bootstrap_personal_basis_ms,
                    "bootstrap_personal_coords_ms": bootstrap_personal_coords_ms,
                    "bootstrap_full_memory_ms": bootstrap_full_memory_ms,
                }
            )
        return bootstrapped, {
            "total_bootstrap_ms": (time.perf_counter() - bootstrap_started) * 1000.0,
            "bootstrap_server_basis_ms": bootstrap_server_basis_ms,
            "bootstrap_server_projection_ms": bootstrap_server_projection_ms,
            "bootstrap_server_consensus_ms": bootstrap_server_consensus_ms,
            "client_profiles": client_profiles,
        }

    def bootstrap(self, clients: list[ClientData], states: list[ClientState]) -> list[ClientState]:
        bootstrapped, _ = self.profiled_bootstrap(clients, states)
        return bootstrapped


# Backward-compatible alias for legacy imports/checkpoints.
SubspaceTrialRowGateCommonDeltaZeroCommonBasisMethod = HoRUCoreMethod
