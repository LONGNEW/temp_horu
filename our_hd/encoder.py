from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class BaseHDEncoder(ABC):
    """Shared encoder interface for all HD methods."""

    def __init__(self, input_dim: int, hd_dim: int, *, device: torch.device | str = "cpu") -> None:
        self.input_dim = input_dim
        self.hd_dim = hd_dim
        self.device = torch.device(device)

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def hardsign(x: torch.Tensor) -> torch.Tensor:
        out = torch.ones_like(x)
        out[x < 0] = -1.0
        return out

    @staticmethod
    def row_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return x / (torch.linalg.norm(x, dim=1, keepdim=True) + eps)


class RandomProjectionEncoder(BaseHDEncoder):
    """Linear random projection followed by optional binarization."""

    def __init__(
        self,
        input_dim: int,
        hd_dim: int,
        *,
        binary: bool = True,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(input_dim, hd_dim, device=device)
        self.binary = binary
        self.projection = torch.randn(input_dim, hd_dim, device=self.device)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        hv = x.to(self.device) @ self.projection
        if self.binary:
            hv = self.hardsign(hv)
        return hv


class CosineProjectionEncoder(BaseHDEncoder):
    """Random projection with cosine nonlinearity, matching common HD practice."""

    def __init__(
        self,
        input_dim: int,
        hd_dim: int,
        *,
        binary: bool = False,
        random_phase: bool = False,
        seed: int | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(input_dim, hd_dim, device=device)
        self.binary = binary
        self.random_phase = random_phase
        self.seed = None if seed is None else int(seed)
        if self.seed is None:
            self.projection = torch.randn(input_dim, hd_dim, device=self.device)
            self.phase = (
                (2.0 * torch.pi) * torch.rand(hd_dim, device=self.device)
                if self.random_phase
                else None
            )
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed)
            self.projection = torch.randn(
                input_dim,
                hd_dim,
                generator=generator,
                dtype=torch.float32,
            ).to(self.device)
            self.phase = (
                (2.0 * torch.pi)
                * torch.rand(
                    hd_dim,
                    generator=generator,
                    dtype=torch.float32,
                ).to(self.device)
                if self.random_phase
                else None
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        logits = x.to(self.device) @ self.projection
        if self.phase is not None:
            logits = logits + self.phase
        hv = torch.cos(logits)
        if self.binary:
            hv = self.hardsign(hv)
        return self.row_normalize(hv)


class MaskedCosineProjectionEncoder(BaseHDEncoder):
    """Cosine random projection where each HD dimension only sees a feature subset."""

    def __init__(
        self,
        input_dim: int,
        hd_dim: int,
        *,
        subspace_size: int,
        binary: bool = False,
        random_phase: bool = False,
        mask_seed: int = 13,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(input_dim, hd_dim, device=device)
        self.binary = binary
        self.random_phase = random_phase
        self.subspace_size = max(1, min(int(subspace_size), int(input_dim)))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(mask_seed))

        projection = torch.zeros(input_dim, hd_dim, dtype=torch.float32)
        for col in range(hd_dim):
            indices = torch.randperm(input_dim, generator=generator)[: self.subspace_size]
            projection[indices, col] = torch.randn(self.subspace_size, generator=generator)
        self.projection = projection.to(self.device)
        self.phase = (
            (2.0 * torch.pi) * torch.rand(hd_dim, generator=generator, dtype=torch.float32).to(self.device)
            if self.random_phase
            else None
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        logits = x.to(self.device) @ self.projection
        if self.phase is not None:
            logits = logits + self.phase
        hv = torch.cos(logits)
        if self.binary:
            hv = self.hardsign(hv)
        return self.row_normalize(hv)


class GroupLearnedCosineProjectionEncoder(BaseHDEncoder):
    """MLP-guided grouped cosine encoder with offline feature-group discovery."""

    def __init__(
        self,
        input_dim: int,
        hd_dim: int,
        *,
        num_groups: int = 4,
        feature_topk: int | None = None,
        mlp_hidden_dim: int | None = None,
        mlp_epochs: int = 30,
        mlp_lr: float = 1e-2,
        binary: bool = False,
        random_phase: bool = False,
        mask_seed: int = 13,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(input_dim, hd_dim, device=device)
        self.binary = binary
        self.random_phase = random_phase
        self.num_groups = max(1, int(num_groups))
        self.feature_topk = (
            max(1, min(int(feature_topk), input_dim))
            if feature_topk is not None
            else max(1, input_dim // self.num_groups)
        )
        self.mlp_hidden_dim = max(1, int(mlp_hidden_dim or self.num_groups))
        self.mlp_epochs = max(1, int(mlp_epochs))
        self.mlp_lr = float(mlp_lr)
        self.mask_seed = int(mask_seed)
        self._is_fitted = False
        self.projection = torch.randn(input_dim, hd_dim, device=self.device)
        self.group_weights = torch.ones(self.num_groups, dtype=torch.float32, device=self.device) / float(self.num_groups)
        self.group_ids = torch.arange(hd_dim, device=self.device) % self.num_groups
        self.phase = (
            (2.0 * torch.pi) * torch.rand(hd_dim, device=self.device)
            if self.random_phase
            else None
        )

    def fit(self, x: torch.Tensor, y: torch.Tensor, num_classes: int) -> None:
        x_cpu = x.detach().to("cpu")
        y_cpu = y.detach().to("cpu").long()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.mask_seed)

        model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.mlp_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.mlp_hidden_dim, int(num_classes)),
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.mlp_lr)

        x_train = x_cpu.to(self.device)
        y_train = y_cpu.to(self.device)
        batch_size = min(1024, int(x_train.shape[0]))

        model.train()
        for _ in range(self.mlp_epochs):
            permutation = torch.randperm(int(x_train.shape[0]), generator=generator)
            for start in range(0, int(x_train.shape[0]), batch_size):
                index = permutation[start : start + batch_size].to(x_train.device)
                logits = model(x_train.index_select(0, index))
                loss = F.cross_entropy(logits, y_train.index_select(0, index))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        first = model[0].weight.detach().abs()  # [hidden, p]
        second = model[2].weight.detach().abs()  # [classes, hidden]
        hidden_importance = second.mean(dim=0)  # [hidden]

        # Map hidden units into groups and aggregate feature relevance.
        group_relevance = torch.zeros(self.num_groups, self.input_dim, device=self.device)
        for h in range(self.mlp_hidden_dim):
            group_idx = h % self.num_groups
            group_relevance[group_idx].add_(hidden_importance[h] * first[h])

        masks = torch.zeros(self.num_groups, self.input_dim, dtype=torch.float32, device=self.device)
        for g in range(self.num_groups):
            topk = min(self.feature_topk, self.input_dim)
            top_idx = torch.topk(group_relevance[g], k=topk, largest=True).indices
            masks[g, top_idx] = 1.0
            if float(masks[g].sum().item()) <= 0.0:
                fallback_idx = torch.randperm(self.input_dim, generator=generator)[:topk].to(self.device)
                masks[g, fallback_idx] = 1.0

        group_weights = group_relevance.sum(dim=1).clamp_min(1e-8)
        group_weights = group_weights / group_weights.sum()

        dims_per_group = [self.hd_dim // self.num_groups for _ in range(self.num_groups)]
        for idx in range(self.hd_dim % self.num_groups):
            dims_per_group[idx] += 1

        projection = torch.zeros(self.input_dim, self.hd_dim, dtype=torch.float32, device=self.device)
        group_ids = torch.empty(self.hd_dim, dtype=torch.long, device=self.device)
        phase = torch.zeros(self.hd_dim, dtype=torch.float32, device=self.device)
        cursor = 0
        for g, dim_count in enumerate(dims_per_group):
            feat_idx = torch.nonzero(masks[g] > 0, as_tuple=False).squeeze(1)
            if feat_idx.numel() == 0:
                feat_idx = torch.randperm(self.input_dim, generator=generator)[:1].to(self.device)
            scale = torch.sqrt(group_weights[g]).item()
            for _ in range(dim_count):
                rand_w = torch.randn(int(feat_idx.numel()), generator=generator, dtype=torch.float32).to(self.device)
                projection[feat_idx, cursor] = rand_w * scale
                group_ids[cursor] = g
                if self.random_phase:
                    phase[cursor] = (2.0 * torch.pi) * torch.rand((), generator=generator, dtype=torch.float32).item()
                cursor += 1

        self.projection = projection
        self.group_ids = group_ids
        self.group_weights = group_weights
        self.phase = phase if self.random_phase else None
        self._is_fitted = True

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        logits = x.to(self.device) @ self.projection
        if self.phase is not None:
            logits = logits + self.phase
        hv = torch.cos(logits)
        if self.binary:
            hv = self.hardsign(hv)
        return self.row_normalize(hv)


class PackedAdditiveCosineEncoder(BaseHDEncoder):
    """Single-vector packed additive encoder with block/superposition/hash packing."""

    def __init__(
        self,
        input_dim: int,
        hd_dim: int,
        *,
        packing_mode: str = "block",
        num_groups: int = 4,
        feature_group_mode: str = "random_partition",
        feature_topk: int | None = None,
        source_dim: int | None = None,
        mlp_hidden_dim: int | None = None,
        mlp_epochs: int = 30,
        mlp_lr: float = 1e-2,
        binary: bool = False,
        random_phase: bool = False,
        mask_seed: int = 13,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(input_dim, hd_dim, device=device)
        mode = str(packing_mode).lower()
        if mode not in {"block", "superposition", "hash"}:
            raise ValueError(f"Unsupported packing_mode: {packing_mode}")
        group_mode = str(feature_group_mode).lower()
        if group_mode not in {"random_partition", "random_topk", "contiguous_partition", "mlp"}:
            raise ValueError(f"Unsupported feature_group_mode: {feature_group_mode}")

        self.packing_mode = mode
        self.feature_group_mode = group_mode
        self.num_groups = max(1, int(num_groups))
        self.feature_topk = (
            max(1, min(int(feature_topk), int(input_dim)))
            if feature_topk is not None
            else max(1, int(input_dim) // self.num_groups)
        )
        self.source_dim = (
            max(1, int(source_dim))
            if source_dim is not None
            else (int(hd_dim) if mode in {"superposition", "hash"} else int(hd_dim))
        )
        if mode == "superposition" and self.source_dim != self.hd_dim:
            raise ValueError(
                f"superposition packing requires source_dim == hd_dim ({self.hd_dim}), got {self.source_dim}"
            )
        self.mlp_hidden_dim = max(1, int(mlp_hidden_dim or self.num_groups))
        self.mlp_epochs = max(1, int(mlp_epochs))
        self.mlp_lr = float(mlp_lr)
        self.binary = bool(binary)
        self.random_phase = bool(random_phase)
        self.mask_seed = int(mask_seed)
        self._fitted = False

        self.group_weights = torch.ones(self.num_groups, dtype=torch.float32, device=self.device)
        self.group_weights = self.group_weights / self.group_weights.sum().clamp_min(1e-8)
        self.group_masks = torch.zeros(self.num_groups, self.input_dim, dtype=torch.float32, device=self.device)

        self.block_projection = torch.zeros(self.input_dim, self.hd_dim, dtype=torch.float32, device=self.device)
        self.block_phase = torch.zeros(self.hd_dim, dtype=torch.float32, device=self.device) if self.random_phase else None

        self.group_projection = torch.zeros(
            self.num_groups,
            self.input_dim,
            self.source_dim,
            dtype=torch.float32,
            device=self.device,
        )
        self.group_phase = (
            torch.zeros(self.num_groups, self.source_dim, dtype=torch.float32, device=self.device)
            if self.random_phase
            else None
        )
        self.group_keys = self._random_sign(self.num_groups, self.source_dim)
        self.hash_index = torch.zeros(self.num_groups, self.source_dim, dtype=torch.long, device=self.device)
        self.hash_sign = self._random_sign(self.num_groups, self.source_dim)

        if self.feature_group_mode != "mlp":
            self._init_group_masks_without_labels()
            self._rebuild_projection_tensors()

    def _random_sign(self, *shape: int) -> torch.Tensor:
        rng = torch.Generator(device="cpu")
        rng.manual_seed(self.mask_seed + 17)
        bits = torch.randint(0, 2, shape, generator=rng, dtype=torch.int64)
        return ((bits * 2) - 1).to(self.device, dtype=torch.float32)

    def _dims_per_group(self) -> list[int]:
        dims = [self.hd_dim // self.num_groups for _ in range(self.num_groups)]
        for idx in range(self.hd_dim % self.num_groups):
            dims[idx] += 1
        return dims

    def _init_group_masks_without_labels(self) -> None:
        rng = torch.Generator(device="cpu")
        rng.manual_seed(self.mask_seed)
        masks = torch.zeros(self.num_groups, self.input_dim, dtype=torch.float32)
        if self.feature_group_mode == "contiguous_partition":
            boundaries = [0]
            step = self.input_dim // self.num_groups
            rem = self.input_dim % self.num_groups
            cursor = 0
            for g in range(self.num_groups):
                cursor += step + (1 if g < rem else 0)
                boundaries.append(cursor)
            for g in range(self.num_groups):
                left, right = boundaries[g], boundaries[g + 1]
                if left < right:
                    masks[g, left:right] = 1.0
        elif self.feature_group_mode == "random_partition":
            perm = torch.randperm(self.input_dim, generator=rng)
            chunks = torch.tensor_split(perm, self.num_groups)
            for g, idx in enumerate(chunks):
                if idx.numel() > 0:
                    masks[g, idx] = 1.0
        else:  # random_topk
            for g in range(self.num_groups):
                idx = torch.randperm(self.input_dim, generator=rng)[: self.feature_topk]
                masks[g, idx] = 1.0

        for g in range(self.num_groups):
            if float(masks[g].sum().item()) <= 0.0:
                fallback = torch.randperm(self.input_dim, generator=rng)[:1]
                masks[g, fallback] = 1.0
        self.group_masks = masks.to(self.device)

    def _learn_group_masks_with_mlp(self, x: torch.Tensor, y: torch.Tensor, num_classes: int) -> None:
        x_train = x.detach().to(self.device)
        y_train = y.detach().to(self.device).long()
        rng = torch.Generator(device="cpu")
        rng.manual_seed(self.mask_seed)

        model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.mlp_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.mlp_hidden_dim, int(num_classes)),
        ).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.mlp_lr)
        batch_size = min(1024, int(x_train.shape[0]))

        model.train()
        for _ in range(self.mlp_epochs):
            perm = torch.randperm(int(x_train.shape[0]), generator=rng)
            for start in range(0, int(x_train.shape[0]), batch_size):
                idx = perm[start : start + batch_size].to(self.device)
                logits = model(x_train.index_select(0, idx))
                loss = F.cross_entropy(logits, y_train.index_select(0, idx))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        first = model[0].weight.detach().abs()  # [hidden, p]
        second = model[2].weight.detach().abs()  # [classes, hidden]
        hidden_importance = second.mean(dim=0)  # [hidden]

        relevance = torch.zeros(self.num_groups, self.input_dim, device=self.device)
        for h in range(self.mlp_hidden_dim):
            relevance[h % self.num_groups].add_(hidden_importance[h] * first[h])

        masks = torch.zeros(self.num_groups, self.input_dim, dtype=torch.float32, device=self.device)
        for g in range(self.num_groups):
            topk = min(self.feature_topk, self.input_dim)
            idx = torch.topk(relevance[g], k=topk, largest=True).indices
            masks[g, idx] = 1.0
            if float(masks[g].sum().item()) <= 0.0:
                fallback = torch.randperm(self.input_dim, generator=rng)[:1].to(self.device)
                masks[g, fallback] = 1.0

        weights = relevance.sum(dim=1).clamp_min(1e-8)
        weights = weights / weights.sum().clamp_min(1e-8)
        self.group_masks = masks
        self.group_weights = weights

    def _rebuild_projection_tensors(self) -> None:
        rng = torch.Generator(device="cpu")
        rng.manual_seed(self.mask_seed + 1)
        sqrt_weights = torch.sqrt(self.group_weights.clamp_min(1e-8))

        if self.packing_mode == "block":
            projection = torch.zeros(self.input_dim, self.hd_dim, dtype=torch.float32, device=self.device)
            phase = (
                torch.zeros(self.hd_dim, dtype=torch.float32, device=self.device)
                if self.random_phase
                else None
            )
            dims = self._dims_per_group()
            cursor = 0
            for g, dim_count in enumerate(dims):
                feat_idx = torch.nonzero(self.group_masks[g] > 0, as_tuple=False).squeeze(1)
                if feat_idx.numel() == 0:
                    feat_idx = torch.tensor([g % self.input_dim], device=self.device)
                for _ in range(dim_count):
                    rand_w = torch.randn(int(feat_idx.numel()), generator=rng, dtype=torch.float32).to(self.device)
                    projection[feat_idx, cursor] = rand_w * sqrt_weights[g]
                    if phase is not None:
                        phase[cursor] = (2.0 * torch.pi) * torch.rand((), generator=rng, dtype=torch.float32).item()
                    cursor += 1
            self.block_projection = projection
            self.block_phase = phase
            return

        group_projection = torch.zeros(
            self.num_groups,
            self.input_dim,
            self.source_dim,
            dtype=torch.float32,
            device=self.device,
        )
        group_phase = (
            torch.zeros(self.num_groups, self.source_dim, dtype=torch.float32, device=self.device)
            if self.random_phase
            else None
        )
        for g in range(self.num_groups):
            feat_idx = torch.nonzero(self.group_masks[g] > 0, as_tuple=False).squeeze(1)
            if feat_idx.numel() == 0:
                feat_idx = torch.tensor([g % self.input_dim], device=self.device)
            rand_w = torch.randn(int(feat_idx.numel()), self.source_dim, generator=rng, dtype=torch.float32).to(self.device)
            group_projection[g, feat_idx, :] = rand_w
            if group_phase is not None:
                group_phase[g] = (2.0 * torch.pi) * torch.rand(self.source_dim, generator=rng, dtype=torch.float32).to(self.device)

        self.group_projection = group_projection
        self.group_phase = group_phase
        self.group_keys = self._random_sign(self.num_groups, self.source_dim)
        self.hash_index = torch.randint(
            low=0,
            high=self.hd_dim,
            size=(self.num_groups, self.source_dim),
            generator=rng,
            dtype=torch.long,
        ).to(self.device)
        self.hash_sign = self._random_sign(self.num_groups, self.source_dim)

    def fit(self, x: torch.Tensor, y: torch.Tensor, num_classes: int) -> None:
        if self.feature_group_mode == "mlp":
            self._learn_group_masks_with_mlp(x, y, num_classes)
        elif float(self.group_masks.sum().item()) <= 0.0:
            self._init_group_masks_without_labels()
        self._rebuild_projection_tensors()
        self._fitted = True

    def _encode_block(self, x: torch.Tensor) -> torch.Tensor:
        logits = x @ self.block_projection
        if self.block_phase is not None:
            logits = logits + self.block_phase
        hv = torch.cos(logits)
        if self.binary:
            hv = self.hardsign(hv)
        return self.row_normalize(hv)

    def _encode_superposition(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(x.shape[0], self.hd_dim, dtype=torch.float32, device=self.device)
        for g in range(self.num_groups):
            logits = x @ self.group_projection[g]
            if self.group_phase is not None:
                logits = logits + self.group_phase[g]
            hv = torch.cos(logits)
            hv = self.row_normalize(hv)
            out = out + (torch.sqrt(self.group_weights[g]) * (hv * self.group_keys[g]))
        if self.binary:
            out = self.hardsign(out)
        return self.row_normalize(out)

    def _encode_hash(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(x.shape[0], self.hd_dim, dtype=torch.float32, device=self.device)
        for g in range(self.num_groups):
            logits = x @ self.group_projection[g]
            if self.group_phase is not None:
                logits = logits + self.group_phase[g]
            hv = torch.cos(logits)
            hv = self.row_normalize(hv)
            weighted = torch.sqrt(self.group_weights[g]) * (hv * self.hash_sign[g])
            index = self.hash_index[g].unsqueeze(0).expand(weighted.shape[0], -1)
            out.scatter_add_(1, index, weighted)
        if self.binary:
            out = self.hardsign(out)
        return self.row_normalize(out)

    def encode_group_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return per-group encoded features before packing (each row-normalized)."""
        if not self._fitted and self.feature_group_mode == "mlp":
            raise RuntimeError("PackedAdditiveCosineEncoder with feature_group_mode='mlp' requires fit() before encode().")
        x = x.to(self.device)
        group_hv: list[torch.Tensor] = []
        for g in range(self.num_groups):
            logits = x @ self.group_projection[g]
            if self.group_phase is not None:
                logits = logits + self.group_phase[g]
            hv = torch.cos(logits)
            if self.binary:
                hv = self.hardsign(hv)
            group_hv.append(self.row_normalize(hv))
        return group_hv

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if not self._fitted and self.feature_group_mode == "mlp":
            raise RuntimeError("PackedAdditiveCosineEncoder with feature_group_mode='mlp' requires fit() before encode().")
        x = x.to(self.device)
        if self.packing_mode == "block":
            return self._encode_block(x)
        if self.packing_mode == "superposition":
            return self._encode_superposition(x)
        return self._encode_hash(x)


class ResidualPackedCosineEncoder(BaseHDEncoder):
    """Budget-split residual encoder: concat(full branch, packed group branch) within one HD vector."""

    def __init__(
        self,
        input_dim: int,
        hd_dim: int,
        *,
        eta: float = 0.5,
        group_packing_mode: str = "hash",
        num_groups: int = 4,
        feature_group_mode: str = "random_partition",
        feature_topk: int | None = None,
        source_dim: int | None = None,
        mlp_hidden_dim: int | None = None,
        mlp_epochs: int = 30,
        mlp_lr: float = 1e-2,
        binary: bool = False,
        random_phase: bool = False,
        mask_seed: int = 13,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(input_dim, hd_dim, device=device)
        self.binary = bool(binary)
        self.random_phase = bool(random_phase)
        self.group_packing_mode = str(group_packing_mode)
        self.num_groups = int(num_groups)
        self.feature_group_mode = str(feature_group_mode)
        self.feature_topk = feature_topk
        self.source_dim = source_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_epochs = int(mlp_epochs)
        self.mlp_lr = float(mlp_lr)
        self.mask_seed = int(mask_seed)
        self.full_seed = self.mask_seed + 101

        self.eta = 0.0
        self.group_dim = 0
        self.full_dim = self.hd_dim
        self.full_scale = 1.0
        self.group_scale = 0.0
        self.full_encoder: CosineProjectionEncoder | None = None
        self.group_encoder: PackedAdditiveCosineEncoder | None = None
        self.set_eta(float(eta))

    @staticmethod
    def _resolve_eta(eta: float) -> float:
        return float(min(1.0, max(0.0, float(eta))))

    def _update_eta_dimensions(self, eta: float) -> None:
        self.eta = self._resolve_eta(eta)
        group_dim = int(round(self.hd_dim * self.eta))
        if self.eta > 0.0:
            group_dim = max(1, group_dim)
        if self.eta < 1.0:
            group_dim = min(self.hd_dim - 1, group_dim) if self.hd_dim > 1 else 0
        self.group_dim = max(0, min(group_dim, self.hd_dim))
        self.full_dim = int(self.hd_dim - self.group_dim)
        self.full_scale = float(torch.sqrt(torch.tensor(max(0.0, 1.0 - self.eta))).item())
        self.group_scale = float(torch.sqrt(torch.tensor(max(0.0, self.eta))).item())

    def _rebuild_branches(self) -> None:
        self.full_encoder = None
        if self.full_dim > 0:
            self.full_encoder = CosineProjectionEncoder(
                input_dim=self.input_dim,
                hd_dim=self.full_dim,
                binary=self.binary,
                random_phase=self.random_phase,
                seed=self.full_seed,
                device=self.device,
            )

        self.group_encoder = None
        if self.group_dim > 0:
            self.group_encoder = PackedAdditiveCosineEncoder(
                input_dim=self.input_dim,
                hd_dim=self.group_dim,
                packing_mode=self.group_packing_mode,
                num_groups=self.num_groups,
                feature_group_mode=self.feature_group_mode,
                feature_topk=self.feature_topk,
                source_dim=self.source_dim,
                mlp_hidden_dim=self.mlp_hidden_dim,
                mlp_epochs=self.mlp_epochs,
                mlp_lr=self.mlp_lr,
                binary=self.binary,
                random_phase=self.random_phase,
                mask_seed=self.mask_seed,
                device=self.device,
            )

    def set_eta(self, eta: float) -> None:
        self._update_eta_dimensions(eta)
        self._rebuild_branches()

    def fit(self, x: torch.Tensor, y: torch.Tensor, num_classes: int) -> None:
        if self.group_encoder is not None:
            self.group_encoder.fit(x, y, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        parts: list[torch.Tensor] = []
        if self.full_encoder is not None and self.full_scale > 0.0:
            parts.append(self.full_scale * self.full_encoder.encode(x))
        if self.group_encoder is not None and self.group_scale > 0.0:
            parts.append(self.group_scale * self.group_encoder.encode(x))
        if not parts:
            raise RuntimeError("ResidualPackedCosineEncoder has no active branch.")
        if len(parts) == 1:
            return self.row_normalize(parts[0])
        return self.row_normalize(torch.cat(parts, dim=1))
