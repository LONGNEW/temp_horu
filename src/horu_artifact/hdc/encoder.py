"""HDZoo-compatible non-binary nonlinear encoder."""

import torch


def make_projection(input_dim: int, hd_dim: int, seed: int) -> torch.Tensor:
    """Create the CPU projection in HDZoo's Gaussian base generation order."""
    if input_dim <= 0 or hd_dim <= 0:
        raise ValueError("input_dim and hd_dim must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    raw = torch.empty((hd_dim, input_dim), dtype=torch.float32, device="cpu")
    raw.normal_(mean=0.0, std=1.0, generator=generator)
    return raw.transpose(0, 1).contiguous()


class NonlinearEncoder:
    """Encode feature batches as ``cos(X @ E)`` without normalization or signs."""

    def __init__(self, projection: torch.Tensor) -> None:
        if projection.ndim != 2 or projection.numel() == 0:
            raise ValueError("projection must be a non-empty rank-2 tensor")
        if projection.dtype != torch.float32:
            raise TypeError("projection must have torch.float32 dtype")
        self.projection = projection

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """Encode a non-empty ``(batch, input_dim)`` tensor on the same device."""
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("features must be a non-empty rank-2 tensor")
        if features.shape[1] != self.projection.shape[0]:
            raise ValueError("feature dimension does not match projection")
        if features.device != self.projection.device:
            raise ValueError("features and projection must be on the same device")
        if features.dtype != torch.float32:
            raise TypeError("features must have torch.float32 dtype")
        return torch.cos(features @ self.projection)
