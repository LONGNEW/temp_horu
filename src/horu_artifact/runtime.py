"""Device selection and reproducibility helpers."""

import torch


def resolve_device(requested: str) -> torch.device:
    """Resolve an allowed device request without silently falling back from cuda."""
    if requested == "auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda"):
        if not torch.cuda.is_available(): raise RuntimeError("CUDA was requested but is unavailable")
        try:
            index = int(requested.split(":", 1)[1]) if ":" in requested else 0
        except ValueError as error:
            raise ValueError("CUDA device must be cuda or cuda:<index>") from error
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} is unavailable")
        return torch.device(f"cuda:{index}")
    if requested not in {"cpu", "auto"}: raise ValueError("device must be cpu, cuda, cuda:<index>, or auto")
    return torch.device(requested)
