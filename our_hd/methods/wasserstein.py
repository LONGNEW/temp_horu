from __future__ import annotations

import torch


def normalize_vec(vec: torch.Tensor) -> torch.Tensor:
    return vec / torch.linalg.norm(vec, dim=0, keepdim=False).clamp_min(1e-8)


def kmeans_centers_labels_sse(
    points: torch.Tensor,
    *,
    n_clusters: int,
    max_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    n_samples = int(points.shape[0])
    if n_samples == 0:
        return points, torch.zeros(0, dtype=torch.long, device=points.device), 0.0

    n_clusters = max(1, min(int(n_clusters), n_samples))
    init_idx = (
        torch.linspace(0, n_samples - 1, steps=n_clusters, device=points.device)
        .round()
        .long()
    )
    centers = points.index_select(0, init_idx).clone()

    for _ in range(int(max_iters)):
        distances = torch.cdist(points, centers, p=2.0)
        labels = torch.argmin(distances, dim=1)
        counts = torch.bincount(labels, minlength=n_clusters)
        new_centers = torch.zeros_like(centers)
        new_centers.index_add_(0, labels, points)
        nonempty = counts > 0
        if bool(nonempty.any()):
            new_centers[nonempty] = new_centers[nonempty] / counts[nonempty].to(points.dtype).unsqueeze(1)
        if bool((~nonempty).any()):
            new_centers[~nonempty] = centers[~nonempty]
        if torch.allclose(new_centers, centers, atol=1e-6):
            centers = new_centers
            break
        centers = new_centers

    final_dist = torch.cdist(points, centers, p=2.0)
    min_dist, labels = torch.min(final_dist, dim=1)
    return centers, labels, float((min_dist ** 2).sum().item())
