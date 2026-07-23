import json
from pathlib import Path

import torch


def write_fixture_cache(root: Path) -> Path:
    """Create a local, network-free UCI-HAR-shaped cache usable by integration tests."""
    target = root / "ucihar"; target.mkdir(parents=True)
    rows, labels, subjects = [], [], []
    for subject in (1, 2, 3):
        for label in range(6):
            for item in range(10):
                # Shared vectors deliberately induce nonzero initial classification errors.
                value = torch.zeros(561); value[0] = 1.0; value[10 + item] = 0.1
                rows.append(value); labels.append(label); subjects.append(subject)
    features = torch.nn.functional.normalize(torch.stack(rows), p=2, dim=1)
    manifest = {"source_url": "fixture://ucihar", "sha256": "fixture-sha", "file_size": 1, "downloaded_at": "now", "dataset_doi": "10.24432/C54S4K"}
    torch.save({"features": features, "labels": torch.tensor(labels, dtype=torch.long), "subjects": torch.tensor(subjects, dtype=torch.long), "row_indices": torch.arange(len(labels)), "manifest": manifest}, target / "processed.pt")
    (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root
