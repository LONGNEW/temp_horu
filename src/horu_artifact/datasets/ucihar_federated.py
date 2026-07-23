"""T006 UCI-HAR adapter from the original verified cache."""
from __future__ import annotations
from pathlib import Path
from .ucihar import load_cache, split_subjects
from .federated import ClientData, FederatedDataset, write_cache


def prepare_data(data_root: str | Path, seed: int = 0) -> FederatedDataset:
    raw = load_cache(data_root)
    splits = split_subjects(raw, list(range(1, 31)), .3, seed)
    clients = {f"{subject:03d}": ClientData(raw.features[ix["train"]], raw.labels[ix["train"]], raw.features[ix["test"]], raw.labels[ix["test"]], raw.row_indices[ix["train"]], raw.row_indices[ix["test"]]) for subject,ix in splits.items()}
    manifest = dict(raw.manifest)
    manifest.update({"parser": "ucihar_official_v1", "clients": 30, "features": 561, "classes": 6,
                     "split": "subject_internal_stratified_70_30_singleton_train_only", "seed": seed,
                     "normalization": "samplewise_l2", "provenance": "USER_SPECIFIED_PERSONALIZED_SPLIT"})
    return write_cache(FederatedDataset("ucihar", clients, 561, 6, manifest), data_root)
