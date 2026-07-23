"""Offline-cacheable loader for the official UCI-HAR archive."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
DOI = "10.24432/C54S4K"
CACHE_NAME = "processed.pt"


@dataclass
class UCIHARData:
    """Combined, normalized UCI-HAR tensors retaining subject and source row IDs."""

    features: torch.Tensor
    labels: torch.Tensor
    subjects: torch.Tensor
    row_indices: torch.Tensor
    manifest: dict


def prepare_data(data_root: str | Path) -> UCIHARData:
    """Download official data only when necessary, validate it, and write local cache."""
    root = Path(data_root) / "ucihar"
    cache = root / CACHE_NAME
    if cache.exists():
        return load_cache(data_root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "source.zip"
    with urllib.request.urlopen(URL, timeout=60) as response, archive.open("wb") as output:
        shutil.copyfileobj(response, output)
    digest = _sha256(archive)
    manifest = {"source_url": URL, "sha256": digest, "file_size": archive.stat().st_size,
                "downloaded_at": datetime.now(timezone.utc).isoformat(), "dataset_doi": DOI,
                "preprocessing": {"labels": "one_based_to_zero_based", "features": "sample_wise_l2"}}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(tmp)
        candidates = list(Path(tmp).rglob("UCI HAR Dataset"))
        # The current official UCI distribution wraps the original archive once.
        for nested in Path(tmp).rglob("*.zip"):
            with zipfile.ZipFile(nested) as inner:
                inner.extractall(nested.parent)
        candidates = list(Path(tmp).rglob("UCI HAR Dataset"))
        if not candidates:
            raise RuntimeError("official UCI-HAR archive has unexpected layout")
        data = _read_raw(candidates[0], manifest)
    torch.save(_payload(data), cache)
    return data


def load_cache(data_root: str | Path) -> UCIHARData:
    """Load and validate the prepared local cache without any network access."""
    root = Path(data_root) / "ucihar"
    cache = root / CACHE_NAME
    if not cache.exists():
        raise FileNotFoundError(f"UCI-HAR cache is absent at {cache}; run prepare-data ucihar first")
    payload = torch.load(cache, map_location="cpu", weights_only=False)
    data = UCIHARData(**payload)
    _validate(data, strict_metadata=False)
    return data


def split_subjects(data: UCIHARData, subject_ids: list[int], test_ratio: float, seed: int) -> dict[int, dict[str, torch.Tensor]]:
    """Produce deterministic per-subject stratified train/test index splits."""
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between zero and one")
    selected: dict[int, dict[str, torch.Tensor]] = {}
    for subject in subject_ids:
        where_subject = torch.nonzero(data.subjects == subject, as_tuple=False).flatten()
        if where_subject.numel() == 0:
            raise ValueError(f"subject {subject} is absent from cache")
        train, test = [], []
        for label in range(6):
            indices = where_subject[data.labels[where_subject] == label]
            if indices.numel() < 2:
                raise ValueError(f"subject {subject}, class {label} lacks samples for split")
            generator = torch.Generator(device="cpu").manual_seed(seed + subject * 1009 + label)
            shuffled = indices[torch.randperm(indices.numel(), generator=generator)]
            count = max(1, min(indices.numel() - 1, round(indices.numel() * test_ratio)))
            test.append(shuffled[:count]); train.append(shuffled[count:])
        selected[subject] = {"train": torch.cat(train).sort().values, "test": torch.cat(test).sort().values}
    return selected


def _read_raw(base: Path, manifest: dict) -> UCIHARData:
    parts = []
    offset = 0
    for split in ("train", "test"):
        directory = base / split
        needed = [directory / f"X_{split}.txt", directory / f"y_{split}.txt", directory / f"subject_{split}.txt"]
        if not all(path.exists() for path in needed):
            raise FileNotFoundError(f"missing required UCI-HAR {split} file")
        features = torch.tensor(_read_matrix(needed[0]), dtype=torch.float32)
        labels = torch.tensor(_read_vector(needed[1]), dtype=torch.long) - 1
        subjects = torch.tensor(_read_vector(needed[2]), dtype=torch.long)
        if not (features.shape[0] == labels.numel() == subjects.numel()):
            raise ValueError(f"UCI-HAR {split} files have inconsistent row counts")
        parts.append((features, labels, subjects, torch.arange(offset, offset + labels.numel(), dtype=torch.long)))
        offset += labels.numel()
    data = UCIHARData(torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts]), torch.cat([p[2] for p in parts]), torch.cat([p[3] for p in parts]), manifest)
    data.features = torch.nn.functional.normalize(data.features, p=2, dim=1)
    _validate(data, strict_metadata=True)
    return data


def _read_matrix(path: Path) -> list[list[float]]:
    return [[float(x) for x in line.split()] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_vector(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _validate(data: UCIHARData, strict_metadata: bool) -> None:
    if data.features.ndim != 2 or data.features.shape[1] != 561: raise ValueError("UCI-HAR must have 561 features")
    if not torch.isfinite(data.features).all(): raise ValueError("UCI-HAR features contain missing or non-finite values")
    if data.labels.numel() != data.features.shape[0] or torch.any((data.labels < 0) | (data.labels > 5)): raise ValueError("invalid UCI-HAR labels")
    if data.subjects.numel() != data.features.shape[0] or torch.any((data.subjects < 1) | (data.subjects > 30)): raise ValueError("invalid UCI-HAR subjects")
    if not {"source_url", "sha256", "downloaded_at"}.issubset(data.manifest): raise ValueError("UCI-HAR cache manifest is incomplete")
    if strict_metadata and (data.features.shape[0] != 10299 or torch.unique(data.subjects).numel() != 30 or torch.unique(data.labels).numel() != 6): raise ValueError("official UCI-HAR metadata validation failed")


def _payload(data: UCIHARData) -> dict: return {"features": data.features, "labels": data.labels, "subjects": data.subjects, "row_indices": data.row_indices, "manifest": data.manifest}
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()
