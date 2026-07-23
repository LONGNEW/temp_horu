import torch
import pytest

from horu_artifact.datasets.ucihar import load_cache, split_subjects
from conftest import write_fixture_cache


def test_cache_load_normalization_and_deterministic_split(tmp_path):
    root = write_fixture_cache(tmp_path); data = load_cache(root)
    assert data.features.shape[1] == 561 and set(data.labels.tolist()) == set(range(6))
    assert torch.allclose(torch.linalg.vector_norm(data.features, dim=1), torch.ones(data.features.shape[0]))
    first, second = split_subjects(data, [1, 2, 3], .3, 0), split_subjects(data, [1, 2, 3], .3, 0)
    assert torch.equal(first[1]["train"], second[1]["train"])
    assert not set(first[1]["train"].tolist()) & set(first[1]["test"].tolist())
    assert first[1]["test"].numel() == 18


def test_missing_cache_is_clear(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare-data"): load_cache(tmp_path)
