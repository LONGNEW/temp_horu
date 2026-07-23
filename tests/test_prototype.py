import pytest
import torch

from horu_artifact.hdc.prototype import PrototypeMemory


def test_initialization_prediction_and_zero_rows():
    encoded = torch.tensor([[1., 0.], [2., 0.], [0., 1.]])
    labels = torch.tensor([0, 0, 1], dtype=torch.long)
    memory = PrototypeMemory.initialize(encoded, labels, 3)
    assert memory.memory.shape == (3, 2)
    assert torch.allclose(torch.linalg.vector_norm(memory.memory[:2], dim=1), torch.ones(2))
    assert torch.equal(memory.memory[2], torch.zeros(2))
    assert memory.predict(torch.tensor([[.1, .9]])).item() == 1
    with pytest.raises(RuntimeError): PrototypeMemory(torch.zeros((2, 2))).predict(torch.ones((1, 2)))


def test_initialization_can_preserve_raw_class_means():
    encoded = torch.tensor([[1., 0.], [2., 0.], [0., 1.]])
    labels = torch.tensor([0, 0, 1], dtype=torch.long)
    raw = PrototypeMemory.initialize(encoded, labels, 3, normalize_rows=False)
    assert torch.allclose(raw.memory, torch.tensor([[1.5, 0.], [0., 1.], [0., 0.]]))


def test_push_pull_and_validation():
    memory = PrototypeMemory(torch.tensor([[1., 0.], [0., 1.], [0., 0.]]))
    before = memory.memory.clone(); changed = memory.update(torch.tensor([[0., 1.]]), torch.tensor([0]), .5)
    assert changed == 1 and torch.allclose(memory.memory[0], before[0] + torch.tensor([0., .5])) and torch.allclose(memory.memory[1], before[1] - torch.tensor([0., .5]))
    after = memory.memory.clone(); assert memory.update(torch.tensor([[1., 0.]]), torch.tensor([0]), .5) == 0 and torch.equal(memory.memory, after)
    with pytest.raises(ValueError): memory.update(torch.ones((1, 2)), torch.tensor([3]), .5)


def test_update_repredicts_each_sample_after_prior_update():
    """The second sample must not use a prediction made before the first update."""
    memory = PrototypeMemory(torch.tensor([[1., 0.], [0., 1.]]))
    encoded = torch.tensor([[-1., 1.], [.2, .8]])
    labels = torch.tensor([0, 0])
    # The first update swaps the preferred directions; the second sample is
    # then correct. A stale batch prediction would incorrectly update twice.
    assert memory.update(encoded, labels, 1.0) == 1
    assert torch.allclose(memory.memory, torch.tensor([[0., 1.], [1., 0.]]))


def test_dot_prediction_uses_prototype_magnitude():
    memory = PrototypeMemory(torch.tensor([[2., 0.], [0., 1.]]))
    query = torch.tensor([[.5, .8]])
    assert memory.predict(query, "cosine").item() == 1
    assert memory.predict(query, "dot").item() == 0


def test_hdzoo_batch_update_aggregates_stale_batch_predictions():
    memory = PrototypeMemory(torch.tensor([[1., 0.], [0., 1.]]))
    encoded = torch.tensor([[-1., 1.], [.2, .8]])
    labels = torch.tensor([0, 0])
    assert memory.update_hdzoo_batch(encoded, labels, 1.0, "dot") == 2
    assert torch.allclose(memory.memory, torch.tensor([[.2, 1.8], [.8, -.8]]))


def test_update_can_normalize_only_the_update_hypervector():
    memory = PrototypeMemory(torch.tensor([[1., 0.], [0., 1.]]))
    assert memory.update(torch.tensor([[0., 2.]]), torch.tensor([0]), .5, normalize_hypervectors=True) == 1
    assert torch.allclose(memory.memory, torch.tensor([[1., .5], [0., .5]]))
