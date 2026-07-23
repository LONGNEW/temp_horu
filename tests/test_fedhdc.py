import torch
from horu_artifact.methods.fedhdc import bundled_model, train_batches, weighted_aggregate

def test_bundling_and_weighted_normalized_aggregation():
    encoded = torch.tensor([[1., 0.], [2., 0.], [0., 3.]])
    labels = torch.tensor([0, 0, 1])
    local = bundled_model(encoded, labels, 3)
    assert torch.allclose(local, torch.tensor([[1., 0.], [0., 1.], [0., 0.]]))
    global_model = weighted_aggregate([torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([[0., 1.], [1., 0.]])], [3, 1])
    assert torch.allclose(global_model, torch.tensor([[3., 1.], [1., 3.]]) / (10 ** .5))

def test_dot_unit_stale_batch_incomplete_and_changed_rows_only():
    model = torch.tensor([[1., 0.], [0., 1.], [-3., -4.]])
    # Both labels are 0 under fixed predictions; vectors are deliberately non-unit.
    changed = train_batches(model, torch.tensor([[0., 2.], [0., 2.]]), torch.tensor([0, 0]), 0.5)
    assert changed == 2
    assert torch.allclose(torch.linalg.vector_norm(model[0:1], dim=1), torch.ones(1))
    assert torch.equal(model[1], torch.zeros(2))
    assert torch.equal(model[2], torch.tensor([-3., -4.]))
