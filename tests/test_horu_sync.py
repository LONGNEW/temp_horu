import torch
from horu_artifact.horu.state import ClientBootstrapState
from horu_artifact.horu.synchronization import aggregate_shared, absorb_shared, follow_ratio


def _state(cid, counts, common, global_c):
    return ClientBootstrapState(cid, torch.zeros(3, 2), torch.tensor(counts), torch.tensor(common, dtype=torch.float32), torch.tensor(global_c, dtype=torch.float32), torch.ones(3, 1), torch.eye(2, 1), torch.ones(3, 1), {}, {}, torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))


def test_weighted_shared_aggregate_zero_row_and_absorption_keeps_local_state():
    one = _state(1, [2, 0, 1], [[1.], [8.], [3.]], [[2.], [7.], [6.]])
    two = _state(2, [1, 0, 3], [[4.], [9.], [5.]], [[8.], [9.], [10.]])
    common, global_c, _ = aggregate_shared([one, two])
    assert torch.allclose(common[:, 0], torch.tensor([2., 0., 4.5]))
    assert torch.allclose(global_c[:, 0], torch.tensor([4., 0., 9.]))
    before = (one.delta.clone(), one.personal.clone(), one.personal_basis.clone(), one.common.clone())
    absorb_shared(one, common, global_c, torch.tensor([2, 0, 1]), torch.tensor([1, 0, 0]), .2)
    assert torch.allclose(one.common[0], torch.tensor([1.1]))
    assert torch.allclose(one.common[2], torch.tensor([3.03]))
    assert torch.equal(one.delta, before[0]) and torch.equal(one.personal, before[1]) and torch.equal(one.personal_basis, before[2])


def test_follow_ratio_matches_canonical_rowgate_and_zeros_empty_classes():
    actual = follow_ratio(
        torch.tensor([10, 10, 0]),
        torch.tensor([0, 10, 0]),
        gate_alpha=1.0,
        gate_min=0.1,
        gate_max=0.9,
    )
    assert torch.allclose(actual, torch.tensor([0.1, 0.9, 0.0]))
