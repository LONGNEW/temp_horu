import torch

from horu_artifact.methods import hyperfeel


def test_bootstrap_sum_and_zero_delta_personalization():
    encoded = torch.tensor([[1., 0.], [0., 2.], [3., 0.]])
    labels = torch.tensor([0, 1, 0])
    local = hyperfeel.bundled_model(encoded, labels, 2)
    assert torch.equal(local, torch.tensor([[4., 0.], [0., 2.]]))
    central = hyperfeel.sum_deltas([local, local])
    memory = central.clone(); hyperfeel.apply_personalization(memory, torch.zeros_like(memory), torch.zeros(2), .035)
    assert torch.equal(memory, central)


def test_raw_q_sequential_update_delta_and_personalization_weights():
    memory = torch.tensor([[1., 0.], [0., 1.]])
    encoded, labels = torch.tensor([[2., 0.], [1., 0.]]), torch.tensor([1, 1])
    delta, errors, counts, updates = hyperfeel.retrain_samplewise(memory, encoded, labels, .5)
    # The first item is predicted class 0 and changes exactly two rows. The
    # second was initially class 0 too, but observes the changed memory and is
    # now correct, proving current-model (rather than stale batch) prediction.
    assert updates == 1
    assert torch.equal(delta, torch.tensor([[-1., 0.], [1., 0.]]))
    assert torch.equal(memory, torch.tensor([[0., 0.], [1., 1.]]))
    assert torch.equal(counts, torch.tensor([0., 2.]))
    assert torch.equal(errors, torch.tensor([0., 1.]))
    assert torch.equal(hyperfeel.personalization_weights(errors, counts), torch.tensor([0., .5]))
    assert torch.equal(hyperfeel.personalization_weights(torch.tensor([1., 2.]), torch.tensor([1., 0.])), torch.tensor([1., 0.]))
