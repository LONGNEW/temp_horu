import torch
from horu_artifact.horu.inference import coefficient_gram, predict, scores
from horu_artifact.horu.state import ClientBootstrapState
from horu_artifact.horu.training import error_statistics, train_client


def _state():
    state = ClientBootstrapState(1, torch.zeros(3, 3), torch.tensor([1, 1, 1]), torch.tensor([[0.], [0.], [0.]]), torch.tensor([[0.], [0.], [0.]]), torch.tensor([[0.], [0.], [0.]]), torch.tensor([[0., 0.], [0., 1.], [1., 0.]]), torch.tensor([[1.], [2.], [3.]]), {"z_c": torch.tensor([[1.], [0.]]), "z_g": torch.tensor([[0.], [0.]]), "z_p": torch.tensor([[0.], [1.]])}, {"z_c": torch.tensor([[1.]]), "z_g": torch.tensor([[0.]]), "z_p": torch.tensor([[0.]])}, torch.tensor([0, 1]), torch.tensor([0]))
    return state


def test_coefficient_score_is_direct_dot_product_and_ignores_gram():
    bc = torch.tensor([[1.], [0.], [0.]])
    bg = torch.tensor([[.2], [1.], [0.]])
    bp = torch.tensor([[.1], [.3], [1.]])
    gram = coefficient_gram(bc, bg, bp); zc, zg, zp = torch.tensor([.4]), torch.tensor([-.2]), torch.tensor([.7])
    common, global_c, delta, personal = torch.tensor([[1.], [-1.]]), torch.tensor([[.3], [.2]]), torch.tensor([[.1], [.4]]), torch.tensor([[.2], [-.5]])
    actual = scores(zc, zg, zp, common, global_c, delta, personal, gram)
    q = torch.cat([zc, zg, zp])
    rows = torch.cat([common + delta, global_c, personal], dim=1)
    expected = rows @ q
    assert torch.allclose(actual, expected, atol=1e-6)


def test_misclassification_adds_to_only_true_and_predicted_rows():
    state = _state(); gram = torch.eye(3)
    state.train_cache = {"z_c": torch.tensor([[1.]]), "z_g": torch.tensor([[1.]]), "z_p": torch.tensor([[1.]])}
    predicted = predict(state.train_cache["z_c"][0], state.train_cache["z_g"][0], state.train_cache["z_p"][0], state.common, state.global_coefficients, state.delta, state.personal, gram)
    target = (predicted + 1) % 3; untouched = (predicted + 2) % 3
    state.train_labels = torch.tensor([target])
    before = [state.common.clone(), state.global_coefficients.clone(), state.delta.clone(), state.personal.clone()]
    updates, _ = train_client(state, 1, 16, 1.0, 1.0, gram, 0, 0)
    assert updates == 1
    after = [state.common, state.global_coefficients, state.delta, state.personal]
    query_parts = (torch.tensor([1.]), torch.tensor([1.]), torch.tensor([1.]), torch.tensor([1.]))
    for old, new, query in zip(before, after, query_parts):
        expected = old.clone()
        expected[target].add_(query)
        expected[predicted].sub_(query)
        assert torch.equal(new, expected)
        assert torch.equal(new[untouched], old[untouched])


def test_batch_accumulates_predictions_before_additive_update():
    # Both samples initially predict row 0. Batch accumulation records both
    # errors before applying one aggregated update.
    state = ClientBootstrapState(1, torch.zeros(3, 2), torch.tensor([0, 2, 0]), torch.tensor([[0., 1.], [1., 0.], [-1., 0.]]), torch.empty(3, 0), torch.zeros(3, 2), torch.empty(2, 0), torch.empty(3, 0), {"z_c": torch.tensor([[0., 1.], [0., 1.]]), "z_g": torch.empty(2, 0), "z_p": torch.empty(2, 0)}, {}, torch.tensor([1, 1]), torch.tensor([], dtype=torch.long))
    updates, _ = train_client(state, 1, 16, 1.0, 1.0, torch.eye(2), 0, 0)
    assert updates == 2
    assert torch.equal(state.common, torch.tensor([[0., -1.], [1., 2.], [-1., 0.]]))
    assert torch.equal(state.delta, torch.tensor([[0., -2.], [0., 2.], [0., 0.]]))


def test_final_error_statistics_matches_manual_counts():
    state = _state(); gram = torch.eye(3)
    counts, errors, ratios, prediction_ms, statistics_ms = error_statistics(state, gram, 3)
    assert counts.tolist() == [1, 1, 0]
    assert errors.tolist() == [0, 1, 0]
    assert torch.allclose(ratios, torch.tensor([0., 1., 0.]))
    assert prediction_ms >= 0 and statistics_ms >= 0
