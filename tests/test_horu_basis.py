import torch
from horu_artifact.horu.basis import canonicalize_signs, personal_basis, shared_basis
from horu_artifact.horu.bootstrap import bootstrap_horu, class_prototype


def test_prototype_basis_and_personal_projector():
    encoded = torch.tensor([[1., 0., 0.], [3., 0., 0.], [0., 2., 0.]])
    labels = torch.tensor([0, 0, 2])
    prototype, counts = class_prototype(encoded, labels, 3)
    assert counts.tolist() == [2, 0, 1]
    assert torch.allclose(prototype[0], torch.tensor([1., 0., 0.])) and torch.equal(prototype[1], torch.zeros(3))
    common, global_basis, info = shared_basis([prototype], 1, 1)
    basis = torch.cat([common, global_basis], 1)
    assert torch.allclose(basis.T @ basis, torch.eye(2), atol=1e-6)
    assert info["projector_sha256"]
    residual = prototype - (prototype @ basis) @ basis.T
    personal, personal_info = personal_basis(residual, 1, "reduced_svd")
    assert torch.allclose(personal.T @ personal, torch.ones((1, 1)), atol=1e-6)
    assert personal_info["personal_projector_sha256"]


def test_sign_canonicalization_tie_is_lowest_index():
    value = canonicalize_signs(torch.tensor([[-2.], [2.], [1.]]))
    assert value[0, 0] > 0


def test_full_svd_labels_zero_singular_completion():
    residual = torch.zeros((6, 16))
    basis, info = personal_basis(residual, 10, "full_svd")
    assert basis.shape == (16, 10)
    assert info["zero_singular_directions_selected"] == 10
    assert info["completion_provenance"] == "USER_SPECIFIED_NUMERICAL_COMPLETION"


def test_bootstrap_uses_global_consensus_and_canonical_personal_residual():
    clients = {
        0: {
            "train_h": torch.tensor([[1., 0., 0.], [0., 1., 0.]]),
            "train_y": torch.tensor([0, 1]),
            "test_h": torch.tensor([[1., 0., 0.]]),
            "test_y": torch.tensor([0]),
        },
        1: {
            "train_h": torch.tensor([[0., 0., 1.], [1., 1., 0.]]),
            "train_y": torch.tensor([0, 1]),
            "test_h": torch.tensor([[0., 0., 1.]]),
            "test_y": torch.tensor([0]),
        },
    }
    states, common, global_basis, _, _, _ = bootstrap_horu(
        clients, common_rank=1, global_rank=1, personal_rank=1, personal_policy="reduced_svd", num_classes=2
    )
    prototypes = {cid: class_prototype(client["train_h"], client["train_y"], 2)[0] for cid, client in clients.items()}
    expected_global = torch.stack([prototype @ global_basis for prototype in prototypes.values()]).mean(dim=0)
    for cid, state in states.items():
        assert torch.allclose(state.global_coefficients, expected_global)
        expected_residual = prototypes[cid] - (
            (prototypes[cid] @ common) @ common.T
            + expected_global @ global_basis.T
        )
        assert torch.allclose(state.personal, expected_residual @ state.personal_basis)
        assert torch.count_nonzero(state.delta) == 0
