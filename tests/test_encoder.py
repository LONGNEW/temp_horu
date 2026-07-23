import pytest
import torch

from horu_artifact.hdc.encoder import NonlinearEncoder, make_projection


def test_projection_hdzoo_order_and_encoding():
    projection = make_projection(3, 5, 7)
    generator = torch.Generator(device="cpu").manual_seed(7)
    raw = torch.empty((5, 3), dtype=torch.float32).normal_(generator=generator)
    assert projection.shape == (3, 5) and projection.dtype == torch.float32 and projection.device.type == "cpu"
    assert torch.equal(projection, raw.T.contiguous()) and not torch.equal(projection, make_projection(3, 5, 8))
    features = torch.tensor([[1., -2., .5], [0., 1., 2.]])
    assert torch.allclose(NonlinearEncoder(projection).encode(features), torch.cos(features @ projection))


def test_encoder_validation_and_batches():
    projection = make_projection(2, 4, 0); encoder = NonlinearEncoder(projection); values = torch.tensor([[.2, .3], [.4, .5]])
    assert torch.allclose(encoder.encode(values), torch.cat([encoder.encode(values[:1]), encoder.encode(values[1:])]))
    with pytest.raises(ValueError): encoder.encode(torch.empty((0, 2)))
    with pytest.raises(ValueError): encoder.encode(torch.ones((1, 3)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_parity():
    projection = make_projection(2, 8, 0); values = torch.rand(4, 2)
    assert torch.allclose(NonlinearEncoder(projection).encode(values), NonlinearEncoder(projection.cuda()).encode(values.cuda()).cpu(), rtol=1e-4, atol=1e-5)
