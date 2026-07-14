import torch

from nimloth.training.reconstruction.deterministic_state_decoder import reconstruction_loss
from nimloth.wm.reconstruction import WMImageDecoder, WMImageDecoderConfig


def test_deterministic_decoder_loss_is_finite_and_condition_dependent() -> None:
    torch.manual_seed(5)
    decoder = WMImageDecoder(
        WMImageDecoderConfig(
            emb_dim=8,
            image_size=16,
            patch_size=4,
            hidden_dim=16,
            depth=1,
            heads=4,
            mlp_ratio=2,
        )
    )
    states = torch.randn(2, 8)
    target = torch.rand(2, 3, 16, 16)
    prediction = decoder(states)
    wrong_prediction = decoder(states.flip(0))
    loss, l1, mse = reconstruction_loss(prediction, target)
    assert prediction.shape == target.shape
    assert all(torch.isfinite(value) for value in (loss, l1, mse))
    assert not torch.equal(prediction, wrong_prediction)
    loss.backward()
    assert any(parameter.grad is not None for parameter in decoder.parameters())
