import pytest
import torch

from nimloth.cfm import (
    CFMConfig,
    TokenConditionedFlowUNet,
    conditional_flow_matching_loss,
    sample_euler,
)


def _tiny_model() -> TokenConditionedFlowUNet:
    return TokenConditionedFlowUNet(
        CFMConfig(
            image_size=16,
            token_count=1,
            token_dim=12,
            base_channels=4,
            condition_dim=8,
            time_dim=16,
        )
    )


def test_cfm_config_rejects_non_divisible_image_size() -> None:
    with pytest.raises(ValueError, match="divisible by 8"):
        CFMConfig(image_size=18)


def test_cfm_forward_and_loss_are_finite() -> None:
    torch.manual_seed(7)
    model = _tiny_model()
    images = torch.randn(2, 3, 16, 16).clamp(-1, 1)
    states = torch.randn(2, 12)
    time = torch.rand(2)
    velocity = model(images, time, states)
    assert velocity.shape == images.shape

    loss = conditional_flow_matching_loss(model, images, states)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_cfm_rejects_wrong_condition_width() -> None:
    model = _tiny_model()
    with pytest.raises(ValueError, match="expected condition shape"):
        model(torch.randn(2, 3, 16, 16), torch.rand(2), torch.randn(2, 11))


def test_cfm_euler_sampling_is_deterministic() -> None:
    torch.manual_seed(11)
    model = _tiny_model().eval()
    states = torch.randn(2, 12)
    noise = torch.randn(2, 3, 16, 16)
    first = sample_euler(
        model, states, noise, steps=2, device=torch.device("cpu"), chunk_size=1
    )
    second = sample_euler(
        model, states, noise, steps=2, device=torch.device("cpu"), chunk_size=2
    )
    assert first.shape == noise.shape
    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)
