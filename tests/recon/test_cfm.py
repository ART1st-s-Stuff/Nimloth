import pytest
import torch
from PIL import Image

from nimloth.recon.cfm import (
    CFMConfig,
    SpatialConditionedFlowUNet,
    TokenConditionedFlowUNet,
    conditional_flow_matching_loss,
    sample_euler,
    sample_euler_cfg,
)
from nimloth.recon.rcdm.image_utils import image_to_diffusion_tensor
from nimloth.training.reconstruction.cfm_sft2 import _load_image_uint8
from nimloth.training.reconstruction.residual_cfm_sft2 import biased_flow_loss


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


def test_residual_cfm_accepts_spatial_scaffold_channels() -> None:
    model = TokenConditionedFlowUNet(
        CFMConfig(
            image_size=16,
            token_count=1,
            token_dim=12,
            base_channels=4,
            condition_dim=8,
            time_dim=16,
            input_channels=6,
            output_channels=3,
        )
    )
    scaffold = torch.randn(2, 3, 16, 16).clamp(-1, 1)
    target = torch.randn(2, 3, 16, 16).clamp(-1, 1)
    condition = torch.randn(2, 12)
    loss, parts = biased_flow_loss(
        model,
        scaffold,
        target,
        condition,
        reconstruction_weight=0.5,
    )
    assert torch.isfinite(loss)
    assert set(parts) == {"velocity_mse", "reconstruction_l1", "loss"}
    loss.backward()


def test_cfm_rejects_wrong_condition_width() -> None:
    model = _tiny_model()
    with pytest.raises(ValueError, match="expected condition shape"):
        model(torch.randn(2, 3, 16, 16), torch.rand(2), torch.randn(2, 11))


def test_cfm_uint8_loader_matches_existing_image_normalization(tmp_path) -> None:
    values = torch.arange(3 * 16 * 16, dtype=torch.int64).remainder(256).byte()
    image = values.view(3, 16, 16).permute(1, 2, 0).numpy()
    path = tmp_path / "image.png"
    Image.fromarray(image, mode="RGB").save(path)
    loaded = _load_image_uint8(path, 16)
    expected = image_to_diffusion_tensor(path, image_size=16)
    torch.testing.assert_close(loaded.float().div(127.5).sub(1.0), expected)


def test_cfm_cfg_scale_one_matches_conditional_sampling() -> None:
    torch.manual_seed(13)
    model = _tiny_model().eval()
    states = torch.randn(2, 12)
    noise = torch.randn(2, 3, 16, 16)
    expected = sample_euler(
        model, states, noise, steps=2, device=torch.device("cpu"), chunk_size=1
    )
    actual = sample_euler_cfg(
        model,
        states,
        noise,
        steps=2,
        cfg_scale=1.0,
        device=torch.device("cpu"),
        chunk_size=1,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=3e-6)


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
    torch.testing.assert_close(first, second, rtol=1e-4, atol=3e-6)


def _tiny_k16_config() -> CFMConfig:
    return CFMConfig(
        image_size=16,
        token_count=16,
        token_dim=12,
        base_channels=4,
        condition_dim=8,
        time_dim=16,
    )


def test_legacy_token_condition_is_permutation_offset_and_scale_invariant() -> None:
    torch.manual_seed(19)
    model = TokenConditionedFlowUNet(_tiny_k16_config()).eval()
    assert model.decoder_family == "token_set_v1"
    image = torch.randn(2, 3, 16, 16)
    time = torch.tensor([0.1, 0.7])
    tokens = torch.randn(2, 16, 12)
    permutation = torch.tensor([5, 1, 15, 0, 8, 3, 12, 7, 4, 11, 2, 14, 9, 6, 13, 10])
    offsets = torch.arange(16, dtype=tokens.dtype).view(1, 16, 1)
    positive_scales = torch.linspace(0.5, 2.0, 16).view(1, 16, 1)

    expected = model(image, time, tokens.flatten(1))
    permuted = model(image, time, tokens[:, permutation].flatten(1))
    shifted = model(image, time, (tokens + offsets).flatten(1))
    scaled = model(image, time, (tokens * positive_scales).flatten(1))

    torch.testing.assert_close(permuted, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(shifted, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(scaled, expected, rtol=1e-5, atol=1e-6)


def test_spatial_condition_uses_row_major_grid_position_and_relative_scale() -> None:
    torch.manual_seed(23)
    model = SpatialConditionedFlowUNet(_tiny_k16_config()).eval()
    assert model.decoder_family == "spatial_grid_v1"
    tokens = torch.arange(2 * 16 * 12, dtype=torch.float32).reshape(2, 16, 12)
    grid = model.reshape_condition(tokens.flatten(1))
    assert grid.shape == (2, 12, 4, 4)
    torch.testing.assert_close(grid[:, :, 0, 0], tokens[:, 0, :])
    torch.testing.assert_close(grid[:, :, 0, 1], tokens[:, 1, :])
    torch.testing.assert_close(grid[:, :, 3, 3], tokens[:, 15, :])

    image = torch.randn(2, 3, 16, 16)
    time = torch.tensor([0.2, 0.8])
    random_tokens = torch.randn(2, 16, 12)
    permutation = torch.arange(15, -1, -1)
    original = model(image, time, random_tokens.flatten(1))
    permuted = model(image, time, random_tokens[:, permutation].flatten(1))
    slot_scales = torch.linspace(0.5, 2.0, 16).view(1, 16, 1)
    rescaled = model(image, time, (random_tokens * slot_scales).flatten(1))

    assert not torch.allclose(permuted, original, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(rescaled, original, rtol=1e-5, atol=1e-6)


def test_spatial_condition_has_finite_decoder_only_gradients() -> None:
    torch.manual_seed(29)
    model = SpatialConditionedFlowUNet(_tiny_k16_config())
    images = torch.randn(2, 3, 16, 16).clamp(-1, 1)
    condition = torch.randn(2, 16 * 12)
    loss = conditional_flow_matching_loss(model, images, condition)
    assert torch.isfinite(loss)
    loss.backward()
    assert condition.grad is None
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_spatial_condition_rejects_non_k16_or_wrong_flat_width() -> None:
    with pytest.raises(ValueError, match="K16|token_count"):
        SpatialConditionedFlowUNet(
            CFMConfig(
                image_size=16,
                token_count=4,
                token_dim=12,
                base_channels=4,
                condition_dim=8,
                time_dim=16,
            )
        )
    model = SpatialConditionedFlowUNet(_tiny_k16_config())
    with pytest.raises(ValueError, match="expected condition shape"):
        model(
            torch.randn(2, 3, 16, 16),
            torch.rand(2),
            torch.randn(2, 16 * 12 - 1),
        )
