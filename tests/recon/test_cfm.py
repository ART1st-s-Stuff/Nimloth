import pytest
import torch
from PIL import Image

from nimloth.recon.cfm import (
    CFMConfig,
    SpatialCLSCFMConfig,
    SpatialCLSConditionedFlowUNet,
    TokenConditionedFlowUNet,
    conditional_flow_matching_loss,
    sample_euler,
    sample_euler_cfg,
    spatial_cls_condition_sensitivity,
    spatial_cls_condition_variants,
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


def _tiny_spatial_cls_model() -> SpatialCLSConditionedFlowUNet:
    return SpatialCLSConditionedFlowUNet(
        SpatialCLSCFMConfig(
            image_size=16,
            spatial_grid_size=8,
            spatial_token_count=64,
            global_token_count=1,
            token_dim=4,
            base_channels=4,
            condition_dim=8,
            time_dim=16,
        )
    )


def test_spatial_cls_cfm_keeps_cls_outside_the_spatial_grid() -> None:
    model = _tiny_spatial_cls_model()
    state = torch.arange(2 * 65 * 4, dtype=torch.float32).reshape(2, 65, 4)
    spatial, cls = model.split_condition(state.flatten(1))
    assert spatial.shape == (2, 64, 4)
    assert cls.shape == (2, 4)
    torch.testing.assert_close(spatial, state[:, :64])
    torch.testing.assert_close(cls, state[:, 64])
    grid = model.reshape_spatial_condition(spatial)
    assert grid.shape == (2, 4, 8, 8)
    assert grid[0, 3, 2, 5] == state[0, 2 * 8 + 5, 3]
    with pytest.raises(ValueError, match=r"K64\+K1"):
        model.split_condition(state[:, :64].flatten(1))


def test_spatial_cls_cfm_routes_cls_gradient_without_changing_spatial_tokens() -> None:
    torch.manual_seed(19)
    model = _tiny_spatial_cls_model()
    image = torch.randn(2, 3, 16, 16)
    time = torch.rand(2)
    state = torch.randn(2, 65, 4, requires_grad=True)
    output = model(image, time, state.flatten(1))
    assert output.shape == image.shape
    output.square().mean().backward()
    assert state.grad is not None
    assert torch.isfinite(state.grad).all()
    assert float(state.grad[:, 64].abs().sum()) > 0


def test_spatial_cls_variants_only_change_the_final_slot() -> None:
    state = torch.arange(3 * 65 * 4, dtype=torch.float32).reshape(3, 65, 4)
    variants = spatial_cls_condition_variants(
        state,
        spatial_token_count=64,
        global_token_count=1,
        token_dim=4,
        shuffle_indices=torch.tensor([1, 2, 0]),
    )
    for value in variants.values():
        torch.testing.assert_close(value[:, :64], state[:, :64])
    torch.testing.assert_close(variants["correct"][:, 64], state[:, 64])
    torch.testing.assert_close(variants["zero_cls"][:, 64], torch.zeros_like(state[:, 64]))
    torch.testing.assert_close(variants["shuffled_cls"][:, 64], state[[1, 2, 0], 64])
    with pytest.raises(ValueError, match="must not keep"):
        spatial_cls_condition_variants(
            state,
            spatial_token_count=64,
            token_dim=4,
            shuffle_indices=torch.arange(3),
        )


def test_spatial_cls_condition_sensitivity_uses_matched_noise_and_time() -> None:
    torch.manual_seed(23)
    model = _tiny_spatial_cls_model()
    state = torch.randn(3, 65 * 4)
    images = torch.randint(0, 256, (3, 3, 16, 16), dtype=torch.uint8)
    first = spatial_cls_condition_sensitivity(
        model,
        state,
        images,
        torch.device("cpu"),
        batch_size=2,
        seed=29,
    )
    second = spatial_cls_condition_sensitivity(
        model,
        state,
        images,
        torch.device("cpu"),
        batch_size=2,
        seed=29,
    )
    assert first["num_items"] == 3
    for key in first:
        assert first[key] == pytest.approx(second[key], rel=1e-5, abs=1e-6)
