import pytest
import torch
from PIL import Image

from nimloth.cfm import (
    CFMConfig,
    TokenConditionedFlowUNet,
    conditional_flow_matching_loss,
    sample_euler,
)
from nimloth.rcdm.image_utils import image_to_diffusion_tensor
from nimloth.training.reconstruction.cfm_sft2 import (
    _load_image_uint8,
    initialize_from_legacy_cfm,
    resolve_condition_token_shape,
)
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


def test_query_hidden_manifest_resolves_eight_condition_tokens() -> None:
    manifest = {
        "representation": "qwen_query_hidden",
        "state_shape": [8, 2048],
        "cond_dim": 16384,
    }
    assert resolve_condition_token_shape(manifest) == (8, 2048)
    assert resolve_condition_token_shape({"cond_dim": 1024}) == (1, 1024)
    assert resolve_condition_token_shape(
        {"cond_dim": 8192, "representation": "projected"},
        token_count_override=8,
    ) == (8, 1024)
    with pytest.raises(ValueError, match="does not divide"):
        resolve_condition_token_shape(
            {"cond_dim": 8192}, token_count_override=7
        )


def test_legacy_cfm_initialization_loads_shape_compatible_body(tmp_path) -> None:
    source = TokenConditionedFlowUNet(
        CFMConfig(
            image_size=16,
            token_count=16,
            token_dim=8,
            base_channels=4,
            condition_dim=8,
            time_dim=16,
        )
    )
    reverse = (
        ("condition_mlp.", "cond_mlp."),
        ("block1.", "rb1."),
        ("block2.", "rb2."),
        ("block3.", "rb3."),
        ("attention3.", "attn3."),
        ("block4.", "rb4."),
        ("attention4.", "attn4."),
        ("middle1.", "mid1."),
        ("middle_attention.", "mid_attn."),
        ("middle2.", "mid2."),
        ("up_block3.", "urb3."),
        ("up_attention3.", "uattn3."),
        ("up_block2.", "urb2."),
        ("up_block1.", "urb1."),
    )
    legacy = {}
    for key, value in source.state_dict().items():
        old_key = key
        for current, old in reverse:
            if key.startswith(current):
                old_key = old + key[len(current) :]
                break
        legacy[old_key] = value
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"model": legacy}, checkpoint)
    target = TokenConditionedFlowUNet(
        CFMConfig(
            image_size=16,
            token_count=8,
            token_dim=12,
            base_channels=4,
            condition_dim=8,
            time_dim=16,
        )
    )
    result = initialize_from_legacy_cfm(target, checkpoint)
    assert result["loaded_keys"] > result["skipped_keys"]
    assert "token_norm.weight" in result["skipped"]
    torch.testing.assert_close(target.in_conv.weight, source.in_conv.weight)
