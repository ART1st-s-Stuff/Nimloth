import json
from pathlib import Path

import torch

from nimloth.cfm.model import CFMConfig, TokenConditionedFlowUNet
from nimloth.training.reconstruction.qwen_positive_cache import (
    AttentionTokenCompressor,
    AttentionTokenCompressorConfig,
    positive_cache_fingerprint,
)
from nimloth.training.reconstruction.state_to_vision_tokens import (
    StateToVisionTokens,
    VisionTokenAdapterConfig,
    _legacy_key,
    evaluate_adapter,
)


def test_compressor_checkpoint_round_trip(tmp_path: Path) -> None:
    config = AttentionTokenCompressorConfig(
        input_dim=16,
        output_dim=8,
        input_tokens=5,
        num_output_tokens=3,
        depth=1,
        heads=2,
        dropout=0.0,
    )
    model = AttentionTokenCompressor(config)
    (tmp_path / "config.json").write_text(json.dumps(config.__dict__))
    torch.save(model.state_dict(), tmp_path / "compressor.pt")
    loaded = AttentionTokenCompressor.load_checkpoint(tmp_path)
    output = loaded(torch.randn(2, 5, 16))
    assert output.shape == (2, 3, 8)
    assert torch.isfinite(output).all()


def test_state_adapters_support_query_and_projected_inputs() -> None:
    query = StateToVisionTokens(
        VisionTokenAdapterConfig(
            input_tokens=8,
            input_dim=16,
            output_tokens=4,
            output_dim=8,
            depth=1,
            heads=2,
        )
    )
    projected = StateToVisionTokens(
        VisionTokenAdapterConfig(
            input_tokens=1,
            input_dim=12,
            output_tokens=4,
            output_dim=8,
            depth=1,
            heads=2,
        )
    )
    query_input = torch.randn(3, 8, 16, requires_grad=True)
    projected_input = torch.randn(3, 12, requires_grad=True)
    query_output = query(query_input)
    projected_output = projected(projected_input)
    assert query_output.shape == projected_output.shape == (3, 4, 8)
    (query_output.mean() + projected_output.mean()).backward()
    assert query_input.grad is not None and float(query_input.grad.abs().sum()) > 0
    assert projected_input.grad is not None and float(projected_input.grad.abs().sum()) > 0


def test_adapter_sensitivity_metrics_compare_wrong_states() -> None:
    adapter = StateToVisionTokens(
        VisionTokenAdapterConfig(
            input_tokens=2,
            input_dim=8,
            output_tokens=4,
            output_dim=8,
            depth=1,
            heads=2,
        )
    )
    metrics = evaluate_adapter(
        adapter,
        torch.randn(6, 2, 8),
        torch.randn(6, 4, 8),
        torch.device("cpu"),
        batch_size=3,
        max_items=-1,
    )
    assert metrics["num_items"] == 6
    assert metrics["correct_mse"] > 0
    assert metrics["wrong_mse"] > 0
    assert metrics["delta"] > 0


def test_legacy_cfm_keys_translate_to_current_model() -> None:
    config = CFMConfig(
        image_size=16,
        token_count=4,
        token_dim=8,
        base_channels=4,
        condition_dim=8,
        time_dim=16,
    )
    model = TokenConditionedFlowUNet(config)
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
    for key, value in model.state_dict().items():
        old_key = key
        for current, old in reverse:
            if key.startswith(current):
                old_key = old + key[len(current) :]
                break
        legacy[old_key] = value
    translated = {_legacy_key(key): value for key, value in legacy.items()}
    model.load_state_dict(translated, strict=True)


def test_positive_cache_fingerprint_changes_with_source(tmp_path: Path) -> None:
    qwen = tmp_path / "qwen"
    compressor = tmp_path / "compressor"
    qwen.mkdir()
    compressor.mkdir()
    (qwen / "config.json").write_text("{}")
    (compressor / "config.json").write_text("{}")
    first = positive_cache_fingerprint(
        source_fingerprint="source-a",
        qwen_checkpoint=qwen,
        compressor_checkpoint=compressor,
        max_pixels=100,
        max_items=-1,
    )
    second = positive_cache_fingerprint(
        source_fingerprint="source-b",
        qwen_checkpoint=qwen,
        compressor_checkpoint=compressor,
        max_pixels=100,
        max_items=-1,
    )
    assert first != second
