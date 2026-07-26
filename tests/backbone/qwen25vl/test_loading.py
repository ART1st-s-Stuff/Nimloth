"""Qwen 配置兼容解析测试。"""

from types import SimpleNamespace

import pytest

from nimloth.backbone.qwen25vl import loading
from nimloth.backbone.qwen25vl.loading import (
    load_qwen_processor,
    qwen_hidden_size,
)


def test_hidden_size_supports_top_level_config() -> None:
    assert qwen_hidden_size(SimpleNamespace(hidden_size=2048)) == 2048


def test_hidden_size_supports_nested_text_config() -> None:
    config = SimpleNamespace(
        text_config=SimpleNamespace(hidden_size=3584),
    )
    assert qwen_hidden_size(config) == 3584


def test_hidden_size_rejects_missing_value() -> None:
    with pytest.raises(ValueError, match="hidden_size"):
        qwen_hidden_size(SimpleNamespace())


def test_processor_preserves_checkpoint_pixel_bounds_by_default(monkeypatch) -> None:
    processor = SimpleNamespace(
        image_processor=SimpleNamespace(min_pixels=3136, max_pixels=100352),
        tokenizer=SimpleNamespace(),
    )
    monkeypatch.setattr(
        loading.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: processor,
    )
    monkeypatch.setattr(loading, "add_special_tokens", lambda *args, **kwargs: 0)
    monkeypatch.setattr(loading, "special_token_ids", lambda *args, **kwargs: {})

    loaded = load_qwen_processor("/model", max_pixels=None)

    assert loaded.processor.image_processor.min_pixels == 3136
    assert loaded.processor.image_processor.max_pixels == 100352


def test_processor_applies_explicit_max_pixel_override(monkeypatch) -> None:
    processor = SimpleNamespace(
        image_processor=SimpleNamespace(min_pixels=3136, max_pixels=100352),
        tokenizer=SimpleNamespace(),
    )
    monkeypatch.setattr(
        loading.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: processor,
    )
    monkeypatch.setattr(loading, "add_special_tokens", lambda *args, **kwargs: 0)
    monkeypatch.setattr(loading, "special_token_ids", lambda *args, **kwargs: {})

    loaded = load_qwen_processor("/model", max_pixels=50176)

    assert loaded.processor.image_processor.min_pixels == 3136
    assert loaded.processor.image_processor.max_pixels == 50176
