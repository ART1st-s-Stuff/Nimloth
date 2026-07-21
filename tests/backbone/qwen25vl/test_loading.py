"""Qwen 配置兼容解析测试。"""

from types import SimpleNamespace

import pytest

from nimloth.backbone.qwen25vl.loading import qwen_hidden_size


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
