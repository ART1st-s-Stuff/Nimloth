"""Qwen processor、token 与 hidden-size 加载工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoProcessor

from nimloth.latent import add_special_tokens, special_token_ids


@dataclass(frozen=True)
class QwenProcessorBundle:
    processor: Any
    token_id_map: dict[str, int]
    added_special_token_count: int


def load_qwen_processor(
    model_path: str | Path,
    *,
    max_pixels: int | None,
    latent_token_count: int = 1,
) -> QwenProcessorBundle:
    """加载 processor，并建立当前 latent/action token 契约。

    ``max_pixels=None`` 保留 checkpoint 自带的图像处理语义。显式值只用于
    调试或经批准的分辨率覆盖，调用方还必须把同一覆盖传给推理后端。
    """

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if max_pixels is not None:
        if max_pixels <= 0:
            raise ValueError("max_pixels must be positive")
        processor.image_processor.max_pixels = int(max_pixels)
    added_count = add_special_tokens(
        processor.tokenizer,
        latent_token_count=latent_token_count,
    )
    return QwenProcessorBundle(
        processor=processor,
        token_id_map=special_token_ids(
            processor.tokenizer,
            latent_token_count=latent_token_count,
        ),
        added_special_token_count=added_count,
    )


def qwen_processor_pixel_bounds(processor: Any) -> tuple[int, int]:
    """返回实际生效的 Qwen image processor 像素上下界。"""

    image_processor = processor.image_processor
    return int(image_processor.min_pixels), int(image_processor.max_pixels)


def qwen_hidden_size(model_config: Any) -> int:
    """兼容顶层或 text_config 保存 hidden size 的 Qwen 配置。"""

    for config in (model_config, getattr(model_config, "text_config", None)):
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is not None:
            result = int(hidden_size)
            if result > 0:
                return result
    raise ValueError("Qwen config does not expose a positive hidden_size")
