"""SFT2、RL 与独立 rollout 共用的 Qwen 加载基础工具。"""

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
    max_pixels: int,
    latent_token_count: int = 1,
) -> QwenProcessorBundle:
    """加载 processor，并建立当前 latent/action token 契约。"""

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = max_pixels
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


def qwen_hidden_size(model_config: Any) -> int:
    """兼容顶层或 text_config 保存 hidden size 的 Qwen 配置。"""

    for config in (model_config, getattr(model_config, "text_config", None)):
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is not None:
            result = int(hidden_size)
            if result > 0:
                return result
    raise ValueError("Qwen config does not expose a positive hidden_size")
