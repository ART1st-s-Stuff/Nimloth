"""Normalize Qwen2.5-VL model_type for SGLang mrope on saved HF checkpoints.

Saved checkpoints report AutoConfig.model_type == "qwen2_5_vl_text", but
SGLang MRotaryEmbedding.get_rope_index only handles "qwen2_5_vl". Hub
Qwen/Qwen2.5-VL-3B-Instruct uses "qwen2_5_vl", which is why baseline
rollouts worked with base MODEL_PATH while SFT HF paths failed.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

_ORIGINAL_GET_ROPE_INDEX = None


def _normalize_model_type(model_type: str) -> str:
    if model_type in {"qwen2_5_vl", "qwen2_5_vl_text"} or (
        model_type and model_type.startswith("qwen2_5_vl")
    ):
        return "qwen2_5_vl"
    return model_type


def _patched_get_rope_index(
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    model_type: str,
    tokens_per_second: Optional[int] = None,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _ORIGINAL_GET_ROPE_INDEX(
        spatial_merge_size=spatial_merge_size,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        vision_start_token_id=vision_start_token_id,
        model_type=_normalize_model_type(model_type),
        tokens_per_second=tokens_per_second,
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        **kwargs,
    )


def apply() -> None:
    global _ORIGINAL_GET_ROPE_INDEX
    from sglang.srt.layers.rotary_embedding import MRotaryEmbedding

    if getattr(MRotaryEmbedding.get_rope_index, "__nimloth_patched__", False):
        return
    if _ORIGINAL_GET_ROPE_INDEX is None:
        _ORIGINAL_GET_ROPE_INDEX = MRotaryEmbedding.get_rope_index
    _patched_get_rope_index.__nimloth_patched__ = True  # type: ignore[attr-defined]
    MRotaryEmbedding.get_rope_index = staticmethod(_patched_get_rope_index)


apply()
