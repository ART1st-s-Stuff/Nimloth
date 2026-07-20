"""Encode SFT2 transition rows for the on-disk cache."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoProcessor

from nimloth.backbone.qwen25vl.batch import (
    assistant_char_spans,
    encode_qwen_item,
    labels_for_text_rows,
    render_messages,
)

def encode_transition_item(
    item: dict[str, Any],
    processor: AutoProcessor,
    max_length: int,
    *,
    latent_token_count: int = 1,
    mask_latent_query_labels: bool = True,
) -> dict[str, Any]:
    current_enc = encode_qwen_item(
        item["messages"],
        processor,
        max_length,
        include_labels=True,
        latent_token_count=latent_token_count,
        mask_latent_query_labels=mask_latent_query_labels,
    )
    next_enc = None
    if item.get("next_messages"):
        next_enc = encode_qwen_item(
            item["next_messages"],
            processor,
            max_length,
            include_labels=False,
            latent_token_count=latent_token_count,
            mask_latent_query_labels=mask_latent_query_labels,
        )
    return {
        "id": item["id"],
        "record_id": item.get("record_id", ""),
        "step_index": item.get("step_index", 0),
        "action_index": item["action_index"],
        "action_value_target": item["action_value_target"],
        "success": item["success"],
        "current_enc": current_enc,
        "next_enc": next_enc,
    }


def _expand_qwen_image_tokens(
    text: str,
    image_grid_thw: torch.Tensor,
    processor: AutoProcessor,
) -> str:
    """Expand Qwen image placeholders exactly as Qwen2_5_VLProcessor.__call__."""

    image_token = str(processor.image_token)
    merge_length = int(processor.image_processor.merge_size) ** 2
    expanded = text
    for grid in image_grid_thw:
        if image_token not in expanded:
            raise ValueError("fewer image placeholders than cached image grids")
        token_count = int(grid.prod().item()) // merge_length
        expanded = expanded.replace(image_token, "<|placeholder|>" * token_count, 1)
    if image_token in expanded:
        raise ValueError("more image placeholders than cached image grids")
    return expanded.replace("<|placeholder|>", image_token)


def encode_qwen_item_from_image_grids(
    messages: list[dict[str, Any]],
    image_grid_thw: torch.Tensor,
    processor: AutoProcessor,
    max_length: int,
    *,
    include_labels: bool = True,
    latent_token_count: int = 1,
    mask_latent_query_labels: bool = True,
) -> dict[str, torch.Tensor]:
    """Encode text/labels without reprocessing pixels.

    The image grids come from a per-image cache. Image-token expansion mirrors
    the Hugging Face processor, so input ids are identical to online encoding.
    """

    text = render_messages(
        messages,
        processor,
        add_generation_prompt=False,
        latent_token_count=latent_token_count,
    )
    grids = image_grid_thw.to(dtype=torch.long, device="cpu").reshape(-1, 3).contiguous()
    expanded_text = _expand_qwen_image_tokens(text, grids, processor)
    enc = processor.tokenizer(
        [expanded_text],
        padding=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    out: dict[str, torch.Tensor] = {
        key: value.squeeze(0).contiguous()
        for key, value in enc.items()
        if isinstance(value, torch.Tensor)
    }
    if grids.numel():
        out["image_grid_thw"] = grids
    if include_labels:
        labels = labels_for_text_rows(
            processor,
            enc["input_ids"],
            [text],
            [assistant_char_spans(messages, processor, latent_token_count=latent_token_count)],
            max_length,
            latent_token_count=latent_token_count,
            mask_latent_query_labels=mask_latent_query_labels,
        )
        out["labels"] = labels.squeeze(0).contiguous()
    return out
