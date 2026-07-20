"""Normalize online and cached transitions into the shared SFT2 batch protocol."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from nimloth.backbone.qwen25vl.batch import message_cache_key


def collate_cached_encodings(batch: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    if len(batch) == 1:
        return {k: (v.unsqueeze(0) if v.ndim == 1 else v) for k, v in batch[0].items()}
    out: dict[str, torch.Tensor] = {}
    if "input_ids" in batch[0]:
        out["input_ids"] = pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=pad_token_id,
        )
    if "attention_mask" in batch[0]:
        out["attention_mask"] = pad_sequence(
            [item["attention_mask"] for item in batch],
            batch_first=True,
            padding_value=0,
        )
    if "labels" in batch[0]:
        out["labels"] = pad_sequence(
            [item["labels"] for item in batch],
            batch_first=True,
            padding_value=-100,
        )
    for key in ("pixel_values", "image_grid_thw"):
        if key in batch[0]:
            tensors = []
            for item in batch:
                tensor = item[key]
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)
                tensors.append(tensor)
            out[key] = torch.cat(tensors, dim=0)
    return out


def _collate_next_encoding_bundle(
    items: list[dict[str, Any]],
    next_rows: list[dict[str, torch.Tensor] | None],
    *,
    pad_token_id: int,
) -> dict[str, Any] | None:
    unique_rows: list[dict[str, torch.Tensor]] = []
    unique_keys: list[str] = []
    seen: set[str] = set()
    for item, row in zip(items, next_rows, strict=True):
        messages = item.get("next_messages")
        if not messages or row is None:
            continue
        key = message_cache_key(messages)
        if key in seen:
            continue
        seen.add(key)
        unique_keys.append(key)
        unique_rows.append(row)
    if not unique_rows:
        return None
    return {
        "keys": unique_keys,
        "enc": collate_cached_encodings(unique_rows, pad_token_id),
    }


def collate_cached_transition_batch(
    batch: list[dict[str, Any]],
    *,
    pad_token_id: int,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    current_rows: list[dict[str, torch.Tensor]] = []
    next_rows: list[dict[str, torch.Tensor] | None] = []
    for entry in batch:
        items.append(
            {
                "id": entry["id"],
                "record_id": entry.get("record_id", ""),
                "step_index": entry.get("step_index", 0),
                "action_index": entry["action_index"],
                "action_value_target": entry["action_value_target"],
                "success": entry["success"],
                "messages": entry.get("messages"),
                "next_messages": entry.get("next_messages"),
            }
        )
        current_rows.append(entry["current_enc"])
        next_rows.append(entry.get("next_enc"))
    return {
        "items": items,
        "current_enc": collate_cached_encodings(current_rows, pad_token_id),
        "current_enc_rows": current_rows,
        "next_enc_rows": next_rows,
        "next_enc_bundle": _collate_next_encoding_bundle(
            items,
            next_rows,
            pad_token_id=pad_token_id,
        ),
    }


def unpack_transition_batch(
    batch,
    processor,
    max_length: int,
    *,
    pad_token_id: int | None = None,
    latent_token_count: int = 1,
    mask_latent_query_labels: bool = True,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    Any,
]:
    from nimloth.backbone.qwen25vl.batch import build_qwen_batch

    if isinstance(batch, dict) and "current_enc" in batch:
        items = batch["items"]
        enc = batch["current_enc"]
        next_rows = batch.get("next_enc_bundle", batch.get("next_enc_rows"))
        return items, enc, next_rows
    items = batch
    enc = build_qwen_batch(
        items,
        processor,
        max_length,
        latent_token_count=latent_token_count,
        mask_latent_query_labels=mask_latent_query_labels,
    )
    return items, enc, None
