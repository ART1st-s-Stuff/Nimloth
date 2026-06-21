"""Per-step WM and value loss computation during SFT2 training."""

from __future__ import annotations

import contextlib
from typing import Any

import torch
from transformers import AutoProcessor

from nimloth.training.common.qwen_batch import _message_cache_key, build_qwen_batch
from nimloth.training.sft2.preprocess_cache import collate_cached_encodings
from nimloth.backbone.vision_ema import VisionEncoderEMA
from nimloth.training.sft2.loss import compute_value_loss, compute_wm_latent_loss
from nimloth.training.sft2.qwen_latent import extract_qwen_latents
from nimloth.wm.predictor import LatentWMPredictor
from nimloth.wm.state_proj import StateProjector
from nimloth.wm.value_head import ValueHead


def wm_eligible_indices(items: list[dict[str, Any]]) -> list[int]:
    return [i for i, item in enumerate(items) if item.get("next_messages")]


def _next_messages_key(messages: list[dict[str, Any]] | None) -> str:
    if not messages:
        return ""
    return _message_cache_key(messages)


def _collate_next_enc_rows(
    rows: list[dict[str, torch.Tensor] | None],
    indices: list[int],
    *,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    selected = [rows[i] for i in indices]
    if any(row is None for row in selected):
        raise ValueError("cached next_enc missing for WM-eligible transition")
    return collate_cached_encodings(selected, pad_token_id)  # type: ignore[arg-type]


def _forward_next_latents(
    model,
    items: list[dict[str, Any]],
    indices: list[int],
    processor: AutoProcessor,
    token_id_map: dict[str, int],
    device: torch.device,
    max_length: int,
    *,
    vision_ema: VisionEncoderEMA | None,
    next_enc_rows: list[dict[str, torch.Tensor] | None] | None,
    pad_token_id: int | None,
) -> torch.Tensor:
    if not indices:
        raise ValueError("indices must be non-empty")

    unique_keys: list[str] = []
    key_to_unique_row: dict[str, int] = {}
    for i in indices:
        key = _next_messages_key(items[i].get("next_messages"))
        if key not in key_to_unique_row:
            key_to_unique_row[key] = len(unique_keys)
            unique_keys.append(key)

    unique_indices = [indices[key_to_unique_row[key]] for key in unique_keys]

    if next_enc_rows is not None:
        if pad_token_id is None:
            raise ValueError("pad_token_id required when using cached next_enc_rows")
        next_enc = _collate_next_enc_rows(next_enc_rows, unique_indices, pad_token_id=pad_token_id)
    else:
        next_items = [{"messages": items[i]["next_messages"]} for i in unique_indices]
        next_enc = build_qwen_batch(next_items, processor, max_length)

    next_enc.pop("labels", None)
    ema_ctx = vision_ema.use_ema_weights(model) if vision_ema is not None else contextlib.nullcontext()
    with torch.no_grad(), ema_ctx:
        next_latent_unique, _ = extract_qwen_latents(model, next_enc, token_id_map, device)

    rows: list[torch.Tensor] = []
    for i in indices:
        key = _next_messages_key(items[i].get("next_messages"))
        rows.append(next_latent_unique[key_to_unique_row[key] : key_to_unique_row[key] + 1])
    return torch.cat(rows, dim=0)


def compute_step_wm_loss(
    model,
    items: list[dict[str, Any]],
    current_latent: torch.Tensor,
    processor: AutoProcessor,
    token_id_map: dict[str, int],
    device: torch.device,
    state_proj: StateProjector,
    wm_predictor: LatentWMPredictor,
    max_length: int,
    *,
    vision_ema: VisionEncoderEMA | None = None,
    next_enc_rows: list[dict[str, torch.Tensor] | None] | None = None,
    pad_token_id: int | None = None,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    indices = wm_eligible_indices(items)
    # Every rank must enter the same number of DDP model forwards; terminal steps have no
    # next prefix, so use the current prefix as a throwaway forward on those ranks.
    if indices:
        next_latent = _forward_next_latents(
            model,
            items,
            indices,
            processor,
            token_id_map,
            device,
            max_length,
            vision_ema=vision_ema,
            next_enc_rows=next_enc_rows,
            pad_token_id=pad_token_id,
        )
    else:
        next_items = [{"messages": items[0]["messages"]}]
        next_enc = build_qwen_batch(next_items, processor, max_length)
        next_enc.pop("labels", None)
        ema_ctx = vision_ema.use_ema_weights(model) if vision_ema is not None else contextlib.nullcontext()
        with torch.no_grad(), ema_ctx:
            next_latent, _ = extract_qwen_latents(model, next_enc, token_id_map, device)
    if not indices:
        # DDP-wrapped aux modules must also be called on terminal-only ranks.
        # A parameter-tied zero without DDP forward leaves other ranks waiting
        # for aux gradient all-reduce when they do have WM-eligible samples.
        dummy_state_emb = state_proj(current_latent[:1])
        with torch.no_grad():
            dummy_target_emb = state_proj(next_latent[:1])
        dummy_actions = torch.zeros((dummy_state_emb.shape[0],), device=dummy_state_emb.device, dtype=torch.long)
        dummy_pred = wm_predictor(dummy_state_emb, dummy_actions)
        return (dummy_state_emb.sum() + dummy_target_emb.sum() + dummy_pred.sum()) * 0.0, {}
    action_indices = torch.tensor([items[i]["action_index"] for i in indices], device=current_latent.device)
    return compute_wm_latent_loss(
        qwen_hidden_at_latent=current_latent[indices],
        qwen_hidden_at_next_latent=next_latent,
        action_indices=action_indices,
        state_proj=state_proj,
        wm_predictor=wm_predictor,
    )


def compute_step_value_loss(
    current_latent: torch.Tensor,
    items: list[dict[str, Any]],
    state_proj: StateProjector,
    value_head: ValueHead,
    device: torch.device,
    *,
    rank_margin: float,
    lambda_rank: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    latent_device = current_latent.device
    action_indices = torch.tensor([item["action_index"] for item in items], device=latent_device, dtype=torch.long)
    targets = torch.tensor(
        [float(item["action_value_target"]) for item in items],
        device=latent_device,
        dtype=torch.float32,
    )
    state_emb = state_proj(current_latent)
    return compute_value_loss(
        state_emb=state_emb,
        action_indices=action_indices,
        action_value_targets=targets,
        value_head=value_head,
        rank_margin=rank_margin,
        lambda_rank=lambda_rank,
    )


def compute_trajectory_wm_loss(
    items: list[dict[str, Any]],
    current_latents: torch.Tensor,
    next_latents: torch.Tensor | None,
    state_proj: StateProjector,
    wm_predictor: LatentWMPredictor,
    device: torch.device,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """WM loss from precomputed trajectory latents (no extra Qwen forward)."""

    indices = wm_eligible_indices(items)
    if indices:
        if next_latents is None:
            raise ValueError("next_latents required for WM-eligible trajectory steps")
        next_rows = torch.stack([next_latents[i] for i in indices], dim=0)
        action_indices = torch.tensor([items[i]["action_index"] for i in indices], device=device)
        return compute_wm_latent_loss(
            qwen_hidden_at_latent=current_latents[indices],
            qwen_hidden_at_next_latent=next_rows,
            action_indices=action_indices,
            state_proj=state_proj,
            wm_predictor=wm_predictor,
        )

    dummy_state_emb = state_proj(current_latents[:1])
    with torch.no_grad():
        dummy_target_emb = state_proj(current_latents[:1])
    dummy_actions = torch.zeros((dummy_state_emb.shape[0],), device=dummy_state_emb.device, dtype=torch.long)
    dummy_pred = wm_predictor(dummy_state_emb, dummy_actions)
    return (dummy_state_emb.sum() + dummy_target_emb.sum() + dummy_pred.sum()) * 0.0, {}
