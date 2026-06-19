"""Per-step WM and value loss computation during SFT2 training."""

from __future__ import annotations

import contextlib
from typing import Any

import torch
from transformers import AutoProcessor

from nimloth.training.common.qwen_batch import build_qwen_batch
from nimloth.backbone.vision_ema import VisionEncoderEMA
from nimloth.training.sft2.loss import compute_value_loss, compute_wm_latent_loss
from nimloth.training.sft2.qwen_latent import extract_qwen_latents
from nimloth.wm.predictor import LatentWMPredictor
from nimloth.wm.state_proj import StateProjector
from nimloth.wm.value_head import ValueHead


def wm_eligible_indices(items: list[dict[str, Any]]) -> list[int]:
    return [i for i, item in enumerate(items) if item.get("next_messages")]


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
) -> tuple[torch.Tensor | None, dict[str, float]]:
    indices = wm_eligible_indices(items)
    # Every rank must enter the same number of DDP model forwards; terminal steps have no
    # next prefix, so use the current prefix as a throwaway forward on those ranks.
    if indices:
        next_items = [{"messages": items[i]["next_messages"]} for i in indices]
    else:
        next_items = [{"messages": items[0]["messages"]}]
    next_enc = build_qwen_batch(next_items, processor, max_length)
    # Next-prefix latent is a stop-grad WM target; do not ask Qwen to compute CE.
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
        dummy_actions = torch.zeros((dummy_state_emb.shape[0],), device=device, dtype=torch.long)
        dummy_pred = wm_predictor(dummy_state_emb, dummy_actions)
        return (dummy_state_emb.sum() + dummy_target_emb.sum() + dummy_pred.sum()) * 0.0, {}
    action_indices = torch.tensor([items[i]["action_index"] for i in indices], device=device)
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
    action_indices = torch.tensor([item["action_index"] for item in items], device=device, dtype=torch.long)
    targets = torch.tensor(
        [float(item["action_value_target"]) for item in items],
        device=device,
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
