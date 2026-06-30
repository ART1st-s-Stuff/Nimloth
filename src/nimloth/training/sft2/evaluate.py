"""SFT2 validation loop."""

from __future__ import annotations

import contextlib

import torch

from nimloth.training.common.metrics import MetricAccumulator
from nimloth.backbone.vision_ema import VisionEncoderEMA
from nimloth.training.sft2.metrics import batch_step_success_rate
from nimloth.training.sft2.qwen_latent import extract_qwen_latents
from nimloth.training.sft2.preprocess_cache import unpack_transition_batch
from nimloth.training.sft2.step import compute_step_value_loss, compute_step_wm_loss, compute_trajectory_wm_loss
from nimloth.training.sft2.trajectory_once import forward_trajectory_once


@torch.no_grad()
def evaluate(
    model,
    state_proj,
    wm_predictor,
    value_head,
    loader,
    processor,
    token_id_map,
    device,
    *,
    max_batches: int = -1,
    max_length: int = 20000,
    vision_ema: VisionEncoderEMA | None = None,
    pad_token_id: int | None = None,
    packed_forward: bool = False,
    sigreg_module=None,
    lambda_sigreg: float = 0.0,
) -> dict[str, float]:
    model.eval()
    state_proj.eval()
    wm_predictor.eval()
    value_head.eval()
    acc = MetricAccumulator()
    ema_ctx = vision_ema.use_ema_weights(model) if vision_ema is not None else contextlib.nullcontext()
    with ema_ctx:
        for i, batch_samples in enumerate(loader):
            if max_batches > 0 and i >= max_batches:
                break
            if packed_forward:
                assert isinstance(batch_samples, dict) and "transition_samples" in batch_samples
                items = batch_samples["items"]
                transition_samples = batch_samples["transition_samples"]
                traj = forward_trajectory_once(
                    model,
                    transition_samples,
                    processor,
                    token_id_map,
                    device,
                    max_length=max_length,
                    vision_ema=vision_ema,
                    full_enc=batch_samples.get("full_enc"),
                )
                latent_hidden = traj.current_latents
                wm_loss, sigreg_loss, wm_metrics = compute_trajectory_wm_loss(
                    items,
                    latent_hidden,
                    traj.next_latents,
                    state_proj,
                    wm_predictor,
                    device,
                    sigreg_module=sigreg_module,
                )
            else:
                items, enc, next_enc_rows = unpack_transition_batch(
                    batch_samples,
                    processor,
                    max_length=max_length,
                    pad_token_id=pad_token_id,
                )
                enc.pop("labels", None)
                latent_hidden, _ = extract_qwen_latents(model, enc, token_id_map, device)
                wm_loss, sigreg_loss, wm_metrics = compute_step_wm_loss(
                    model,
                    items,
                    latent_hidden,
                    processor,
                    token_id_map,
                    device,
                    state_proj,
                    wm_predictor,
                    max_length,
                    vision_ema=vision_ema,
                    next_enc_rows=next_enc_rows,
                    pad_token_id=pad_token_id,
                    sigreg_module=sigreg_module,
                )
            _, value_metrics = compute_step_value_loss(
                latent_hidden,
                items,
                state_proj,
                value_head,
                device,
                rank_margin=0.0,
                lambda_rank=0.0,
            )
            success_rate = batch_step_success_rate(items)
            acc.update({**wm_metrics, **value_metrics, "success_rate": success_rate})

    model.train()
    state_proj.train()
    wm_predictor.train()
    value_head.train()
    return acc.averages()
