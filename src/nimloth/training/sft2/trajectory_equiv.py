"""Legacy vs trajectory-once equivalence helpers (GPU validation)."""

from __future__ import annotations

import torch

from nimloth.training.common.qwen_batch import build_qwen_batch
from nimloth.training.sft2.loss import compute_combined_loss, compute_wm_latent_loss
from nimloth.training.sft2.qwen_latent import extract_qwen_latents
from nimloth.training.sft2.step import compute_step_value_loss, compute_step_wm_loss, wm_eligible_indices
from nimloth.training.sft2.trajectory_once import forward_trajectory_once, supervised_token_count
from nimloth.wm.collate import transition_collate_for_qwen
from nimloth.wm.dataset import expand_record_transitions


@torch.no_grad()
def legacy_record_losses(
    model,
    processor,
    token_id_map,
    device,
    record,
    max_length,
    state_proj,
    wm_predictor,
    value_head,
):
    steps = expand_record_transitions(record)
    items = transition_collate_for_qwen(steps)
    latents = []
    lm_total = torch.zeros((), device=device)
    lm_tokens = 0
    for item in items:
        enc = build_qwen_batch([item], processor, max_length)
        latent, lm_loss = extract_qwen_latents(model, enc, token_id_map, device)
        latents.append(latent.squeeze(0))
        if lm_loss is not None:
            n = supervised_token_count(enc["labels"][0])
            lm_total = lm_total + lm_loss * n
            lm_tokens += n
    current = torch.stack(latents, dim=0)
    lm_loss_batch = lm_total / lm_tokens if lm_tokens else None
    wm_loss, _ = compute_step_wm_loss(
        model, items, current, processor, token_id_map, device, state_proj, wm_predictor, max_length
    )
    value_loss, _ = compute_step_value_loss(
        current, items, state_proj, value_head, device, rank_margin=0.1, lambda_rank=1.0
    )
    total, _ = compute_combined_loss(
        wm_loss=wm_loss, value_loss=value_loss, lm_loss=lm_loss_batch, lambda_wm=1.0
    )
    return {
        "current": current,
        "lm_loss": lm_loss_batch,
        "wm_loss": wm_loss,
        "value_loss": value_loss,
        "total_loss": total,
    }


@torch.no_grad()
def packed_record_losses(
    model,
    processor,
    token_id_map,
    device,
    record,
    max_length,
    state_proj,
    wm_predictor,
    value_head,
):
    steps = expand_record_transitions(record)
    items = transition_collate_for_qwen(steps)
    traj = forward_trajectory_once(model, steps, processor, token_id_map, device, max_length=max_length)
    indices = wm_eligible_indices(items)
    if indices:
        assert traj.next_latents is not None
        next_rows = torch.stack([traj.next_latents[i] for i in indices], dim=0)
        action_indices = torch.tensor([items[i]["action_index"] for i in indices], device=device)
        wm_loss, _ = compute_wm_latent_loss(
            qwen_hidden_at_latent=traj.current_latents[indices],
            qwen_hidden_at_next_latent=next_rows,
            action_indices=action_indices,
            state_proj=state_proj,
            wm_predictor=wm_predictor,
        )
    else:
        wm_loss = torch.zeros((), device=device)
    value_loss, _ = compute_step_value_loss(
        traj.current_latents, items, state_proj, value_head, device, rank_margin=0.1, lambda_rank=1.0
    )
    total, _ = compute_combined_loss(
        wm_loss=wm_loss,
        value_loss=value_loss,
        lm_loss=traj.lm_loss,
        lambda_wm=1.0,
    )
    return {
        "current": traj.current_latents,
        "lm_loss": traj.lm_loss,
        "wm_loss": wm_loss,
        "value_loss": value_loss,
        "total_loss": total,
    }
