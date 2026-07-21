"""诊断 legacy per-prefix 与 trajectory-once 数值是否等价。"""

from __future__ import annotations

import torch

from nimloth.backbone.qwen25vl.batch import build_qwen_batch
from nimloth.backbone.qwen25vl.latent import extract_qwen_latents
from nimloth.backbone.qwen25vl.transition import (
    QwenTransitionEncoder,
    QwenTransitionMessages,
    transition_collate_for_qwen,
)
from nimloth.rollout.transitions import expand_record_transitions
from nimloth.training.sft2.algorithm import (
    SFT2Losses,
    combine_sft2_losses,
    compute_sft2_dynamics,
)
from nimloth.training.sft2.diagnosis.trajectory_once import (
    forward_trajectory_once,
    supervised_token_count,
)
from nimloth.wm.objectives import compute_action_value_loss


def _trajectory_step(item: dict) -> tuple[str, int]:
    record_id = str(item.get("record_id", ""))
    if not record_id and ":" in str(item.get("id", "")):
        record_id = str(item["id"]).split(":", 1)[0]
    return record_id, int(item.get("step_index", 0))


def _auxiliary_losses(
    *,
    current_hidden: torch.Tensor,
    next_hidden: torch.Tensor | None,
    items: list[dict],
    eligible_indices: list[int],
    state_proj,
    wm_predictor,
    value_head,
) -> tuple[torch.Tensor, torch.Tensor]:
    if eligible_indices:
        assert next_hidden is not None
        dynamics, _ = compute_sft2_dynamics(
            current_hidden=current_hidden[eligible_indices],
            next_hidden=next_hidden,
            action_indices=torch.tensor(
                [items[index]["action_index"] for index in eligible_indices],
                dtype=torch.long,
                device=current_hidden.device,
            ),
            trajectory_steps=[
                _trajectory_step(items[index]) for index in eligible_indices
            ],
            state_proj=state_proj,
            wm_predictor=wm_predictor,
            sigreg=None,
        )
        dynamics_loss = dynamics.loss
    else:
        dynamics_loss = torch.zeros((), device=current_hidden.device)

    value = compute_action_value_loss(
        state=state_proj(current_hidden),
        action_indices=torch.tensor(
            [item["action_index"] for item in items],
            dtype=torch.long,
            device=current_hidden.device,
        ),
        return_targets=torch.tensor(
            [item["action_value_target"] for item in items],
            dtype=torch.float32,
            device=current_hidden.device,
        ),
        value_head=value_head,
        rank_margin=0.1,
        rank_weight=1.0,
    )
    return dynamics_loss, value.loss


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
        encoding = build_qwen_batch([item], processor, max_length)
        latent, lm_loss = extract_qwen_latents(model, encoding, token_id_map, device)
        latents.append(latent.squeeze(0))
        if lm_loss is not None:
            token_count = supervised_token_count(encoding["labels"][0])
            lm_total = lm_total + lm_loss * token_count
            lm_tokens += token_count
    current = torch.stack(latents, dim=0)
    lm_loss_batch = lm_total / lm_tokens if lm_tokens else None

    qwen = QwenTransitionEncoder(
        model=model,
        processor=processor,
        token_id_map=token_id_map,
        device=device,
        max_length=max_length,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    messages = [
        QwenTransitionMessages(
            current=item["messages"],
            next=item.get("next_messages"),
        )
        for item in items
    ]
    eligible = [index for index, value in enumerate(messages) if value.next is not None]
    next_hidden = (
        qwen.encode_next(messages, eligible, cached=None, use_vision_ema=False)
        if eligible
        else None
    )
    wm_loss, value_loss = _auxiliary_losses(
        current_hidden=current,
        next_hidden=next_hidden,
        items=items,
        eligible_indices=eligible,
        state_proj=state_proj,
        wm_predictor=wm_predictor,
        value_head=value_head,
    )
    total = combine_sft2_losses(
        SFT2Losses(
            lm=lm_loss_batch,
            dynamics=wm_loss,
            sigreg=None,
            value=value_loss,
            metrics={},
        ),
        wm_weight=1.0,
        sigreg_weight=0.0,
        value_weight=1.0,
        ce_weight=1.0,
    ).loss
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
    trajectory = forward_trajectory_once(
        model,
        steps,
        processor,
        token_id_map,
        device,
        max_length=max_length,
    )
    eligible = [index for index, item in enumerate(items) if item.get("next_messages")]
    next_hidden = (
        torch.stack([trajectory.next_latents[index] for index in eligible], dim=0)
        if eligible and trajectory.next_latents is not None
        else None
    )
    wm_loss, value_loss = _auxiliary_losses(
        current_hidden=trajectory.current_latents,
        next_hidden=next_hidden,
        items=items,
        eligible_indices=eligible,
        state_proj=state_proj,
        wm_predictor=wm_predictor,
        value_head=value_head,
    )
    total = combine_sft2_losses(
        SFT2Losses(
            lm=trajectory.lm_loss,
            dynamics=wm_loss,
            sigreg=None,
            value=value_loss,
            metrics={},
        ),
        wm_weight=1.0,
        sigreg_weight=0.0,
        value_weight=1.0,
        ce_weight=1.0,
    ).loss
    return {
        "current": trajectory.current_latents,
        "lm_loss": trajectory.lm_loss,
        "wm_loss": wm_loss,
        "value_loss": value_loss,
        "total_loss": total,
    }
