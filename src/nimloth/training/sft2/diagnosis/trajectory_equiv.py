"""诊断 legacy per-prefix 与 trajectory-once 数值是否等价。"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nimloth.backbone.qwen25vl.batch import build_qwen_batch
from nimloth.backbone.qwen25vl.latent import extract_qwen_latents
from nimloth.rollout.transitions import collate_transition_training_items
from nimloth.rollout.transitions import expand_record_transitions
from nimloth.training.common import action_value_loss
from nimloth.training.sft2.diagnosis.trajectory_once import (
    forward_trajectory_once,
    supervised_token_count,
)
from nimloth.wm.model import WorldModel


def _world_model_and_value_losses(
    *,
    current_hidden: torch.Tensor,
    next_hidden: torch.Tensor | None,
    items: list[dict],
    eligible_indices: list[int],
    wm: WorldModel,
) -> tuple[torch.Tensor, torch.Tensor]:
    actions = torch.tensor(
        [item["action_index"] for item in items],
        dtype=torch.long,
        device=current_hidden.device,
    )
    current_state = wm.project_state(current_hidden)
    predicted_next_state = wm.predict_next_state(current_state, actions)
    if eligible_indices:
        assert next_hidden is not None
        target_next_state = wm.project_state(next_hidden)
        dynamics_loss = F.mse_loss(
            predicted_next_state[eligible_indices],
            target_next_state,
        )
    else:
        dynamics_loss = torch.zeros((), device=current_hidden.device)

    values = wm.predict_action_values(current_state)
    targets = torch.tensor(
        [item["action_value_target"] for item in items],
        dtype=values.dtype,
        device=values.device,
    )
    value_objective = action_value_loss(
        values,
        actions,
        targets,
    )
    return dynamics_loss, value_objective.loss


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
    wm = WorldModel(
        state_proj=state_proj,
        wm_predictor=wm_predictor,
        value_head=value_head,
    )
    steps = expand_record_transitions(record)
    items = collate_transition_training_items(steps)
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

    eligible = [index for index, item in enumerate(items) if item.get("next_messages")]
    if eligible:
        next_encoding = build_qwen_batch(
            [{"messages": items[index]["next_messages"]} for index in eligible],
            processor,
            max_length,
        )
        next_encoding.pop("labels", None)
        next_hidden, _ = extract_qwen_latents(
            model,
            next_encoding,
            token_id_map,
            device,
        )
    else:
        next_hidden = None
    wm_loss, value_loss = _world_model_and_value_losses(
        current_hidden=current,
        next_hidden=next_hidden,
        items=items,
        eligible_indices=eligible,
        wm=wm,
    )
    total = wm_loss + value_loss
    if lm_loss_batch is not None:
        total = total + lm_loss_batch
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
    wm = WorldModel(
        state_proj=state_proj,
        wm_predictor=wm_predictor,
        value_head=value_head,
    )
    steps = expand_record_transitions(record)
    items = collate_transition_training_items(steps)
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
    wm_loss, value_loss = _world_model_and_value_losses(
        current_hidden=trajectory.current_latents,
        next_hidden=next_hidden,
        items=items,
        eligible_indices=eligible,
        wm=wm,
    )
    total = wm_loss + value_loss
    if trajectory.lm_loss is not None:
        total = total + trajectory.lm_loss
    return {
        "current": trajectory.current_latents,
        "lm_loss": trajectory.lm_loss,
        "wm_loss": wm_loss,
        "value_loss": value_loss,
        "total_loss": total,
    }
