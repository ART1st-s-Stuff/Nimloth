"""Batch latent extraction from Qwen2.5-VL forward passes."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from nimloth.latent import (
    extract_latent_state,
    extract_latent_state_block,
    find_last_latent_state_block,
    find_last_latent_state_index,
)
from nimloth.latent.extraction import LatentActionTokens


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def reset_model_rope_state(model) -> None:
    root = _unwrap_model(model)
    inner = getattr(root, "model", root)
    if hasattr(inner, "rope_deltas"):
        inner.rope_deltas = None


def _get_attr_path(obj: Any, path: str) -> Any | None:
    cur = obj
    for name in path.split("."):
        cur = getattr(cur, name, None)
        if cur is None:
            return None
    return cur


def _final_norm_module(model) -> torch.nn.Module:
    """Resolve the final text-model norm used to produce last hidden states.

    Calling Qwen with ``output_hidden_states=True`` returns every layer's hidden
    states. Agent state extraction only needs last-layer activations at configured
    latent query tokens, so we capture the output of the final decoder norm.
    The candidate paths cover current HF Qwen2.5-VL naming and older variants.
    """

    root = _unwrap_model(model)
    for path in (
        "model.language_model.norm",
        "model.model.norm",
        "base_model.model.model.language_model.norm",
        "base_model.model.model.model.norm",
        "base_model.model.language_model.norm",
        "language_model.norm",
        "model.norm",
    ):
        module = _get_attr_path(root, path)
        if isinstance(module, torch.nn.Module):
            return module
    raise RuntimeError(
        "Could not locate Qwen final norm module for latent extraction; "
        "update _final_norm_module for this model architecture."
    )


def _capture_last_hidden(
    model, model_inputs: dict[str, torch.Tensor], *, full_logits: bool = False,
    logit_mask: torch.Tensor | None = None,
):
    # These are complete-prefix feature/teacher-forcing forwards, never KV-cache
    # decoding. Explicitly disable the model default even under no_grad (EMA
    # targets), where gradient checkpointing does not disable it for us.
    model_inputs = {**model_inputs, "use_cache": False}
    captured: dict[str, torch.Tensor] = {}

    # Capture complete decoder states, but project only positions needed by the
    # caller. A packed supervised projection remains one head forward on every
    # rank, including ranks with zero successful rows; FSDP ownership is intact.
    if (full_logits or logit_mask is not None) and "labels" in model_inputs:
        raise ValueError("external logit capture computes its loss without labels")
    if logit_mask is not None:
        if logit_mask.dtype != torch.bool or logit_mask.shape != model_inputs["input_ids"].shape:
            raise ValueError("logit mask must be boolean with the input sequence shape")
        if not logit_mask.any():
            raise ValueError("supervised projection requires at least one position")

    projection_handle = None
    if logit_mask is not None or (not full_logits and "labels" not in model_inputs):
        root = _unwrap_model(model)
        head = root.get_output_embeddings()
        if not isinstance(head, torch.nn.Module):
            raise RuntimeError("Qwen output embedding module is required for bounded logits")

        def select_projection(_module, args):
            hidden = args[0]
            selected = (hidden[logit_mask].unsqueeze(0) if logit_mask is not None
                        else hidden[:, -1:, :])
            return (selected, *args[1:])

        projection_handle = head.register_forward_pre_hook(select_projection)

    def hook(_module, _inputs, output):
        captured["hidden"] = output[0] if isinstance(output, tuple) else output

    handle = None
    try:
        handle = _final_norm_module(model).register_forward_hook(hook)
        output = model(**model_inputs, output_hidden_states=False, return_dict=True)
    finally:
        if handle is not None:
            handle.remove()
        if projection_handle is not None:
            projection_handle.remove()
    hidden = captured.get("hidden")
    if hidden is None:
        raise RuntimeError("Qwen final norm hook did not capture last hidden states.")
    return hidden, output


def forward_qwen_last_hidden(model, enc: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    """Run Qwen forward and return last-layer hidden states ``[batch, seq, dim]``."""

    model_inputs = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
    hidden, _ = _capture_last_hidden(model, model_inputs)
    return hidden


def extract_qwen_action_boundary_hidden(
    model,
    enc: dict[str, torch.Tensor],
    token_id_map: dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    """Return final-norm hidden rows at each sample's last ``action_start``.

    Action-head repair freezes Qwen and must not materialize supervised
    full-vocabulary logits.  Labels are therefore rejected and the ordinary
    final-norm hook captures the causal boundary state in the same forward used
    for the K-slot state prompt.
    """

    if "labels" in enc:
        raise ValueError("action boundary extraction must not include labels")
    model_inputs = {key: value.to(device, non_blocking=True) for key, value in enc.items()}
    hidden, _output = _capture_last_hidden(model, model_inputs)
    action_start_id = token_id_map[LatentActionTokens().action_start]
    input_ids = enc["input_ids"].detach().cpu()
    rows: list[torch.Tensor] = []
    for row in range(hidden.shape[0]):
        positions = (input_ids[row] == int(action_start_id)).nonzero(as_tuple=True)[0]
        if positions.numel() < 1:
            raise RuntimeError(
                f"Qwen input row {row} has no action_start token for repair"
            )
        rows.append(hidden[row, int(positions[-1].item())])
    boundary = torch.stack(rows, dim=0)
    if boundary.ndim != 2 or not torch.isfinite(boundary).all():
        raise RuntimeError("Qwen action boundary hidden is invalid")
    return boundary


def extract_qwen_latents(
    model,
    enc: dict[str, torch.Tensor],
    token_id_map: dict[str, int],
    device: torch.device,
    *,
    latent_token_count: int = 1,
    lm_row_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Extract configured latent query hidden states from a Qwen batch.

    Returns ``[B, H]`` for the legacy single-token case and ``[B, k, H]`` when
    ``latent_token_count > 1``.
    """

    model_inputs = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
    labels = model_inputs.pop("labels") if lm_row_weights is not None else None
    logit_mask = None
    counts = None
    if labels is not None:
        weights = lm_row_weights.to(device)
        if weights.shape != (labels.shape[0],) or not torch.all((weights == 0) | (weights == 1)):
            raise ValueError("LM row weights must be zero or one for each input row")
        logit_mask = torch.zeros_like(labels, dtype=torch.bool)
        logit_mask[:, :-1] = labels[:, 1:] != -100
        counts = logit_mask.sum(dim=1).tolist()
        if any(count == 0 for count in counts):
            raise ValueError("LM window has no supervised answer tokens")
    hidden, output = _capture_last_hidden(model, model_inputs, logit_mask=logit_mask)
    lm_loss = output.loss
    if labels is not None:
        # Packed positions are row-major. Keep each window's token mean and its
        # original binary weight, including differentiable zero for failed rows.
        targets = labels[:, 1:][logit_mask[:, :-1]]
        scores = output.logits.squeeze(0)
        if scores.ndim != 2 or scores.shape[0] != sum(counts):
            raise RuntimeError("Qwen supervised projection did not preserve packed positions")
        row_losses = []
        offset = 0
        def token_ce(scores, targets):
            return F.cross_entropy(scores.float(), targets, reduction="sum")
        for count in counts:
            row_sum = scores.new_zeros((), dtype=torch.float32)
            for start in range(offset, offset + count, 128):
                end = min(start + 128, offset + count)
                chunk_scores, chunk_targets = scores[start:end], targets[start:end]
                row_sum = row_sum + (
                    checkpoint(token_ce, chunk_scores, chunk_targets, use_reentrant=False)
                    if torch.is_grad_enabled() and chunk_scores.requires_grad
                    else token_ce(chunk_scores, chunk_targets)
                )
            row_losses.append(row_sum / count)
            offset += count
        lm_loss = (torch.stack(row_losses) * weights).sum() / weights.sum().clamp_min(1)
    tokens = LatentActionTokens()
    rows: list[torch.Tensor] = []
    input_ids = enc["input_ids"].detach().cpu()
    for row in range(hidden.shape[0]):
        if latent_token_count == 1:
            latent_index = find_last_latent_state_index(input_ids[row], token_id_map, tokens)
            rows.append(extract_latent_state(hidden[row : row + 1], latent_index))
        else:
            latent_block = find_last_latent_state_block(
                input_ids[row],
                token_id_map,
                tokens,
                latent_token_count=latent_token_count,
            )
            rows.append(extract_latent_state_block(hidden[row : row + 1], latent_block))
    state_hidden = torch.stack(rows, dim=0)
    if lm_loss is None and torch.is_grad_enabled():
        state_hidden = connect_unused_logits(state_hidden, output.logits)
    return state_hidden, lm_loss


def connect_unused_logits(hidden: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Keep an already-computed LM projection in hidden-only backward graphs.

    Stage3 alternates LM and SIGReg forwards under static DDP. The bounded
    hidden-only logits otherwise receive no backward hook, violating that
    reducer contract for trainable vocabulary rows. This scalar zero adds no
    LM objective, no projection forward, and no full-sequence vocabulary buffer.
    """
    if not logits.requires_grad:
        return hidden
    return hidden + (logits.float().sum() * 0.0).to(hidden.dtype)
