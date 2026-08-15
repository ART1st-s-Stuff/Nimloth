"""Batch latent extraction from Qwen2.5-VL forward passes."""

from __future__ import annotations

from typing import Any

import torch

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


def _capture_last_hidden(model, model_inputs: dict[str, torch.Tensor]):
    captured: dict[str, torch.Tensor] = {}

    # State extraction reads the final decoder norm through the hook below; it
    # does not consume vocabulary logits.  Restrict the causal-LM projection to
    # one trailing position so long trajectory prefixes do not materialize a
    # full ``[sequence, vocab]`` tensor.  Supervised forwards keep their labels
    # and therefore retain the model's complete LM-loss semantics.
    if "labels" not in model_inputs:
        model_inputs = {**model_inputs, "logits_to_keep": 1}

    def hook(_module, _inputs, output):
        captured["hidden"] = output[0] if isinstance(output, tuple) else output

    handle = _final_norm_module(model).register_forward_hook(hook)
    try:
        output = model(**model_inputs, output_hidden_states=False, return_dict=True)
    finally:
        handle.remove()
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
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Extract configured latent query hidden states from a Qwen batch.

    Returns ``[B, H]`` for the legacy single-token case and ``[B, k, H]`` when
    ``latent_token_count > 1``.
    """

    model_inputs = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
    hidden, output = _capture_last_hidden(model, model_inputs)
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
    return torch.stack(rows, dim=0), output.loss
