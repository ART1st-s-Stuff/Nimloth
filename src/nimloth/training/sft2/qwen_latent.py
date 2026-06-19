"""Batch latent extraction from Qwen forward passes."""

from __future__ import annotations

from typing import Any

import torch

from nimloth.latent import extract_latent_state, find_last_latent_state_index
from nimloth.latent.extraction import LatentActionTokens


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


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
    states.  For SFT2 we only need the last-layer activations at
    ``<|latent_state|>``, so we capture the output of the final decoder norm.
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


def extract_qwen_latents(
    model,
    enc: dict[str, torch.Tensor],
    token_id_map: dict[str, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    model_inputs = {k: v.to(device) for k, v in enc.items()}
    hidden, output = _capture_last_hidden(model, model_inputs)
    tokens = LatentActionTokens()
    rows: list[torch.Tensor] = []
    input_ids = enc["input_ids"].detach().cpu()
    for row in range(hidden.shape[0]):
        latent_index = find_last_latent_state_index(input_ids[row], token_id_map, tokens)
        rows.append(extract_latent_state(hidden[row : row + 1], latent_index))
    return torch.stack(rows, dim=0), output.loss
