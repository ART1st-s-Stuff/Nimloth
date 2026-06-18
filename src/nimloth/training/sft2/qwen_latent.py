"""Batch latent extraction from Qwen forward passes."""

from __future__ import annotations

import torch

from nimloth.latent import extract_latent_state, find_last_latent_state_index, last_hidden_state
from nimloth.latent.extraction import LatentActionTokens


def extract_qwen_latents(
    model,
    enc: dict[str, torch.Tensor],
    token_id_map: dict[str, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    model_inputs = {k: v.to(device) for k, v in enc.items()}
    output = model(**model_inputs, output_hidden_states=True, return_dict=True)
    hidden = last_hidden_state(output)
    tokens = LatentActionTokens()
    rows: list[torch.Tensor] = []
    for row in range(hidden.shape[0]):
        latent_index = find_last_latent_state_index(enc["input_ids"][row], token_id_map, tokens)
        rows.append(extract_latent_state(hidden[row : row + 1], latent_index))
    return torch.stack(rows, dim=0), output.loss
