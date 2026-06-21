"""Trajectory-level forward equivalence checks (P4 prototype, not default trainer path)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoProcessor

from nimloth.latent import extract_latent_state, find_last_latent_state_index
from nimloth.training.common.qwen_batch import encode_qwen_item
from nimloth.training.sft2.qwen_latent import extract_qwen_latents, forward_qwen_last_hidden, reset_model_rope_state
from nimloth.wm.collate import prefix_messages_with_images
from nimloth.wm.dataset import TransitionSample, expand_record_transitions, load_jsonl_records


@dataclass(frozen=True)
class TrajectoryLatentEquivalence:
    record_id: str
    num_steps: int
    max_abs_diff: float
    mean_abs_diff: float
    passed: bool


@torch.no_grad()
def compare_prefix_vs_full_trajectory_latents(
    model,
    processor: AutoProcessor,
    record: dict[str, Any],
    token_id_map: dict[str, int],
    device: torch.device,
    *,
    max_length: int,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> TrajectoryLatentEquivalence:
    """Check per-step prefix latents match latents from one full-trajectory forward."""

    transitions = expand_record_transitions(record)
    if not transitions:
        raise ValueError(f"record {record.get('id')!r} produced no transitions")

    last = transitions[-1]
    full_messages = prefix_messages_with_images(last)
    full_enc = encode_qwen_item(full_messages, processor, max_length, include_labels=False)
    reset_model_rope_state(model)
    full_hidden = forward_qwen_last_hidden(model, _batch_enc(full_enc), device)
    full_ids = full_enc["input_ids"].tolist()

    prefix_latents: list[torch.Tensor] = []
    full_latents: list[torch.Tensor] = []
    for sample in transitions:
        messages = prefix_messages_with_images(sample)
        prefix_enc = encode_qwen_item(messages, processor, max_length, include_labels=False)
        prefix_ids = prefix_enc["input_ids"].tolist()
        prefix_latent_pos = find_last_latent_state_index(prefix_enc["input_ids"], token_id_map)
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(
                f"record {record.get('id')!r} step {sample.step_index}: prefix tokenization is not a "
                "prefix of full trajectory encoding"
            )
        prefix_latents.append(_prefix_latent(model, sample, processor, token_id_map, device, max_length))
        full_latents.append(extract_latent_state(full_hidden[0], prefix_latent_pos))

    prefix_stack = torch.stack(prefix_latents, dim=0)
    full_latents_stack = torch.stack(full_latents, dim=0)

    diff = (prefix_stack - full_latents_stack).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    passed = bool(torch.allclose(prefix_stack, full_latents_stack, rtol=rtol, atol=atol))
    return TrajectoryLatentEquivalence(
        record_id=str(record.get("id", "")),
        num_steps=len(transitions),
        max_abs_diff=max_abs,
        mean_abs_diff=mean_abs,
        passed=passed,
    )


def _batch_enc(enc: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in enc.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.unsqueeze(0) if value.ndim == 1 else value
        else:
            out[key] = value
    return out


@torch.no_grad()
def _prefix_latent(
    model,
    sample: TransitionSample,
    processor: AutoProcessor,
    token_id_map: dict[str, int],
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    messages = prefix_messages_with_images(sample)
    enc = encode_qwen_item(messages, processor, max_length, include_labels=False)
    reset_model_rope_state(model)
    latent, _ = extract_qwen_latents(model, _batch_enc(enc), token_id_map, device)
    return latent.squeeze(0)


def run_equivalence_on_jsonl(
    model,
    processor: AutoProcessor,
    jsonl_path,
    token_id_map: dict[str, int],
    device: torch.device,
    *,
    max_records: int = 3,
    max_length: int = 12000,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> list[TrajectoryLatentEquivalence]:
    results: list[TrajectoryLatentEquivalence] = []
    for record in load_jsonl_records(jsonl_path, max_records=max_records):
        results.append(
            compare_prefix_vs_full_trajectory_latents(
                model,
                processor,
                record,
                token_id_map,
                device,
                max_length=max_length,
                rtol=rtol,
                atol=atol,
            )
        )
    return results
