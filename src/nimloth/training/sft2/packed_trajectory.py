"""P4: trajectory-level KV incremental forward (strictly equivalent to per-prefix legacy)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoProcessor

from nimloth.latent import extract_latent_state, find_last_latent_state_index
from nimloth.training.common.qwen_batch import encode_qwen_item
from nimloth.training.sft2.qwen_latent import _capture_last_hidden, extract_qwen_latents
from nimloth.training.sft2.trajectory_forward import _batch_enc
from nimloth.wm.collate import prefix_messages_with_images, transition_collate_for_qwen
from nimloth.wm.dataset import TransitionSample


@dataclass(frozen=True)
class TrajectoryStepResult:
    step_index: int
    current_latent: torch.Tensor
    enc: dict[str, torch.Tensor]
    item: dict[str, Any]


@dataclass
class TrajectoryForwardState:
    past_key_values: Any | None = None
    rope_deltas: torch.Tensor | None = None
    prev_enc: dict[str, torch.Tensor] | None = None


def assert_prefix_stable(encodings: list[dict[str, torch.Tensor]], *, record_id: str) -> None:
    if not encodings:
        return
    full_ids = encodings[-1]["input_ids"].tolist()
    for enc in encodings:
        prefix_ids = enc["input_ids"].tolist()
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(f"record {record_id!r}: prefix tokenization is not stable in full trajectory")


def vision_delta(
    enc_prev: dict[str, torch.Tensor] | None,
    enc_cur: dict[str, torch.Tensor],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if "pixel_values" not in enc_cur:
        return None, None
    cur_pv = enc_cur["pixel_values"]
    prev_rows = 0
    if enc_prev is not None and "pixel_values" in enc_prev:
        prev_rows = int(enc_prev["pixel_values"].shape[0])
    if cur_pv.shape[0] <= prev_rows:
        return None, None
    delta_pv = cur_pv[prev_rows:]
    if "image_grid_thw" not in enc_cur:
        return delta_pv, None
    prev_images = 0
    if enc_prev is not None and "image_grid_thw" in enc_prev:
        prev_images = int(enc_prev["image_grid_thw"].shape[0])
    delta_grid = enc_cur["image_grid_thw"][prev_images:]
    if delta_grid.shape[0] == 0:
        raise ValueError("vision delta has pixel rows but empty image_grid_thw")
    return delta_pv, delta_grid


def _to_device(enc: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in enc.items() if isinstance(v, torch.Tensor)}


def legacy_forward_trajectory(
    model,
    transitions: list[TransitionSample],
    processor: AutoProcessor,
    token_id_map: dict[str, int],
    device: torch.device,
    *,
    max_length: int,
) -> list[TrajectoryStepResult]:
    results: list[TrajectoryStepResult] = []
    for sample in transitions:
        item = transition_collate_for_qwen([sample])[0]
        enc = encode_qwen_item(item["messages"], processor, max_length, include_labels=True)
        latent, _ = extract_qwen_latents(model, _batch_enc(enc), token_id_map, device)
        results.append(
            TrajectoryStepResult(
                step_index=sample.step_index,
                current_latent=latent.squeeze(0),
                enc=enc,
                item=item,
            )
        )
    return results


from nimloth.training.sft2.qwen_latent import reset_model_rope_state as _reset_model_rope_state


def _enc_model_inputs(enc_cur: dict[str, torch.Tensor], device: torch.device) -> dict[str, Any]:
    batch = _batch_enc(enc_cur)
    inputs = _to_device(batch, device)
    if "labels" in enc_cur and "labels" not in inputs:
        inputs["labels"] = enc_cur["labels"].unsqueeze(0).to(device)
    return inputs


def kv_forward_step_train(
    model,
    enc_cur: dict[str, torch.Tensor],
    state: TrajectoryForwardState | None,
    device: torch.device,
    token_id_map: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor | None, TrajectoryForwardState]:
    pixel_values, _image_grid_thw = vision_delta(None if state is None else state.prev_enc, enc_cur)
    if state is None or state.prev_enc is None or pixel_values is not None:
        hidden, output = _capture_last_hidden(model, _enc_model_inputs(enc_cur, device))
        latent_pos = find_last_latent_state_index(enc_cur["input_ids"], token_id_map)
        return (
            extract_latent_state(hidden[0], latent_pos),
            output.loss,
            TrajectoryForwardState(
                past_key_values=output.past_key_values,
                rope_deltas=getattr(output, "rope_deltas", None),
                prev_enc=enc_cur,
            ),
        )

    prev_enc = state.prev_enc
    prev_len = int(prev_enc["input_ids"].shape[0])
    cur_len = int(enc_cur["input_ids"].shape[0])
    prev_ids = prev_enc["input_ids"].tolist()
    cur_ids = enc_cur["input_ids"].tolist()
    if prev_ids != cur_ids[:prev_len]:
        raise ValueError("incremental KV requires stable prefix tokenization")

    delta_ids = enc_cur["input_ids"][prev_len:].unsqueeze(0).to(device)
    delta_mask = enc_cur["attention_mask"][prev_len:].unsqueeze(0).to(device)
    cache_position = torch.arange(prev_len, cur_len, device=device)
    kwargs: dict[str, Any] = dict(
        input_ids=delta_ids,
        attention_mask=delta_mask,
        past_key_values=state.past_key_values,
        cache_position=cache_position,
        use_cache=True,
    )
    if "labels" in enc_cur:
        kwargs["labels"] = enc_cur["labels"][prev_len:].unsqueeze(0).to(device)

    hidden, output = _capture_last_hidden(model, kwargs)
    latent_pos = find_last_latent_state_index(enc_cur["input_ids"], token_id_map) - prev_len
    latent = extract_latent_state(hidden[0], latent_pos)
    return (
        latent,
        output.loss,
        TrajectoryForwardState(
            past_key_values=output.past_key_values,
            rope_deltas=getattr(output, "rope_deltas", state.rope_deltas),
            prev_enc=enc_cur,
        ),
    )


def kv_forward_encodings_train(
    model,
    encodings: list[dict[str, torch.Tensor]],
    token_id_map: dict[str, int],
    device: torch.device,
    *,
    record_id: str = "",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not encodings:
        raise ValueError("kv_forward_encodings_train requires at least one encoding")
    assert_prefix_stable(encodings, record_id=record_id)
    _reset_model_rope_state(model)
    state: TrajectoryForwardState | None = None
    latents: list[torch.Tensor] = []
    losses: list[torch.Tensor] = []
    for enc in encodings:
        latent, step_loss, state = kv_forward_step_train(model, enc, state, device, token_id_map)
        latents.append(latent)
        if step_loss is not None:
            losses.append(step_loss)
    lm_loss = torch.stack(losses).mean() if losses else None
    return torch.stack(latents, dim=0), lm_loss


def kv_forward_step(
    model,
    enc_cur: dict[str, torch.Tensor],
    state: TrajectoryForwardState | None,
    device: torch.device,
    token_id_map: dict[str, int],
) -> tuple[torch.Tensor, TrajectoryForwardState]:
    pixel_values, image_grid_thw = vision_delta(None if state is None else state.prev_enc, enc_cur)
    if state is None or state.prev_enc is None or pixel_values is not None:
        hidden, output = _capture_last_hidden(model, _to_device(_batch_enc(enc_cur), device))
        latent_pos = find_last_latent_state_index(enc_cur["input_ids"], token_id_map)
        return extract_latent_state(hidden[0], latent_pos), TrajectoryForwardState(
            past_key_values=output.past_key_values,
            rope_deltas=getattr(output, "rope_deltas", None),
            prev_enc=enc_cur,
        )

    prev_enc = state.prev_enc
    prev_len = int(prev_enc["input_ids"].shape[0])
    cur_len = int(enc_cur["input_ids"].shape[0])
    prev_ids = prev_enc["input_ids"].tolist()
    cur_ids = enc_cur["input_ids"].tolist()
    if prev_ids != cur_ids[:prev_len]:
        raise ValueError("incremental KV requires stable prefix tokenization")

    delta_ids = enc_cur["input_ids"][prev_len:].unsqueeze(0).to(device)
    delta_mask = enc_cur["attention_mask"][prev_len:].unsqueeze(0).to(device)
    cache_position = torch.arange(prev_len, cur_len, device=device)
    kwargs: dict[str, Any] = dict(
        input_ids=delta_ids,
        attention_mask=delta_mask,
        past_key_values=state.past_key_values,
        cache_position=cache_position,
        use_cache=True,
    )

    hidden, output = _capture_last_hidden(model, kwargs)
    latent_pos = find_last_latent_state_index(enc_cur["input_ids"], token_id_map) - prev_len
    latent = extract_latent_state(hidden[0], latent_pos)
    return latent, TrajectoryForwardState(
        past_key_values=output.past_key_values,
        rope_deltas=getattr(output, "rope_deltas", state.rope_deltas),
        prev_enc=enc_cur,
    )


def kv_forward_trajectory(
    model,
    transitions: list[TransitionSample],
    processor: AutoProcessor,
    token_id_map: dict[str, int],
    device: torch.device,
    *,
    max_length: int,
) -> list[TrajectoryStepResult]:
    encodings = [
        encode_qwen_item(prefix_messages_with_images(sample), processor, max_length, include_labels=True)
        for sample in transitions
    ]
    record_id = transitions[0].record_id if transitions else ""
    assert_prefix_stable(encodings, record_id=record_id)

    _reset_model_rope_state(model)
    state: TrajectoryForwardState | None = None
    results: list[TrajectoryStepResult] = []
    for sample, enc in zip(transitions, encodings, strict=True):
        item = transition_collate_for_qwen([sample])[0]
        latent, state = kv_forward_step(model, enc, state, device, token_id_map)
        results.append(
            TrajectoryStepResult(
                step_index=sample.step_index,
                current_latent=latent,
                enc=enc,
                item=item,
            )
        )
    return results


def assert_trajectory_latents_equivalent(
    legacy: list[TrajectoryStepResult],
    packed: list[TrajectoryStepResult],
    *,
    atol: float = 1e-2,
    rtol: float = 0.0,
) -> float:
    if len(legacy) != len(packed):
        raise ValueError(f"step count mismatch: legacy={len(legacy)} packed={len(packed)}")
    max_diff = 0.0
    for leg, pack in zip(legacy, packed, strict=True):
        if leg.step_index != pack.step_index:
            raise ValueError(f"step index mismatch: {leg.step_index} vs {pack.step_index}")
        leg_latent = leg.current_latent.detach().cpu()
        pack_latent = pack.current_latent.detach().cpu()
        diff = (leg_latent - pack_latent).abs().max().item()
        max_diff = max(max_diff, float(diff))
        if not torch.allclose(leg_latent, pack_latent, rtol=rtol, atol=atol):
            raise AssertionError(
                f"step {leg.step_index}: latent max_abs_diff={diff:.6f} exceeds atol={atol}"
            )
    return max_diff
