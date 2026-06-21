#!/usr/bin/env python3
"""Validate per-image vision feature extraction as a correctness-preserving packed-forward patch."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.latent import add_special_tokens, extract_latent_state, find_last_latent_state_index, special_token_ids
from nimloth.training.common.qwen_batch import encode_qwen_item
from nimloth.training.sft2.qwen_latent import reset_model_rope_state
from nimloth.training.sft2.qwen_monkey_patch import apply_qwen25vl_force_explicit_causal_mask_patch
from nimloth.training.sft2.trajectory_once import encode_full_trajectory, find_step_latent_indices
from nimloth.training.sft2.trajectory_forward import _batch_enc
from nimloth.wm.collate import prefix_messages_with_images
from nimloth.wm.dataset import NUM_NAVIGATION_ACTIONS, TransitionSample, expand_record_transitions, load_jsonl_records


def _assistant_content(step: int) -> str:
    return (
        f"<think>t{step}</think><|latent_state|>"
        f"<|action_start|><|action_({step % NUM_NAVIGATION_ACTIONS})|><|action_end|>"
    )


def _make_text_steps(num_steps: int) -> list[TransitionSample]:
    steps: list[TransitionSample] = []
    messages: list[dict[str, str]] = []
    for step in range(num_steps):
        messages.append({"role": "user", "content": f"observe text step {step}"})
        messages.append({"role": "assistant", "content": _assistant_content(step)})
        steps.append(
            TransitionSample(
                record_id="synthetic_2step_text",
                step_index=step,
                prefix_messages=list(messages),
                prefix_image_paths=[],
                action_index=step % NUM_NAVIGATION_ACTIONS,
                current_image_path="",
                next_image_path="",
            )
        )
    return steps


def _make_record(num_steps: int, tmpdir: Path) -> dict[str, Any]:
    messages = [{"role": "system", "content": "sys"}]
    image_paths: list[str] = []
    action_indices: list[int] = []
    for step in range(num_steps):
        image_path = tmpdir / f"img_{step}.png"
        Image.new("RGB", (224, 224), color=(step * 40, 80, 160)).save(image_path)
        image_paths.append(str(image_path))
        messages.append({"role": "user", "content": f"observe <image> step {step}"})
        messages.append({"role": "assistant", "content": _assistant_content(step)})
        action_indices.append(step % NUM_NAVIGATION_ACTIONS)
    image_paths.append(str(tmpdir / f"img_{num_steps}.png"))
    Image.new("RGB", (224, 224), color=(120, 120, 120)).save(image_paths[-1])
    return {
        "id": "synthetic_2step",
        "split": "train",
        "success": True,
        "messages": messages,
        "image_paths": image_paths,
        "action_indices": action_indices,
        "reward": 1.0,
    }


def _pooler_output(x):
    if hasattr(x, "pooler_output"):
        return x.pooler_output
    return x


def _image_patch_split_sizes(image_grid_thw: torch.Tensor) -> list[int]:
    return [int(x) for x in image_grid_thw.prod(dim=-1).tolist()]


def _per_image_features(model, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
    chunks = torch.split(pixel_values, _image_patch_split_sizes(image_grid_thw), dim=0)
    rows: list[torch.Tensor] = []
    for chunk, grid in zip(chunks, image_grid_thw, strict=True):
        out = _pooler_output(model.model.get_image_features(chunk, grid.unsqueeze(0)))
        if isinstance(out, (tuple, list)):
            if len(out) != 1:
                raise RuntimeError(f"single-image get_image_features returned {len(out)} outputs")
            feat = out[0]
        else:
            feat = out
        rows.append(feat)
    return torch.cat(rows, dim=0)


def _compute_position_ids(model, batch: dict[str, torch.Tensor], inputs_embeds: torch.Tensor, position_id_mode: str):
    inner = model.model
    if hasattr(inner, "compute_3d_position_ids"):
        position_ids_3d = inner.compute_3d_position_ids(
            input_ids=batch.get("input_ids"),
            image_grid_thw=batch.get("image_grid_thw"),
            video_grid_thw=batch.get("video_grid_thw"),
            inputs_embeds=inputs_embeds,
            attention_mask=batch.get("attention_mask"),
            past_key_values=None,
            second_per_grid_ts=batch.get("second_per_grid_ts"),
            mm_token_type_ids=batch.get("mm_token_type_ids"),
        )
    else:
        position_ids_3d, _ = inner.get_rope_index(
            input_ids=batch["input_ids"],
            image_grid_thw=batch.get("image_grid_thw"),
            video_grid_thw=batch.get("video_grid_thw"),
            attention_mask=batch.get("attention_mask"),
        )
    if position_id_mode == "3d":
        return position_ids_3d
    if position_id_mode == "4d_text_prefix":
        batch_size, seq_len = batch["input_ids"].shape
        text_position_ids = torch.arange(seq_len, device=inputs_embeds.device, dtype=position_ids_3d.dtype).view(1, 1, seq_len).expand(1, batch_size, seq_len)
        return torch.cat([text_position_ids, position_ids_3d], dim=0)
    raise ValueError(f"Unknown position_id_mode: {position_id_mode}")


@torch.no_grad()
def prepare_per_image_vision_decoder_inputs(model, enc: dict[str, torch.Tensor], device: torch.device, position_id_mode: str) -> dict[str, torch.Tensor]:
    reset_model_rope_state(model)
    batch = {k: v.to(device) for k, v in _batch_enc(enc).items()}
    inner = model.model
    inputs_embeds = inner.get_input_embeddings()(batch["input_ids"])

    if batch.get("pixel_values") is not None:
        image_embeds = _per_image_features(model, batch["pixel_values"], batch["image_grid_thw"]).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        image_mask, _ = inner.get_placeholder_mask(
            batch["input_ids"], inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    position_ids = _compute_position_ids(model, batch, inputs_embeds, position_id_mode)
    return {
        "input_ids": batch["input_ids"],
        "attention_mask": batch.get("attention_mask"),
        "inputs_embeds": inputs_embeds,
        "position_ids": position_ids,
    }


@torch.no_grad()
def forward_last_hidden_per_image_vision(model, enc: dict[str, torch.Tensor], device: torch.device, position_id_mode: str = "3d") -> torch.Tensor:
    decoder_inputs = prepare_per_image_vision_decoder_inputs(model, enc, device, position_id_mode)
    inner = model.model
    outputs = inner.language_model(
        input_ids=None,
        position_ids=decoder_inputs["position_ids"],
        attention_mask=decoder_inputs.get("attention_mask"),
        inputs_embeds=decoder_inputs["inputs_embeds"],
        use_cache=False,
    )
    return outputs.last_hidden_state


@torch.no_grad()
def compare_steps(model, processor, token_id_map, device, steps: list[TransitionSample], max_length: int, case_name: str, position_id_mode: str) -> dict[str, Any]:
    try:
        full_enc, _ = encode_full_trajectory(steps, processor, max_length, token_id_map=token_id_map)
    except ValueError as exc:
        return {"case": case_name, "prefix_verify_passed": False, "error": str(exc), "passed": False}

    full_dec = prepare_per_image_vision_decoder_inputs(model, full_enc, device, position_id_mode)
    full_hidden = model.model.language_model(
        input_ids=None,
        position_ids=full_dec["position_ids"],
        attention_mask=full_dec.get("attention_mask"),
        inputs_embeds=full_dec["inputs_embeds"],
        use_cache=False,
    ).last_hidden_state[0]
    latent_indices = find_step_latent_indices(steps, full_enc, processor, token_id_map, max_length)
    diffs: list[float] = []
    hidden_region_diffs: list[float] = []
    embed_region_diffs: list[float] = []
    position_region_diffs: list[float] = []
    latent_positions: list[dict[str, int]] = []
    for sample in steps:
        prefix_enc = encode_qwen_item(prefix_messages_with_images(sample), processor, max_length, include_labels=False)
        prefix_len = int(prefix_enc["input_ids"].shape[0])
        prefix_dec = prepare_per_image_vision_decoder_inputs(model, prefix_enc, device, position_id_mode)
        prefix_hidden = model.model.language_model(
            input_ids=None,
            position_ids=prefix_dec["position_ids"],
            attention_mask=prefix_dec.get("attention_mask"),
            inputs_embeds=prefix_dec["inputs_embeds"],
            use_cache=False,
        ).last_hidden_state[0]
        prefix_latent_pos = find_last_latent_state_index(prefix_enc["input_ids"], token_id_map)
        full_latent_pos = latent_indices[sample.step_index]
        prefix_latent = extract_latent_state(prefix_hidden.unsqueeze(0), prefix_latent_pos)
        full_latent = extract_latent_state(full_hidden.unsqueeze(0), full_latent_pos)
        diffs.append(float((prefix_latent - full_latent).abs().max().item()))
        hidden_region_diffs.append(float((prefix_hidden[:prefix_len] - full_hidden[:prefix_len]).abs().max().item()))
        embed_region_diffs.append(
            float((prefix_dec["inputs_embeds"][0, :prefix_len] - full_dec["inputs_embeds"][0, :prefix_len]).abs().max().item())
        )
        position_region_diffs.append(
            float((prefix_dec["position_ids"][..., :prefix_len] - full_dec["position_ids"][..., :prefix_len]).abs().max().item())
        )
        latent_positions.append({"step_index": sample.step_index, "prefix_latent_pos": prefix_latent_pos, "full_latent_pos": full_latent_pos})
    max_diff = max(diffs) if diffs else 0.0
    return {
        "case": case_name,
        "position_id_mode": position_id_mode,
        "num_steps": len(steps),
        "prefix_verify_passed": True,
        "per_step_latent_max_diff": diffs,
        "per_step_prefix_region_hidden_max_diff": hidden_region_diffs,
        "per_step_inputs_embeds_prefix_max_diff": embed_region_diffs,
        "per_step_position_ids_prefix_max_diff": position_region_diffs,
        "latent_positions": latent_positions,
        "max_latent_diff": max_diff,
        "passed": max_diff <= 1e-2,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--train-jsonl", type=Path, default=None)
    ap.add_argument("--record-index", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=12000)
    ap.add_argument("--max-pixels", type=int, default=602112)
    ap.add_argument("--attn-implementation", default="sdpa")
    ap.add_argument("--position-id-mode", choices=("3d", "4d_text_prefix"), default="3d")
    ap.add_argument(
        "--qwen-monkey-patch",
        choices=("none", "force_explicit_causal_mask"),
        default="none",
    )
    return ap.parse_args()


@torch.no_grad()
def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit(json.dumps({"error": "CUDA required"}))

    device = torch.device("cuda")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = args.max_pixels
    add_special_tokens(processor.tokenizer)
    token_id_map = special_token_ids(processor.tokenizer)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    if args.qwen_monkey_patch == "force_explicit_causal_mask":
        applied = apply_qwen25vl_force_explicit_causal_mask_patch(model)
        print(
            json.dumps(
                {
                    "qwen_monkey_patch": args.qwen_monkey_patch,
                    "attn_implementation": args.attn_implementation,
                    "applied": applied,
                }
            ),
            flush=True,
        )
    model.to(device).eval()

    reports = [
        compare_steps(model, processor, token_id_map, device, _make_text_steps(2), args.max_length, "synthetic_2step_text", args.position_id_mode)
    ]
    with tempfile.TemporaryDirectory() as tmp:
        record = _make_record(2, Path(tmp))
        reports.append(
            compare_steps(
                model,
                processor,
                token_id_map,
                device,
                expand_record_transitions(record),
                args.max_length,
                "synthetic_2step_image",
                args.position_id_mode,
            )
        )
    if args.train_jsonl is not None:
        record = load_jsonl_records(args.train_jsonl, max_records=args.record_index + 1)[args.record_index]
        steps = expand_record_transitions(record)[:2]
        if len(steps) == 2:
            reports.append(
                compare_steps(
                    model,
                    processor,
                    token_id_map,
                    device,
                    steps,
                    args.max_length,
                    f"real_record_{record.get('id')}_2step",
                    args.position_id_mode,
                )
            )

    print(json.dumps({"reports": reports, "passed": all(r["passed"] for r in reports)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
