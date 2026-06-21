#!/usr/bin/env python3
"""Minimal debug per 7-step plan: prefix vs full forward (synthetic 1/2 step only)."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.latent import (
    add_special_tokens,
    extract_latent_state,
    find_all_latent_state_indices,
    find_last_latent_state_index,
    special_token_ids,
)
from nimloth.training.common.qwen_batch import encode_qwen_item
from nimloth.training.sft2.packed_trajectory import _reset_model_rope_state
from nimloth.training.sft2.qwen_latent import _capture_last_hidden, extract_qwen_latents
from nimloth.training.sft2.trajectory_forward import _batch_enc
from nimloth.training.sft2.trajectory_once import encode_full_trajectory, find_step_latent_indices
from nimloth.wm.collate import prefix_messages_with_images
from nimloth.wm.dataset import TransitionSample


def _assistant_content(step: int) -> str:
    action = step % 8
    return (
        f"<think>t{step}</think>"
        f"<|latent_state|><|action_start|><|action_({action})|><|action_end|>"
    )


def build_text_steps(num_steps: int) -> list[TransitionSample]:
    """Build true trajectory prefixes: user0, assistant0, user1, assistant1, ..."""

    steps: list[TransitionSample] = []
    messages: list[dict[str, str]] = []
    for step in range(num_steps):
        messages.append({"role": "user", "content": f"observation text step {step}"})
        messages.append({"role": "assistant", "content": _assistant_content(step)})
        steps.append(
            TransitionSample(
                record_id="synthetic_text",
                step_index=step,
                prefix_messages=list(messages),
                prefix_image_paths=[],
                action_index=step % 8,
                current_image_path="",
                next_image_path="",
            )
        )
    return steps


def build_image_steps(num_steps: int, image_paths: list[str]) -> list[TransitionSample]:
    """Build true multi-image trajectory prefixes with alternating user/assistant turns."""

    steps: list[TransitionSample] = []
    messages: list[dict[str, str]] = []
    for step in range(num_steps):
        messages.append({"role": "user", "content": f"<image>\nobservation image step {step}"})
        messages.append({"role": "assistant", "content": _assistant_content(step)})
        steps.append(
            TransitionSample(
                record_id="synthetic_image",
                step_index=step,
                prefix_messages=list(messages),
                prefix_image_paths=[image_paths[i] for i in range(step + 1)],
                action_index=step % 8,
                current_image_path=image_paths[step],
                next_image_path=image_paths[step + 1] if step + 1 < len(image_paths) else image_paths[step],
            )
        )
    return steps


def _seq_len(enc: dict[str, torch.Tensor]) -> int:
    return int(enc["input_ids"].shape[0])


def _truncated(enc: dict[str, torch.Tensor], max_length: int) -> bool:
    return _seq_len(enc) >= max_length


def _prefix_tensor_equal(prefix_enc: dict[str, torch.Tensor], full_enc: dict[str, torch.Tensor], key: str) -> bool | None:
    if key not in prefix_enc or key not in full_enc:
        return None
    prefix_value = prefix_enc[key]
    full_value = full_enc[key]
    if not isinstance(prefix_value, torch.Tensor) or not isinstance(full_value, torch.Tensor):
        return None
    return bool(torch.equal(prefix_value, full_value[: prefix_value.shape[0]]))


def _position_ids_prefix_equal(model, prefix_enc: dict[str, torch.Tensor], full_enc: dict[str, torch.Tensor]) -> bool | None:
    if not hasattr(model, "model") or not hasattr(model.model, "get_rope_index"):
        return None
    try:
        prefix_batch = _batch_enc(prefix_enc)
        full_batch = _batch_enc(full_enc)
        prefix_pos, _ = model.model.get_rope_index(
            input_ids=prefix_batch["input_ids"],
            image_grid_thw=prefix_batch.get("image_grid_thw"),
            video_grid_thw=prefix_batch.get("video_grid_thw"),
            attention_mask=prefix_batch.get("attention_mask"),
        )
        full_pos, _ = model.model.get_rope_index(
            input_ids=full_batch["input_ids"],
            image_grid_thw=full_batch.get("image_grid_thw"),
            video_grid_thw=full_batch.get("video_grid_thw"),
            attention_mask=full_batch.get("attention_mask"),
        )
    except Exception as exc:  # pragma: no cover - debug-only diagnostic path
        print(f"[debug] position-id comparison failed: {exc}", file=sys.stderr, flush=True)
        return None
    prefix_len = prefix_enc["input_ids"].shape[0]
    return bool(torch.equal(prefix_pos[..., :prefix_len], full_pos[..., :prefix_len]))


@torch.no_grad()
def _forward_hidden_logits(
    model,
    enc: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    _reset_model_rope_state(model)
    batch = _batch_enc(enc)
    model_inputs = {k: v.to(device) for k, v in batch.items()}
    hidden, output = _capture_last_hidden(model, model_inputs)
    return hidden[0], output.logits[0]


def analyze_case(
    name: str,
    steps: list[TransitionSample],
    *,
    model,
    processor,
    token_id_map: dict[str, int],
    device: torch.device,
    max_length: int,
) -> dict[str, Any]:
    print(f"[debug] case={name} num_steps={len(steps)}", flush=True)

    full_enc, _ = encode_full_trajectory(steps, processor, max_length, token_id_map=token_id_map)
    full_ids = full_enc["input_ids"].tolist()
    full_step_latent_indices = find_step_latent_indices(
        steps, full_enc, processor, token_id_map, max_length
    )
    full_hidden, full_logits = _forward_hidden_logits(model, full_enc, device)

    step_reports: list[dict[str, Any]] = []
    for sample in steps:
        prefix_enc = encode_qwen_item(
            prefix_messages_with_images(sample), processor, max_length, include_labels=False
        )
        prefix_ids = prefix_enc["input_ids"].tolist()
        prefix_len = len(prefix_ids)
        ids_match = full_ids[:prefix_len] == prefix_ids
        first_mismatch: int | None = None
        if not ids_match:
            for i, (a, b) in enumerate(zip(full_ids[:prefix_len], prefix_ids, strict=True)):
                if a != b:
                    first_mismatch = i
                    break

        prefix_latent_pos = find_last_latent_state_index(prefix_enc["input_ids"], token_id_map)
        full_latent_indices = find_all_latent_state_indices(full_enc["input_ids"], token_id_map)
        full_latent_pos = full_step_latent_indices[sample.step_index]

        _reset_model_rope_state(model)
        prefix_latent, _ = extract_qwen_latents(model, _batch_enc(prefix_enc), token_id_map, device)
        prefix_latent = prefix_latent.squeeze(0)
        prefix_hidden, prefix_logits = _forward_hidden_logits(model, prefix_enc, device)

        step_reports.append(
            {
                "step_index": sample.step_index,
                "prefix_len": prefix_len,
                "full_len": len(full_ids),
                "prefix_truncated": _truncated(prefix_enc, max_length),
                "full_truncated": _truncated(full_enc, max_length),
                "prefix_input_ids_eq_full_prefix": ids_match,
                "first_token_mismatch_at": first_mismatch,
                "prefix_latent_pos": prefix_latent_pos,
                "full_latent_pos_by_step_index": full_latent_pos,
                "full_latent_token_count": len(full_latent_indices),
                "latent_pos_match": prefix_latent_pos == full_latent_pos,
                "image_grid_thw_eq_full_prefix": _prefix_tensor_equal(prefix_enc, full_enc, "image_grid_thw"),
                "pixel_values_eq_full_prefix": _prefix_tensor_equal(prefix_enc, full_enc, "pixel_values"),
                "position_ids_eq_full_prefix": _position_ids_prefix_equal(model, prefix_enc, full_enc),
                "latent_max_diff_prefix_vs_full_at_prefix_pos": (
                    float((prefix_latent - extract_latent_state(full_hidden, prefix_latent_pos)).abs().max().item())
                    if ids_match
                    else None
                ),
                "prefix_region_hidden_max_diff": (
                    float((prefix_hidden[:prefix_len] - full_hidden[:prefix_len]).abs().max().item())
                    if ids_match
                    else None
                ),
                "prefix_region_logits_max_diff": (
                    float((prefix_logits[:prefix_len] - full_logits[:prefix_len]).abs().max().item())
                    if ids_match
                    else None
                ),
            }
        )
        latent_diff = step_reports[-1]["latent_max_diff_prefix_vs_full_at_prefix_pos"]
        hidden_diff = step_reports[-1]["prefix_region_hidden_max_diff"]
        print(
            f"  step={sample.step_index} ids_ok={ids_match} mismatch_at={first_mismatch} "
            f"latent_diff={latent_diff} hidden_diff={hidden_diff}",
            flush=True,
        )

    return {
        "case": name,
        "num_steps": len(steps),
        "full_len": len(full_ids),
        "full_truncated": _truncated(full_enc, max_length),
        "steps": step_reports,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--max-pixels", type=int, default=602112)
    return ap.parse_args()


@torch.no_grad()
def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print(json.dumps({"error": "CUDA required"}), file=sys.stderr)
        return 2

    device = torch.device("cuda")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = args.max_pixels
    add_special_tokens(processor.tokenizer)
    token_id_map = special_token_ids(processor.tokenizer)

    print("[debug] loading model", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    model.to(device).eval()

    reports: list[dict[str, Any]] = []
    reports.append(
        analyze_case(
            "1step_text",
            build_text_steps(1),
            model=model,
            processor=processor,
            token_id_map=token_id_map,
            device=device,
            max_length=args.max_length,
        )
    )
    reports.append(
        analyze_case(
            "2step_text",
            build_text_steps(2),
            model=model,
            processor=processor,
            token_id_map=token_id_map,
            device=device,
            max_length=args.max_length,
        )
    )

    with tempfile.TemporaryDirectory() as tmp:
        image_paths = []
        for i in range(3):
            path = Path(tmp) / f"fake_{i}.png"
            Image.new("RGB", (224, 224), color=(i * 40, 100, 200)).save(path)
            image_paths.append(str(path))
        reports.append(
            analyze_case(
                "2step_fake_image",
                build_image_steps(2, image_paths),
                model=model,
                processor=processor,
                token_id_map=token_id_map,
                device=device,
                max_length=args.max_length,
            )
        )

    print(json.dumps(reports, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
