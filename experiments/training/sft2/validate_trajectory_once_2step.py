#!/usr/bin/env python3
"""2-step GPU validation: legacy prefix vs trajectory-once after encoding fix."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.latent import add_special_tokens, special_token_ids
from nimloth.training.common.qwen_batch import encode_qwen_item
from nimloth.training.sft2.qwen_latent import extract_qwen_latents, reset_model_rope_state
from nimloth.training.sft2.qwen_monkey_patch import apply_qwen25vl_force_explicit_causal_mask_patch
from nimloth.training.sft2.trajectory_forward import _batch_enc
from nimloth.training.sft2.trajectory_once import (
    encode_full_trajectory,
    find_step_latent_indices,
    forward_trajectory_once,
)
from nimloth.wm.collate import prefix_messages_with_images
from nimloth.wm.dataset import NUM_NAVIGATION_ACTIONS, TransitionSample, expand_record_transitions, load_jsonl_records


def _make_text_steps(num_steps: int) -> list[TransitionSample]:
    steps: list[TransitionSample] = []
    for step in range(num_steps):
        prefix_messages: list[dict[str, str]] = []
        for turn in range(step + 1):
            prefix_messages.append({"role": "user", "content": f"observe text step {turn}"})
            prefix_messages.append(
                {
                    "role": "assistant",
                    "content": (
                        f"<think>t{turn}</think><|latent_state|>"
                        f"<|action_start|><|action_({turn % NUM_NAVIGATION_ACTIONS})|><|action_end|>"
                    ),
                }
            )
        steps.append(
            TransitionSample(
                record_id="synthetic_2step_text",
                step_index=step,
                prefix_messages=prefix_messages,
                prefix_image_paths=[],
                action_index=step % NUM_NAVIGATION_ACTIONS,
                current_image_path="",
                next_image_path="",
            )
        )
    return steps


def _make_record(num_steps: int, tmpdir: Path) -> dict:
    messages = [{"role": "system", "content": "sys"}]
    image_paths: list[str] = []
    action_indices: list[int] = []
    for step in range(num_steps):
        image_path = tmpdir / f"img_{step}.png"
        Image.new("RGB", (224, 224), color=(step * 40, 80, 160)).save(image_path)
        image_paths.append(str(image_path))
        messages.append({"role": "user", "content": f"observe <image> step {step}"})
        messages.append(
            {
                "role": "assistant",
                "content": (
                    f"<think>t{step}</think><|latent_state|>"
                    f"<|action_start|><|action_({step % NUM_NAVIGATION_ACTIONS})|><|action_end|>"
                ),
            }
        )
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


def _prefix_tensor_equal(prefix_enc: dict[str, torch.Tensor], full_enc: dict[str, torch.Tensor], key: str) -> bool | None:
    if key not in prefix_enc or key not in full_enc:
        return None
    prefix_value = prefix_enc[key]
    full_value = full_enc[key]
    if not isinstance(prefix_value, torch.Tensor) or not isinstance(full_value, torch.Tensor):
        return None
    return bool(torch.equal(prefix_value, full_value[: prefix_value.shape[0]]))


@torch.no_grad()
def _image_features_prefix_max_diff(
    model,
    prefix_enc: dict[str, torch.Tensor],
    full_enc: dict[str, torch.Tensor],
    device: torch.device,
) -> float | None:
    if "pixel_values" not in prefix_enc or "image_grid_thw" not in prefix_enc:
        return None
    if not hasattr(model, "model") or not hasattr(model.model, "get_image_features"):
        return None
    prefix_pixel = prefix_enc["pixel_values"].to(device)
    prefix_grid = prefix_enc["image_grid_thw"].to(device)
    full_pixel = full_enc["pixel_values"][: prefix_pixel.shape[0]].to(device)
    full_grid = full_enc["image_grid_thw"][: prefix_grid.shape[0]].to(device)
    prefix_features = torch.cat(model.model.get_image_features(prefix_pixel, prefix_grid), dim=0)
    full_features = torch.cat(model.model.get_image_features(full_pixel, full_grid), dim=0)
    return float((prefix_features - full_features).abs().max().item())


def _position_ids_prefix_equal(model, prefix_enc: dict[str, torch.Tensor], full_enc: dict[str, torch.Tensor]) -> bool | None:
    if not hasattr(model, "model") or not hasattr(model.model, "get_rope_index"):
        return None
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
    prefix_len = prefix_enc["input_ids"].shape[0]
    return bool(torch.equal(prefix_pos[..., :prefix_len], full_pos[..., :prefix_len]))


@torch.no_grad()
def compare_steps(model, processor, token_id_map, device, steps, max_length, case_name) -> dict:
    try:
        full_enc, _ = encode_full_trajectory(
            steps, processor, max_length, token_id_map=token_id_map
        )
    except ValueError as exc:
        return {
            "case": case_name,
            "num_steps": len(steps),
            "prefix_verify_passed": False,
            "error": str(exc),
            "passed": False,
        }

    legacy_latents = []
    alignment_reports = []
    latent_indices = find_step_latent_indices(steps, full_enc, processor, token_id_map, max_length)
    for sample in steps:
        enc = prefix_messages_with_images(sample)
        prefix_enc = encode_qwen_item(enc, processor, max_length, include_labels=False)
        prefix_len = int(prefix_enc["input_ids"].shape[0])
        alignment_reports.append(
            {
                "step_index": sample.step_index,
                "prefix_len": prefix_len,
                "latent_index": latent_indices[sample.step_index],
                "input_ids_eq_full_prefix": bool(
                    torch.equal(prefix_enc["input_ids"], full_enc["input_ids"][:prefix_len])
                ),
                "attention_mask_eq_full_prefix": _prefix_tensor_equal(prefix_enc, full_enc, "attention_mask"),
                "image_grid_thw_eq_full_prefix": _prefix_tensor_equal(prefix_enc, full_enc, "image_grid_thw"),
                "pixel_values_eq_full_prefix": _prefix_tensor_equal(prefix_enc, full_enc, "pixel_values"),
                "image_features_prefix_max_diff": _image_features_prefix_max_diff(
                    model, prefix_enc, full_enc, device
                ),
                "position_ids_eq_full_prefix": _position_ids_prefix_equal(model, prefix_enc, full_enc),
            }
        )
        reset_model_rope_state(model)
        latent, _ = extract_qwen_latents(model, _batch_enc(prefix_enc), token_id_map, device)
        legacy_latents.append(latent.squeeze(0))

    reset_model_rope_state(model)
    try:
        once = forward_trajectory_once(
            model,
            steps,
            processor,
            token_id_map,
            device,
            max_length=max_length,
            full_enc=full_enc,
        )
    except ValueError as exc:
        return {
            "case": case_name,
            "num_steps": len(steps),
            "prefix_verify_passed": True,
            "forward_error": str(exc),
            "passed": False,
        }
    diffs = [
        float((legacy - once.current_latents[i]).abs().max().item())
        for i, legacy in enumerate(legacy_latents)
    ]
    return {
        "case": case_name,
        "num_steps": len(steps),
        "prefix_verify_passed": True,
        "per_step_latent_max_diff": diffs,
        "alignment": alignment_reports,
        "max_latent_diff": max(diffs) if diffs else 0.0,
        "passed": max(diffs) <= 1e-2 if diffs else True,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--train-jsonl", type=Path, default=None)
    ap.add_argument("--record-index", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=12000)
    ap.add_argument("--max-pixels", type=int, default=602112)
    ap.add_argument("--attn-implementation", default="sdpa")
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
        print(json.dumps({"error": "CUDA required"}), file=sys.stderr)
        return 2

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

    reports: list[dict] = []
    reports.append(
        compare_steps(
            model,
            processor,
            token_id_map,
            device,
            _make_text_steps(2),
            args.max_length,
            "synthetic_2step_text",
        )
    )
    with tempfile.TemporaryDirectory() as tmp:
        record = _make_record(2, Path(tmp))
        steps = expand_record_transitions(record)
        reports.append(
            compare_steps(
                model,
                processor,
                token_id_map,
                device,
                steps,
                args.max_length,
                "synthetic_2step_image",
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
                )
            )

    passed = all(r["passed"] for r in reports)
    text_synthetic_passed = next(
        (r["passed"] for r in reports if r["case"] == "synthetic_2step_text"),
        False,
    )
    image_synthetic_passed = next(
        (r["passed"] for r in reports if r["case"] == "synthetic_2step_image"),
        False,
    )
    print(
        json.dumps(
            {
                "reports": reports,
                "passed": passed,
                "text_synthetic_passed": text_synthetic_passed,
                "image_synthetic_passed": image_synthetic_passed,
                "encoding_fix_verified": text_synthetic_passed,
            },
            indent=2,
        )
    )
    return 0 if text_synthetic_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
