#!/usr/bin/env python3
"""Probe Qwen2.5-VL image feature return structure and prefix/full diffs."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.latent import add_special_tokens
from nimloth.training.common.qwen_batch import encode_qwen_item
from nimloth.training.sft2.trajectory_forward import _batch_enc
from nimloth.wm.collate import prefix_messages_with_images
from nimloth.wm.dataset import NUM_NAVIGATION_ACTIONS, TransitionSample, expand_record_transitions, load_jsonl_records


def _assistant_content(step: int) -> str:
    return (
        f"<think>t{step}</think><|latent_state|>"
        f"<|action_start|><|action_({step % NUM_NAVIGATION_ACTIONS})|><|action_end|>"
    )


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


def _describe_feature_output(x) -> dict[str, Any]:
    desc: dict[str, Any] = {"type": type(x).__name__}
    if hasattr(x, "pooler_output"):
        po = x.pooler_output
        desc["has_pooler_output"] = True
        desc["pooler_output_type"] = type(po).__name__
        if isinstance(po, (list, tuple)):
            desc["pooler_output_len"] = len(po)
            desc["pooler_output_shapes"] = [list(t.shape) for t in po]
        elif torch.is_tensor(po):
            desc["pooler_output_shape"] = list(po.shape)
    else:
        desc["has_pooler_output"] = False
        if isinstance(x, (list, tuple)):
            desc["len"] = len(x)
            desc["shapes"] = [list(t.shape) if torch.is_tensor(t) else str(type(t)) for t in x]
        elif torch.is_tensor(x):
            desc["shape"] = list(x.shape)
    return desc


def _flatten_feature_output(x) -> tuple[torch.Tensor, dict[str, Any]]:
    meta = _describe_feature_output(x)
    if hasattr(x, "pooler_output"):
        x = x.pooler_output
    if isinstance(x, (list, tuple)):
        flat = torch.cat(list(x), dim=0)
    elif torch.is_tensor(x):
        flat = x
    else:
        raise TypeError(f"unsupported image feature output type: {type(x)!r}")
    return flat.detach().cpu(), meta


def _prepare(model, enc: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    batch = {k: v.to(device) for k, v in _batch_enc(enc).items()}
    return batch["pixel_values"], batch["image_grid_thw"]


def analyze_case(name: str, steps: list[TransitionSample], model, processor, device: torch.device, max_length: int) -> dict[str, Any]:
    prefix_enc = encode_qwen_item(prefix_messages_with_images(steps[0]), processor, max_length, include_labels=False)
    full_enc = encode_qwen_item(prefix_messages_with_images(steps[-1]), processor, max_length, include_labels=False)
    prefix_pixel, prefix_grid = _prepare(model, prefix_enc, device)
    full_pixel, full_grid = _prepare(model, full_enc, device)

    prefix_raw = model.model.get_image_features(prefix_pixel, prefix_grid)
    full_raw = model.model.get_image_features(full_pixel, full_grid)
    prefix_flat, prefix_meta = _flatten_feature_output(prefix_raw)
    full_flat, full_meta = _flatten_feature_output(full_raw)

    prefix_n = prefix_flat.shape[0]
    return {
        "case": name,
        "prefix_pixel_shape": list(prefix_pixel.shape),
        "full_pixel_shape": list(full_pixel.shape),
        "prefix_grid_shape": list(prefix_grid.shape),
        "full_grid_shape": list(full_grid.shape),
        "prefix_output": prefix_meta,
        "full_output": full_meta,
        "flat_prefix_shape": list(prefix_flat.shape),
        "flat_full_shape": list(full_flat.shape),
        "flat_prefix_max_diff": float((prefix_flat - full_flat[:prefix_n]).abs().max().item()),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--train-jsonl", type=Path, default=None)
    ap.add_argument("--record-index", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=12000)
    ap.add_argument("--max-pixels", type=int, default=602112)
    ap.add_argument("--attn-implementation", default="sdpa")
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

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    model.to(device).eval()

    reports: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        record = _make_record(2, Path(tmp))
        reports.append(analyze_case("synthetic_2step_image_step0", expand_record_transitions(record), model, processor, device, args.max_length))

    if args.train_jsonl is not None:
        record = load_jsonl_records(args.train_jsonl, max_records=args.record_index + 1)[args.record_index]
        steps = expand_record_transitions(record)[:2]
        if len(steps) == 2:
            reports.append(analyze_case(f"real_{record.get('id')}_step0", steps, model, processor, device, args.max_length))

    print(json.dumps({"reports": reports}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
