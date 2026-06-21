#!/usr/bin/env python3
"""Probe which Qwen2.5-VL vision stage first diverges for the same prefix image."""

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


def _capture_vision(model, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> dict[str, Any]:
    visual = model.model.visual
    captured: dict[str, Any] = {}
    handles = []

    def patch_hook(_mod, _inp, out):
        captured["patch_embed"] = out.detach().cpu()

    def merger_hook(_mod, _inp, out):
        captured["merger"] = out.detach().cpu()

    def block_hook(idx: int):
        def _hook(_mod, _inp, out):
            captured.setdefault("blocks", {})[idx] = (out[0] if isinstance(out, tuple) else out).detach().cpu()
        return _hook

    handles.append(visual.patch_embed.register_forward_hook(patch_hook))
    handles.append(visual.merger.register_forward_hook(merger_hook))
    for idx, blk in enumerate(visual.blocks):
        handles.append(blk.register_forward_hook(block_hook(idx)))
    try:
        out = model.model.get_image_features(pixel_values, image_grid_thw)
    finally:
        for h in handles:
            h.remove()

    if hasattr(out, "pooler_output"):
        pooler = out.pooler_output
    else:
        pooler = out
    captured["pooler_output"] = [t.detach().cpu() for t in pooler]
    return captured


def _prepare(enc: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    batch = {k: v.to(device) for k, v in _batch_enc(enc).items()}
    return batch["pixel_values"], batch["image_grid_thw"]


def _max_diff_prefix(a: torch.Tensor, b: torch.Tensor, prefix_len: int) -> float:
    return float((a[:prefix_len] - b[:prefix_len]).abs().max().item())


def analyze_case(name: str, steps: list[TransitionSample], model, processor, device: torch.device, max_length: int) -> dict[str, Any]:
    prefix_enc = encode_qwen_item(prefix_messages_with_images(steps[0]), processor, max_length, include_labels=False)
    full_enc = encode_qwen_item(prefix_messages_with_images(steps[-1]), processor, max_length, include_labels=False)
    prefix_pixel, prefix_grid = _prepare(prefix_enc, device)
    full_pixel, full_grid = _prepare(full_enc, device)

    prefix_cap = _capture_vision(model, prefix_pixel, prefix_grid)
    full_cap = _capture_vision(model, full_pixel, full_grid)

    prefix_seq = int(prefix_cap["patch_embed"].shape[0])
    report = {
        "case": name,
        "prefix_patch_seq": prefix_seq,
        "full_patch_seq": int(full_cap["patch_embed"].shape[0]),
        "patch_embed_prefix_max_diff": _max_diff_prefix(prefix_cap["patch_embed"], full_cap["patch_embed"], prefix_seq),
        "blocks": [],
        "pooler_first_image_max_diff": float(
            (prefix_cap["pooler_output"][0] - full_cap["pooler_output"][0]).abs().max().item()
        ),
    }
    first_nonzero: int | None = None
    for idx in sorted(prefix_cap["blocks"]):
        diff = _max_diff_prefix(prefix_cap["blocks"][idx], full_cap["blocks"][idx], prefix_seq)
        report["blocks"].append(diff)
        if first_nonzero is None and diff > 1e-6:
            first_nonzero = idx
    report["first_nonzero_block"] = first_nonzero
    report["merger_prefix_region_max_diff"] = _max_diff_prefix(
        prefix_cap["merger"],
        full_cap["merger"],
        prefix_cap["merger"].shape[0],
    )
    return report


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
