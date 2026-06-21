#!/usr/bin/env python3
"""Probe where Qwen2.5-VL prefix/full multimodal inputs_embeds first diverge."""

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


def _pooler_output(x):
    if hasattr(x, "pooler_output"):
        return x.pooler_output
    return x


def _first_true_positions(mask: torch.Tensor) -> list[int]:
    if mask.ndim == 3:
        flat = mask.any(dim=-1)
    else:
        flat = mask
    return [int(i) for i in torch.nonzero(flat[0], as_tuple=False).flatten().tolist()]


def _row_max_diff_positions(a: torch.Tensor, b: torch.Tensor, prefix_len: int, threshold: float = 0.0) -> dict[str, Any]:
    diff = (a[:, :prefix_len, :] - b[:, :prefix_len, :]).abs().max(dim=-1).values[0]
    nz = torch.nonzero(diff > threshold, as_tuple=False).flatten().tolist()
    return {
        "max_diff": float(diff.max().item()),
        "num_diff_positions": len(nz),
        "first_diff_positions": [int(x) for x in nz[:16]],
    }


def _prepare_parts(model, enc: dict[str, torch.Tensor], device: torch.device) -> dict[str, Any]:
    batch = {k: v.to(device) for k, v in _batch_enc(enc).items()}
    inner = model.model
    text_embeds = inner.get_input_embeddings()(batch["input_ids"])

    out: dict[str, Any] = {
        "input_ids": batch["input_ids"].detach().cpu(),
        "text_embeds": text_embeds.detach().cpu(),
    }

    if batch.get("pixel_values") is not None:
        image_features = _pooler_output(inner.get_image_features(batch["pixel_values"], batch.get("image_grid_thw")))
        image_embeds = torch.cat(image_features, dim=0).to(text_embeds.device, text_embeds.dtype)
        image_mask, _ = inner.get_placeholder_mask(
            batch["input_ids"],
            inputs_embeds=text_embeds,
            image_features=image_embeds,
        )
        scattered = text_embeds.masked_scatter(image_mask, image_embeds)
        out.update(
            {
                "image_embeds": image_embeds.detach().cpu(),
                "image_mask": image_mask.detach().cpu(),
                "scattered_embeds": scattered.detach().cpu(),
            }
        )
    return out


def analyze_case(name: str, steps: list[TransitionSample], model, processor, device: torch.device, max_length: int) -> dict[str, Any]:
    prefix_enc = encode_qwen_item(prefix_messages_with_images(steps[0]), processor, max_length, include_labels=False)
    full_enc = encode_qwen_item(prefix_messages_with_images(steps[-1]), processor, max_length, include_labels=False)
    prefix_len = int(prefix_enc["input_ids"].shape[0])

    prefix_parts = _prepare_parts(model, prefix_enc, device)
    full_parts = _prepare_parts(model, full_enc, device)

    result = {
        "case": name,
        "prefix_len": prefix_len,
        "full_len": int(full_enc["input_ids"].shape[0]),
        "input_ids_eq_full_prefix": bool(torch.equal(prefix_parts["input_ids"], full_parts["input_ids"][:, :prefix_len])),
        "text_embeds": _row_max_diff_positions(prefix_parts["text_embeds"], full_parts["text_embeds"], prefix_len),
    }

    if "image_embeds" in prefix_parts and "image_embeds" in full_parts:
        prefix_img_n = prefix_parts["image_embeds"].shape[0]
        result["image_embeds_prefix_max_diff"] = float(
            (prefix_parts["image_embeds"] - full_parts["image_embeds"][:prefix_img_n]).abs().max().item()
        )
        result["image_mask_prefix_equal"] = bool(
            torch.equal(prefix_parts["image_mask"], full_parts["image_mask"][:, :prefix_len, :])
        )
        result["prefix_image_mask_positions"] = _first_true_positions(prefix_parts["image_mask"])
        result["full_image_mask_prefix_positions"] = _first_true_positions(full_parts["image_mask"][:, :prefix_len, :])
        result["scattered_embeds"] = _row_max_diff_positions(
            prefix_parts["scattered_embeds"],
            full_parts["scattered_embeds"],
            prefix_len,
        )
    return result


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
