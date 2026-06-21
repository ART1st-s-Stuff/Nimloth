#!/usr/bin/env python3
"""Smoke-test Qwen2.5-VL incremental KV-cache trajectory semantics."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.latent import add_special_tokens, extract_latent_state, find_last_latent_state_index, special_token_ids
from nimloth.training.common.qwen_batch import encode_qwen_item
from nimloth.training.sft2.qwen_latent import reset_model_rope_state
from nimloth.training.sft2.trajectory_forward import _batch_enc
from nimloth.training.sft2.trajectory_once import find_step_latent_indices
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


def _to_device_batch(enc: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in _batch_enc(enc).items()}


def _prefix_position_ids(model, enc: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    batch = _to_device_batch(enc, device)
    if hasattr(model.model, "compute_3d_position_ids"):
        inputs_embeds = model.model.get_input_embeddings()(batch["input_ids"])
        return model.model.compute_3d_position_ids(
            input_ids=batch["input_ids"],
            image_grid_thw=batch.get("image_grid_thw"),
            video_grid_thw=batch.get("video_grid_thw"),
            inputs_embeds=inputs_embeds,
            attention_mask=batch.get("attention_mask"),
            past_key_values=None,
            second_per_grid_ts=batch.get("second_per_grid_ts"),
            mm_token_type_ids=batch.get("mm_token_type_ids"),
        )
    pos, _ = model.model.get_rope_index(
        input_ids=batch["input_ids"],
        image_grid_thw=batch.get("image_grid_thw"),
        video_grid_thw=batch.get("video_grid_thw"),
        attention_mask=batch.get("attention_mask"),
    )
    return pos


def _new_image_slice(prev_enc: dict[str, torch.Tensor] | None, cur_enc: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if "pixel_values" not in cur_enc:
        return {}
    if prev_enc is None or "image_grid_thw" not in prev_enc:
        prev_images = 0
        prev_patches = 0
    else:
        prev_images = int(prev_enc["image_grid_thw"].shape[0])
        prev_patches = int(prev_enc["image_grid_thw"].prod(dim=-1).sum().item())
    return {
        "pixel_values": cur_enc["pixel_values"][prev_patches:],
        "image_grid_thw": cur_enc["image_grid_thw"][prev_images:],
    }


@torch.no_grad()
def _prefix_latent_reference(model, enc: dict[str, torch.Tensor], token_id_map: dict[str, int], device: torch.device) -> torch.Tensor:
    reset_model_rope_state(model)
    out = model(**_to_device_batch(enc, device), output_hidden_states=False, return_dict=True)
    pos = find_last_latent_state_index(enc["input_ids"], token_id_map)
    return extract_latent_state(out.hidden_states[-1] if getattr(out, "hidden_states", None) else out.logits.new_empty(0), pos)


@torch.no_grad()
def _forward_prefix_hidden(model, enc: dict[str, torch.Tensor], device: torch.device):
    reset_model_rope_state(model)
    out = model(**_to_device_batch(enc, device), output_hidden_states=True, return_dict=True, use_cache=False)
    return out.hidden_states[-1][0]


@torch.no_grad()
def run_case(model, processor, token_id_map, device: torch.device, steps: list[TransitionSample], max_length: int, case_name: str) -> dict[str, Any]:
    prefix_encs = [encode_qwen_item(prefix_messages_with_images(s), processor, max_length, include_labels=False) for s in steps]
    prev_enc = None
    prev_len = 0
    past = None
    incremental_latents: list[torch.Tensor] = []
    per_prefix_latents: list[torch.Tensor] = []
    reports: list[dict[str, Any]] = []
    reset_model_rope_state(model)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for step_idx, enc in enumerate(prefix_encs):
        cur_len = int(enc["input_ids"].shape[0])
        suffix_ids = enc["input_ids"][prev_len:]
        suffix_mask = enc["attention_mask"][:cur_len]
        pos_full = _prefix_position_ids(model, enc, device)
        pos_suffix = pos_full[..., prev_len:cur_len]
        model_inputs: dict[str, torch.Tensor] = {
            "input_ids": suffix_ids.unsqueeze(0).to(device),
            "attention_mask": suffix_mask.unsqueeze(0).to(device),
            "position_ids": pos_suffix.to(device),
        }
        model_inputs.update({k: v.unsqueeze(0).to(device) if v.ndim == 1 else v.to(device) for k, v in _new_image_slice(prev_enc, enc).items()})
        out = model(**model_inputs, past_key_values=past, use_cache=True, output_hidden_states=True, return_dict=True)
        past = out.past_key_values
        full_latent_pos = find_last_latent_state_index(enc["input_ids"], token_id_map)
        suffix_latent_pos = full_latent_pos - prev_len
        inc_latent = extract_latent_state(out.hidden_states[-1], suffix_latent_pos)
        incremental_latents.append(inc_latent.detach().cpu())

        ref_hidden = _forward_prefix_hidden(model, enc, device)
        ref_latent = extract_latent_state(ref_hidden.unsqueeze(0), full_latent_pos)
        per_prefix_latents.append(ref_latent.detach().cpu())
        reports.append(
            {
                "step_index": step_idx,
                "prev_len": prev_len,
                "cur_len": cur_len,
                "suffix_len": int(suffix_ids.shape[0]),
                "suffix_latent_pos": int(suffix_latent_pos),
                "past_seq_len": int(past.get_seq_length()) if hasattr(past, "get_seq_length") else None,
                "latent_diff_vs_recompute": float((inc_latent.detach().cpu() - ref_latent.detach().cpu()).abs().max().item()),
            }
        )
        prev_enc = enc
        prev_len = cur_len
    elapsed = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3) if torch.cuda.is_available() else None
    diffs = [r["latent_diff_vs_recompute"] for r in reports]
    return {
        "case": case_name,
        "num_steps": len(steps),
        "steps": reports,
        "max_diff_vs_per_prefix_recompute_reference": max(diffs) if diffs else 0.0,
        "elapsed_sec": elapsed,
        "peak_memory_gb": peak_gb,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--train-jsonl", type=Path, default=None)
    ap.add_argument("--record-index", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=12000)
    ap.add_argument("--max-pixels", type=int, default=602112)
    ap.add_argument("--attn-implementation", default="flash_attention_2")
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
    model.to(device).eval()

    reports = []
    with tempfile.TemporaryDirectory() as tmp:
        reports.append(run_case(model, processor, token_id_map, device, expand_record_transitions(_make_record(args.max_steps, Path(tmp)))[: args.max_steps], args.max_length, "synthetic_image"))
    if args.train_jsonl is not None:
        record = load_jsonl_records(args.train_jsonl, max_records=args.record_index + 1)[args.record_index]
        steps = expand_record_transitions(record)[: args.max_steps]
        reports.append(run_case(model, processor, token_id_map, device, steps, args.max_length, f"real_{record.get('id')}"))
    print(json.dumps({"reports": reports}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
