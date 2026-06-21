#!/usr/bin/env python3
"""Validate Qwen2.5-VL KV-cache incremental semantics under different chunking."""

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
from nimloth.training.sft2.trajectory_forward import _batch_enc
from nimloth.wm.collate import prefix_messages_with_images
from nimloth.wm.dataset import NUM_NAVIGATION_ACTIONS, TransitionSample, expand_record_transitions, load_jsonl_records


def _assistant_content(step: int) -> str:
    return f"<think>t{step}</think><|latent_state|><|action_start|><|action_({step % NUM_NAVIGATION_ACTIONS})|><|action_end|>"


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
    return {"id": "synthetic_2step", "messages": messages, "image_paths": image_paths, "action_indices": action_indices, "reward": 1.0}


def _batch(enc: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in _batch_enc(enc).items()}


def _position_ids(model, enc: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    """Return generation-style [4, batch, seq] position ids for KV/FA2 paths."""

    b = _batch(enc, device)
    if hasattr(model.model, "compute_3d_position_ids"):
        emb = model.model.get_input_embeddings()(b["input_ids"])
        vision_pos = model.model.compute_3d_position_ids(
            input_ids=b["input_ids"], image_grid_thw=b.get("image_grid_thw"), video_grid_thw=b.get("video_grid_thw"),
            inputs_embeds=emb, attention_mask=b.get("attention_mask"), past_key_values=None,
            second_per_grid_ts=b.get("second_per_grid_ts"), mm_token_type_ids=b.get("mm_token_type_ids"),
        )
    else:
        vision_pos, _ = model.model.get_rope_index(input_ids=b["input_ids"], image_grid_thw=b.get("image_grid_thw"), video_grid_thw=b.get("video_grid_thw"), attention_mask=b.get("attention_mask"))
    text_pos = b["attention_mask"].long().cumsum(-1) - 1
    text_pos = text_pos.masked_fill(b["attention_mask"] == 0, 0)
    return torch.cat([text_pos.unsqueeze(0), vision_pos], dim=0)


def _new_image_payload(prev_enc: dict[str, torch.Tensor], cur_enc: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if "pixel_values" not in cur_enc:
        return {}
    prev_images = int(prev_enc.get("image_grid_thw", torch.empty(0, 3)).shape[0])
    prev_patches = int(prev_enc.get("image_grid_thw", torch.empty(0, 3)).prod(dim=-1).sum().item()) if prev_images else 0
    return {"pixel_values": cur_enc["pixel_values"][prev_patches:], "image_grid_thw": cur_enc["image_grid_thw"][prev_images:]}


def _forward_chunk(model, enc, pos_full, start: int, end: int, past, image_payload, device: torch.device):
    model_inputs: dict[str, torch.Tensor] = {
        "input_ids": enc["input_ids"][start:end].unsqueeze(0).to(device),
        "attention_mask": enc["attention_mask"][:end].unsqueeze(0).to(device),
        "position_ids": pos_full[..., start:end].to(device),
    }
    model_inputs.update({k: v.to(device) for k, v in image_payload.items()})
    return model(**model_inputs, past_key_values=past, use_cache=True, output_hidden_states=True, return_dict=True)


@torch.no_grad()
def run_case(model, processor, token_id_map, device, steps, max_length: int, case_name: str) -> dict[str, Any]:
    enc0 = encode_qwen_item(prefix_messages_with_images(steps[0]), processor, max_length, include_labels=False)
    enc1 = encode_qwen_item(prefix_messages_with_images(steps[1]), processor, max_length, include_labels=False)
    len0, len1 = int(enc0["input_ids"].shape[0]), int(enc1["input_ids"].shape[0])
    latent1 = find_last_latent_state_index(enc1["input_ids"], token_id_map)
    split = latent1
    if split <= len0 or split >= len1:
        raise RuntimeError(f"bad split len0={len0} latent1={latent1} len1={len1}")
    image_payload = _new_image_payload(enc0, enc1)
    pos0 = _position_ids(model, enc0, device)
    pos1 = _position_ids(model, enc1, device)

    reset_model_rope_state(model)
    out0 = _forward_chunk(model, enc0, pos0, 0, len0, None, {k: v for k, v in enc0.items() if k in {"pixel_values", "image_grid_thw"}}, device)
    latent0 = extract_latent_state(out0.hidden_states[-1], find_last_latent_state_index(enc0["input_ids"], token_id_map)).detach().cpu()
    out1_one = _forward_chunk(model, enc1, pos1, len0, len1, out0.past_key_values, image_payload, device)
    latent_one = extract_latent_state(out1_one.hidden_states[-1], latent1 - len0).detach().cpu()

    reset_model_rope_state(model)
    out0b = _forward_chunk(model, enc0, pos0, 0, len0, None, {k: v for k, v in enc0.items() if k in {"pixel_values", "image_grid_thw"}}, device)
    out_mid = _forward_chunk(model, enc1, pos1, len0, split, out0b.past_key_values, image_payload, device)
    out1_two = _forward_chunk(model, enc1, pos1, split, len1, out_mid.past_key_values, {}, device)
    latent_two = extract_latent_state(out1_two.hidden_states[-1], 0).detach().cpu()
    latent0b = extract_latent_state(out0b.hidden_states[-1], find_last_latent_state_index(enc0["input_ids"], token_id_map)).detach().cpu()

    return {
        "case": case_name,
        "len0": len0,
        "len1": len1,
        "split_at_latent": split,
        "one_chunk_past_len": int(out1_one.past_key_values.get_seq_length()),
        "two_chunk_mid_past_len": int(out_mid.past_key_values.get_seq_length()),
        "two_chunk_final_past_len": int(out1_two.past_key_values.get_seq_length()),
        "latent0_repeat_diff": float((latent0 - latent0b).abs().max().item()),
        "latent1_one_vs_two_chunk_diff": float((latent_one - latent_two).abs().max().item()),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--train-jsonl", type=Path, default=None)
    ap.add_argument("--record-index", type=int, default=0)
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
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attn_implementation, trust_remote_code=True)
    model.resize_token_embeddings(len(processor.tokenizer))
    model.to(device).eval()
    reports = []
    with tempfile.TemporaryDirectory() as tmp:
        reports.append(run_case(model, processor, token_id_map, device, expand_record_transitions(_make_record(2, Path(tmp)))[:2], args.max_length, "synthetic_image"))
    if args.train_jsonl is not None:
        record = load_jsonl_records(args.train_jsonl, max_records=args.record_index + 1)[args.record_index]
        reports.append(run_case(model, processor, token_id_map, device, expand_record_transitions(record)[:2], args.max_length, f"real_{record.get('id')}"))
    print(json.dumps({"reports": reports}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
