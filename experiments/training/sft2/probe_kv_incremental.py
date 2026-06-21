#!/usr/bin/env python3
"""Probe Qwen2.5-VL incremental KV forward vs full prefix forward."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.latent import add_special_tokens, extract_latent_state, find_last_latent_state_index, special_token_ids
from nimloth.training.common.qwen_batch import encode_qwen_item
from nimloth.training.sft2.qwen_latent import _capture_last_hidden, forward_qwen_last_hidden
from nimloth.training.sft2.trajectory_forward import _batch_enc, _prefix_latent
from nimloth.wm.collate import prefix_messages_with_images
from nimloth.wm.dataset import expand_record_transitions, load_jsonl_records


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--train-jsonl", type=Path, required=True)
    ap.add_argument("--record-index", type=int, default=0)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--max-length", type=int, default=12000)
    return ap.parse_args()


def _vision_delta(enc_prev: dict, enc_cur: dict) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if "pixel_values" not in enc_cur:
        return None, None
    cur_pv = enc_cur["pixel_values"]
    prev_pv = enc_prev.get("pixel_values")
    prev_rows = 0 if prev_pv is None else int(prev_pv.shape[0])
    if cur_pv.shape[0] <= prev_rows:
        return None, None
    delta_pv = cur_pv[prev_rows:]
    if "image_grid_thw" not in enc_cur:
        return delta_pv, None
    prev_images = 0 if "image_grid_thw" not in enc_prev else int(enc_prev["image_grid_thw"].shape[0])
    delta_grid = enc_cur["image_grid_thw"][prev_images:]
    return delta_pv, delta_grid


@torch.no_grad()
def main() -> int:
    args = parse_args()
    device = torch.device("cuda")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = 602112
    add_special_tokens(processor.tokenizer)
    token_id_map = special_token_ids(processor.tokenizer)

    record = load_jsonl_records(args.train_jsonl, max_records=args.record_index + 1)[args.record_index]
    transitions = expand_record_transitions(record)
    step = args.step
    sample_prev = transitions[step - 1] if step > 0 else None
    sample = transitions[step]

    enc_prev = (
        encode_qwen_item(prefix_messages_with_images(sample_prev), processor, args.max_length, include_labels=False)
        if sample_prev is not None
        else None
    )
    enc_cur = encode_qwen_item(prefix_messages_with_images(sample), processor, args.max_length, include_labels=False)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    model.to(device).eval()

    legacy = _prefix_latent(model, sample, processor, token_id_map, device, args.max_length)

    if step == 0:
        hidden = forward_qwen_last_hidden(model, _batch_enc(enc_cur), device)
        pos = find_last_latent_state_index(enc_cur["input_ids"], token_id_map)
        kv_latent = extract_latent_state(hidden[0], pos)
        print("step0", float((legacy - kv_latent).abs().max()))
        return 0

    assert enc_prev is not None
    prev_ids = enc_prev["input_ids"].tolist()
    cur_ids = enc_cur["input_ids"].tolist()
    assert prev_ids == cur_ids[: len(prev_ids)]

    out0 = model(
        **_move_batch(_batch_enc(enc_prev), device),
        use_cache=True,
        return_dict=True,
    )
    past = out0.past_key_values
    rope_deltas = getattr(out0, "rope_deltas", None)
    prev_len = enc_prev["input_ids"].shape[0]
    cur_len = enc_cur["input_ids"].shape[0]
    delta_ids = enc_cur["input_ids"][prev_len:].unsqueeze(0).to(device)
    delta_mask = enc_cur["attention_mask"][prev_len:].unsqueeze(0).to(device)
    cache_position = torch.arange(prev_len, cur_len, device=device)
    pixel_values, image_grid_thw = _vision_delta(enc_prev, enc_cur)
    if step == 1:
        print(
            "debug",
            {
                "prev_len": prev_len,
                "delta_len": int(delta_ids.shape[1]),
                "delta_images": None if image_grid_thw is None else int(image_grid_thw.shape[0]),
                "delta_pv_rows": None if pixel_values is None else int(pixel_values.shape[0]),
            },
        )
    kwargs = dict(
        input_ids=delta_ids,
        attention_mask=delta_mask,
        past_key_values=past,
        cache_position=cache_position,
        use_cache=True,
    )
    if rope_deltas is not None:
        kwargs["rope_deltas"] = rope_deltas
    if pixel_values is not None:
        kwargs["pixel_values"] = pixel_values.to(device)
    if image_grid_thw is not None:
        kwargs["image_grid_thw"] = image_grid_thw.to(device)

    hidden, _ = _capture_last_hidden(model, kwargs)
    pos = find_last_latent_state_index(enc_cur["input_ids"], token_id_map) - prev_len
    kv_latent = extract_latent_state(hidden[0], pos)
    print("step", step, "max_diff", float((legacy - kv_latent).abs().max()))
    return 0


def _move_batch(enc: dict, device: torch.device) -> dict:
    out = {}
    for k, v in enc.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
