#!/usr/bin/env python3
"""KV chunk-invariance using HF prepare_inputs_for_generation where possible."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import argparse

from nimloth.latent import add_special_tokens, extract_latent_state, find_last_latent_state_index, special_token_ids
from nimloth.training.common.qwen_batch import encode_qwen_item
from nimloth.training.sft2.qwen_latent import reset_model_rope_state
from nimloth.wm.collate import prefix_messages_with_images
from nimloth.wm.dataset import NUM_NAVIGATION_ACTIONS, expand_record_transitions, load_jsonl_records


def _assistant_content(step: int) -> str:
    return f"<think>t{step}</think><|latent_state|><|action_start|><|action_({step % NUM_NAVIGATION_ACTIONS})|><|action_end|>"


def _make_record(tmpdir: Path):
    messages = [{"role": "system", "content": "sys"}]
    image_paths = []
    for step in range(2):
        p = tmpdir / f"img_{step}.png"
        Image.new("RGB", (224, 224), color=(step * 40, 80, 160)).save(p)
        image_paths.append(str(p))
        messages.append({"role": "user", "content": f"observe <image> step {step}"})
        messages.append({"role": "assistant", "content": _assistant_content(step)})
    p = tmpdir / "img_2.png"
    Image.new("RGB", (224, 224), color=(120, 120, 120)).save(p)
    image_paths.append(str(p))
    return {"id": "synthetic_2step", "messages": messages, "image_paths": image_paths, "action_indices": [0, 1]}


def _new_image_payload(prev_enc, cur_enc, device):
    if "pixel_values" not in cur_enc:
        return {}
    prev_images = int(prev_enc.get("image_grid_thw", torch.empty(0, 3)).shape[0])
    prev_patches = int(prev_enc.get("image_grid_thw", torch.empty(0, 3)).prod(dim=-1).sum().item()) if prev_images else 0
    return {
        "pixel_values": cur_enc["pixel_values"][prev_patches:].to(device),
        "image_grid_thw": cur_enc["image_grid_thw"][prev_images:].to(device),
    }


def _initial_payload(enc, device):
    return {k: enc[k].to(device) for k in ("pixel_values", "image_grid_thw") if k in enc}


def _call_prepared(model, enc, start, end, past, image_payload, device, *, force_first_iteration):
    input_ids = enc["input_ids"][:end].unsqueeze(0).to(device)
    attention_mask = enc["attention_mask"][:end].unsqueeze(0).to(device)
    cache_position = torch.arange(start, end, device=device, dtype=torch.long)
    prep = model.prepare_inputs_for_generation(
        input_ids=input_ids,
        past_key_values=past,
        attention_mask=attention_mask,
        use_cache=True,
        is_first_iteration=force_first_iteration,
        cache_position=cache_position,
        **image_payload,
    )
    meta = {
        "prepared_input_len": int(prep["input_ids"].shape[1]) if prep.get("input_ids") is not None else None,
        "prepared_attention_len": int(prep["attention_mask"].shape[1]) if prep.get("attention_mask") is not None else None,
        "has_pixel_values": prep.get("pixel_values") is not None,
        "position_ids_shape": list(prep["position_ids"].shape) if prep.get("position_ids") is not None else None,
    }
    prep.pop("is_first_iteration", None)
    out = model(**prep, output_hidden_states=True, return_dict=True)
    return out, meta


@torch.no_grad()
def run_case(model, processor, token_id_map, device, steps, max_length, case):
    enc0 = encode_qwen_item(prefix_messages_with_images(steps[0]), processor, max_length, include_labels=False)
    enc1 = encode_qwen_item(prefix_messages_with_images(steps[1]), processor, max_length, include_labels=False)
    len0, len1 = int(enc0["input_ids"].shape[0]), int(enc1["input_ids"].shape[0])
    latent0_pos = find_last_latent_state_index(enc0["input_ids"], token_id_map)
    latent1_pos = find_last_latent_state_index(enc1["input_ids"], token_id_map)
    split = latent1_pos
    img1 = _new_image_payload(enc0, enc1, device)

    reset_model_rope_state(model)
    out0, m0 = _call_prepared(model, enc0, 0, len0, None, _initial_payload(enc0, device), device, force_first_iteration=True)
    lat0 = extract_latent_state(out0.hidden_states[-1], latent0_pos).detach().cpu()
    out_one, m1 = _call_prepared(model, enc1, len0, len1, out0.past_key_values, img1, device, force_first_iteration=True)
    lat_one = extract_latent_state(out_one.hidden_states[-1], latent1_pos - len0).detach().cpu()

    reset_model_rope_state(model)
    out0b, m0b = _call_prepared(model, enc0, 0, len0, None, _initial_payload(enc0, device), device, force_first_iteration=True)
    lat0b = extract_latent_state(out0b.hidden_states[-1], latent0_pos).detach().cpu()
    out_mid, mmid = _call_prepared(model, enc1, len0, split, out0b.past_key_values, img1, device, force_first_iteration=True)
    out_two, m2 = _call_prepared(model, enc1, split, len1, out_mid.past_key_values, {}, device, force_first_iteration=False)
    lat_two = extract_latent_state(out_two.hidden_states[-1], 0).detach().cpu()

    return {
        "case": case,
        "len0": len0,
        "len1": len1,
        "split": split,
        "latent0_repeat_diff": float((lat0 - lat0b).abs().max().item()),
        "latent1_one_vs_two_diff": float((lat_one - lat_two).abs().max().item()),
        "metas": {"m0": m0, "m1": m1, "m0b": m0b, "mmid": mmid, "m2": m2},
        "past_lens": {
            "one": int(out_one.past_key_values.get_seq_length()),
            "mid": int(out_mid.past_key_values.get_seq_length()),
            "two": int(out_two.past_key_values.get_seq_length()),
        },
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--train-jsonl", type=Path)
    ap.add_argument("--record-index", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=12000)
    ap.add_argument("--max-pixels", type=int, default=602112)
    ap.add_argument("--attn-implementation", default="flash_attention_2")
    return ap.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
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
        reports.append(run_case(model, processor, token_id_map, device, expand_record_transitions(_make_record(Path(tmp)))[:2], args.max_length, "synthetic"))
    if args.train_jsonl:
        rec = load_jsonl_records(args.train_jsonl, max_records=args.record_index + 1)[args.record_index]
        reports.append(run_case(model, processor, token_id_map, device, expand_record_transitions(rec)[:2], args.max_length, f"real_{rec.get('id')}"))
    print(json.dumps({"reports": reports}, indent=2))


if __name__ == "__main__":
    main()
