#!/usr/bin/env python3
"""Validate custom multi-turn multimodal KV semantics by calling Qwen text decoder directly."""

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
from nimloth.wm.dataset import NUM_NAVIGATION_ACTIONS, expand_record_transitions, load_jsonl_records


def _assistant_content(step: int) -> str:
    return f"<think>t{step}</think><|latent_state|><|action_start|><|action_({step % NUM_NAVIGATION_ACTIONS})|><|action_end|>"


def _make_record(tmpdir: Path) -> dict[str, Any]:
    messages = [{"role": "system", "content": "sys"}]
    image_paths: list[str] = []
    action_indices: list[int] = []
    for step in range(2):
        p = tmpdir / f"img_{step}.png"
        Image.new("RGB", (224, 224), color=(step * 40, 80, 160)).save(p)
        image_paths.append(str(p))
        messages.append({"role": "user", "content": f"observe <image> step {step}"})
        messages.append({"role": "assistant", "content": _assistant_content(step)})
        action_indices.append(step)
    p = tmpdir / "img_2.png"
    Image.new("RGB", (224, 224), color=(120, 120, 120)).save(p)
    image_paths.append(str(p))
    return {"id": "synthetic_2step", "messages": messages, "image_paths": image_paths, "action_indices": action_indices}


def _pooler_output(x):
    return x.pooler_output if hasattr(x, "pooler_output") else x


def _split_sizes(grid: torch.Tensor) -> list[int]:
    return [int(x) for x in grid.prod(dim=-1).tolist()]


def _per_image_features(model, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
    rows = []
    for chunk, grid in zip(torch.split(pixel_values, _split_sizes(image_grid_thw), dim=0), image_grid_thw, strict=True):
        out = _pooler_output(model.model.get_image_features(chunk, grid.unsqueeze(0)))
        if isinstance(out, (list, tuple)):
            if len(out) != 1:
                raise RuntimeError(f"single image expected one output, got {len(out)}")
            out = out[0]
        rows.append(out)
    return torch.cat(rows, dim=0)


def _compute_position_ids(model, batch: dict[str, torch.Tensor], inputs_embeds: torch.Tensor):
    inner = model.model
    if hasattr(inner, "compute_3d_position_ids"):
        return inner.compute_3d_position_ids(
            input_ids=batch["input_ids"],
            image_grid_thw=batch.get("image_grid_thw"),
            video_grid_thw=batch.get("video_grid_thw"),
            inputs_embeds=inputs_embeds,
            attention_mask=batch.get("attention_mask"),
            past_key_values=None,
            second_per_grid_ts=batch.get("second_per_grid_ts"),
            mm_token_type_ids=batch.get("mm_token_type_ids"),
        )
    pos, _ = inner.get_rope_index(
        input_ids=batch["input_ids"], image_grid_thw=batch.get("image_grid_thw"),
        video_grid_thw=batch.get("video_grid_thw"), attention_mask=batch.get("attention_mask")
    )
    return pos


def _prepare_full_decoder_inputs(model, enc: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    reset_model_rope_state(model)
    batch = {k: v.to(device) for k, v in _batch_enc(enc).items()}
    embeds = model.model.get_input_embeddings()(batch["input_ids"])
    if "pixel_values" in batch:
        image_embeds = _per_image_features(model, batch["pixel_values"], batch["image_grid_thw"]).to(embeds.device, embeds.dtype)
        image_token_id = getattr(model.config, "image_token_id", getattr(model.config, "image_token_index", None))
        if image_token_id is None:
            image_token_id = model.config.vision_config.image_token_id
        image_mask = (batch["input_ids"] == image_token_id).unsqueeze(-1).expand_as(embeds).to(embeds.device)
        if int(image_mask.sum().item()) != int(image_embeds.numel()):
            raise RuntimeError(f"image mask/features mismatch: mask={int(image_mask.sum().item())} features={image_embeds.numel()}")
        embeds = embeds.masked_scatter(image_mask, image_embeds)
    pos = _compute_position_ids(model, batch, embeds)
    return {"inputs_embeds": embeds, "attention_mask": batch["attention_mask"], "position_ids": pos}


def _forward_decoder_chunk(model, prepared: dict[str, torch.Tensor], start: int, end: int, past):
    return model.model.language_model(
        input_ids=None,
        inputs_embeds=prepared["inputs_embeds"][:, start:end],
        attention_mask=prepared["attention_mask"][:, :end],
        position_ids=prepared["position_ids"][..., start:end],
        past_key_values=past,
        use_cache=True,
    )


@torch.no_grad()
def run_case(model, processor, token_id_map, device, steps, max_length: int, case: str) -> dict[str, Any]:
    enc0 = encode_qwen_item(prefix_messages_with_images(steps[0]), processor, max_length, include_labels=False)
    enc1 = encode_qwen_item(prefix_messages_with_images(steps[1]), processor, max_length, include_labels=False)
    len0, len1 = int(enc0["input_ids"].shape[0]), int(enc1["input_ids"].shape[0])
    lat0_pos = find_last_latent_state_index(enc0["input_ids"], token_id_map)
    lat1_pos = find_last_latent_state_index(enc1["input_ids"], token_id_map)
    split = lat1_pos
    prep0 = _prepare_full_decoder_inputs(model, enc0, device)
    prep1 = _prepare_full_decoder_inputs(model, enc1, device)

    reset_model_rope_state(model)
    out0 = _forward_decoder_chunk(model, prep0, 0, len0, None)
    lat0 = extract_latent_state(out0.last_hidden_state, lat0_pos).detach().cpu()
    out_one = _forward_decoder_chunk(model, prep1, len0, len1, out0.past_key_values)
    lat_one = extract_latent_state(out_one.last_hidden_state, lat1_pos - len0).detach().cpu()

    reset_model_rope_state(model)
    out0b = _forward_decoder_chunk(model, prep0, 0, len0, None)
    lat0b = extract_latent_state(out0b.last_hidden_state, lat0_pos).detach().cpu()
    out_mid = _forward_decoder_chunk(model, prep1, len0, split, out0b.past_key_values)
    out_two = _forward_decoder_chunk(model, prep1, split, len1, out_mid.past_key_values)
    lat_two = extract_latent_state(out_two.last_hidden_state, 0).detach().cpu()
    return {
        "case": case,
        "len0": len0,
        "len1": len1,
        "split": split,
        "past_lens": {"one": int(out_one.past_key_values.get_seq_length()), "mid": int(out_mid.past_key_values.get_seq_length()), "two": int(out_two.past_key_values.get_seq_length())},
        "inputs_embeds_prefix_diff": float((prep0["inputs_embeds"] - prep1["inputs_embeds"][:, :len0]).abs().max().item()),
        "position_ids_prefix_diff": float((prep0["position_ids"] - prep1["position_ids"][..., :len0]).abs().max().item()),
        "latent0_repeat_diff": float((lat0 - lat0b).abs().max().item()),
        "latent1_one_vs_two_diff": float((lat_one - lat_two).abs().max().item()),
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
