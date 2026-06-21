#!/usr/bin/env python3
"""Compare first suffix token as single-token cached extend vs first token of a long cached chunk."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.latent import add_special_tokens, find_last_latent_state_index, special_token_ids
from nimloth.training.common.qwen_batch import encode_qwen_item
from nimloth.training.sft2.qwen_latent import reset_model_rope_state
from nimloth.training.sft2.trajectory_forward import _batch_enc
from nimloth.wm.collate import prefix_messages_with_images
from nimloth.wm.dataset import NUM_NAVIGATION_ACTIONS, expand_record_transitions, load_jsonl_records


def _assistant_content(step: int) -> str:
    return f"<think>t{step}</think><|latent_state|><|action_start|><|action_({step % NUM_NAVIGATION_ACTIONS})|><|action_end|>"


def _make_record(tmpdir: Path) -> dict[str, Any]:
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


def _pooler_output(x):
    return x.pooler_output if hasattr(x, "pooler_output") else x


def _split_sizes(grid: torch.Tensor) -> list[int]:
    return [int(x) for x in grid.prod(dim=-1).tolist()]


def _per_image_features(model, pixel_values, image_grid_thw):
    rows = []
    for chunk, grid in zip(torch.split(pixel_values, _split_sizes(image_grid_thw), dim=0), image_grid_thw, strict=True):
        out = _pooler_output(model.model.get_image_features(chunk, grid.unsqueeze(0)))
        if isinstance(out, (list, tuple)):
            out = out[0]
        rows.append(out)
    return torch.cat(rows, dim=0)


def _prepare(model, enc, device, position_id_mode: str):
    reset_model_rope_state(model)
    batch = {k: v.to(device) for k, v in _batch_enc(enc).items()}
    embeds = model.model.get_input_embeddings()(batch["input_ids"])
    if "pixel_values" in batch:
        image_embeds = _per_image_features(model, batch["pixel_values"], batch["image_grid_thw"]).to(embeds.device, embeds.dtype)
        image_token_id = getattr(model.config, "image_token_id", getattr(model.config, "image_token_index", None))
        mask = (batch["input_ids"] == image_token_id).unsqueeze(-1).expand_as(embeds).to(embeds.device)
        embeds = embeds.masked_scatter(mask, image_embeds)
    position_ids_3d, _ = model.model.get_rope_index(
        input_ids=batch["input_ids"],
        image_grid_thw=batch.get("image_grid_thw"),
        video_grid_thw=batch.get("video_grid_thw"),
        attention_mask=batch.get("attention_mask"),
    )
    if position_id_mode == "3d":
        position_ids = position_ids_3d
    elif position_id_mode == "4d_text_prefix":
        batch_size, seq_len = batch["input_ids"].shape
        text_position_ids = torch.arange(seq_len, device=device, dtype=position_ids_3d.dtype).view(1, 1, seq_len).expand(1, batch_size, seq_len)
        position_ids = torch.cat([text_position_ids, position_ids_3d], dim=0)
    else:
        raise ValueError(f"Unknown position_id_mode: {position_id_mode}")
    return {"inputs_embeds": embeds, "attention_mask": batch["attention_mask"], "position_ids": position_ids}


def _patch_attn(model, layer_idx: int, target_idx: int):
    attn = model.model.language_model.layers[layer_idx].self_attn
    original = attn.forward
    capture: dict[str, Any] = {}

    def patched(self, hidden_states, attention_mask=None, position_ids=None, output_attentions=False, use_cache=False, position_embeddings=None, **kwargs):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        pkv = kwargs.get("past_key_value", None)
        cache_position = kwargs.get("cache_position", None)
        capture["hidden_target"] = hidden_states[0, target_idx].detach().cpu()
        capture["q_proj_target"] = q[0, target_idx].detach().cpu()
        capture["k_proj_target"] = k[0, target_idx].detach().cpu()
        capture["v_proj_target"] = v[0, target_idx].detach().cpu()
        capture["q_len"] = int(hidden_states.shape[1])
        capture["attention_mask_shape"] = list(attention_mask.shape) if hasattr(attention_mask, "shape") else None
        capture["position_ids_shape"] = list(position_ids.shape) if hasattr(position_ids, "shape") else None
        if position_ids is None:
            capture["position_ids_target"] = None
        elif position_ids.ndim == 2:
            capture["position_ids_target"] = [int(position_ids[0, target_idx].detach().cpu().item())]
        else:
            capture["position_ids_target"] = position_ids[:, 0, target_idx].detach().cpu().tolist()
        capture["cache_position"] = cache_position.detach().cpu().tolist() if cache_position is not None else None
        capture["past_len_before"] = int(pkv.get_seq_length()) if pkv is not None and hasattr(pkv, "get_seq_length") else 0
        out = original(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        tensor = out[0] if isinstance(out, tuple) else out
        capture["attn_output_target"] = tensor[0, target_idx].detach().cpu()
        capture["past_len_after"] = int(pkv.get_seq_length()) if pkv is not None and hasattr(pkv, "get_seq_length") else None
        return out

    attn.forward = MethodType(patched, attn)
    return attn, original, capture


def _forward_capture(model, prep, start, end, past, layer_idx: int, target_idx: int):
    attn, original, capture = _patch_attn(model, layer_idx, target_idx)
    try:
        out = model.model.language_model(
            input_ids=None,
            inputs_embeds=prep["inputs_embeds"][:, start:end],
            attention_mask=prep["attention_mask"][:, :end],
            position_ids=prep["position_ids"][..., start:end],
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
    finally:
        attn.forward = original
    return out, capture


def _prefix_forward(model, prep0, len0):
    return model.model.language_model(
        input_ids=None,
        inputs_embeds=prep0["inputs_embeds"][:, :len0],
        attention_mask=prep0["attention_mask"][:, :len0],
        position_ids=prep0["position_ids"][..., :len0],
        use_cache=True,
        return_dict=True,
    )


def _maxdiff(a, b):
    return float((a - b).abs().max().item())


def _cmp(a, b):
    return {
        "hidden_target_diff": _maxdiff(a["hidden_target"], b["hidden_target"]),
        "q_proj_target_diff": _maxdiff(a["q_proj_target"], b["q_proj_target"]),
        "k_proj_target_diff": _maxdiff(a["k_proj_target"], b["k_proj_target"]),
        "v_proj_target_diff": _maxdiff(a["v_proj_target"], b["v_proj_target"]),
        "attn_output_target_diff": _maxdiff(a["attn_output_target"], b["attn_output_target"]),
        "q_len_single": a["q_len"],
        "q_len_long": b["q_len"],
        "past_len_before_single": a["past_len_before"],
        "past_len_before_long": b["past_len_before"],
        "past_len_after_single": a["past_len_after"],
        "past_len_after_long": b["past_len_after"],
        "attention_mask_shape_single": a["attention_mask_shape"],
        "attention_mask_shape_long": b["attention_mask_shape"],
        "position_ids_target_single": a["position_ids_target"],
        "position_ids_target_long": b["position_ids_target"],
        "cache_position_single": a["cache_position"],
        "cache_position_long": b["cache_position"],
    }


@torch.no_grad()
def run_case(model, processor, token_id_map, device, steps, max_length, case, position_id_mode):
    enc0 = encode_qwen_item(prefix_messages_with_images(steps[0]), processor, max_length, include_labels=False)
    enc1 = encode_qwen_item(prefix_messages_with_images(steps[1]), processor, max_length, include_labels=False)
    len0 = int(enc0["input_ids"].shape[0])
    latent1 = find_last_latent_state_index(enc1["input_ids"], token_id_map)
    prep0 = _prepare(model, enc0, device, position_id_mode)
    prep1 = _prepare(model, enc1, device, position_id_mode)

    results: dict[str, Any] = {"case": case, "position_id_mode": position_id_mode, "len0": len0, "latent1": latent1, "long_q_len": latent1 - len0}
    for layer_idx in (0, 1, 2):
        reset_model_rope_state(model)
        out0_single = _prefix_forward(model, prep0, len0)
        out_single, cap_single = _forward_capture(model, prep1, len0, len0 + 1, out0_single.past_key_values, layer_idx, 0)
        reset_model_rope_state(model)
        out0_long = _prefix_forward(model, prep0, len0)
        out_long, cap_long = _forward_capture(model, prep1, len0, latent1, out0_long.past_key_values, layer_idx, 0)
        d = _cmp(cap_single, cap_long)
        d["final_hidden_target_diff"] = _maxdiff(out_single.last_hidden_state[0, 0], out_long.last_hidden_state[0, 0])
        results[f"layer{layer_idx}"] = d
    return results


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--train-jsonl", type=Path)
    ap.add_argument("--record-index", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=12000)
    ap.add_argument("--max-pixels", type=int, default=602112)
    ap.add_argument("--attn-implementation", default="flash_attention_2")
    ap.add_argument("--position-id-mode", choices=["3d", "4d_text_prefix"], default="3d")
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
        reports.append(run_case(model, processor, token_id_map, device, expand_record_transitions(_make_record(Path(tmp)))[:2], args.max_length, "synthetic", args.position_id_mode))
    if args.train_jsonl:
        rec = load_jsonl_records(args.train_jsonl, max_records=args.record_index + 1)[args.record_index]
        reports.append(run_case(model, processor, token_id_map, device, expand_record_transitions(rec)[:2], args.max_length, f"real_{rec.get('id')}", args.position_id_mode))
    print(json.dumps({"reports": reports}, indent=2))


if __name__ == "__main__":
    main()
