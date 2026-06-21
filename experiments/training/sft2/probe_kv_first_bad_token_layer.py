#!/usr/bin/env python3
"""Find first pre-latent suffix token/layer that differs between one-chunk and two-chunk KV paths."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
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


def _prepare(model, enc, device):
    reset_model_rope_state(model)
    batch = {k: v.to(device) for k, v in _batch_enc(enc).items()}
    embeds = model.model.get_input_embeddings()(batch["input_ids"])
    if "pixel_values" in batch:
        image_embeds = _per_image_features(model, batch["pixel_values"], batch["image_grid_thw"]).to(embeds.device, embeds.dtype)
        image_token_id = getattr(model.config, "image_token_id", getattr(model.config, "image_token_index", None))
        mask = (batch["input_ids"] == image_token_id).unsqueeze(-1).expand_as(embeds).to(embeds.device)
        embeds = embeds.masked_scatter(mask, image_embeds)
    position_ids, _ = model.model.get_rope_index(
        input_ids=batch["input_ids"],
        image_grid_thw=batch.get("image_grid_thw"),
        video_grid_thw=batch.get("video_grid_thw"),
        attention_mask=batch.get("attention_mask"),
    )
    return {"inputs_embeds": embeds, "attention_mask": batch["attention_mask"], "position_ids": position_ids}


def _forward(model, prep, start, end, past):
    return model.model.language_model(
        input_ids=None,
        inputs_embeds=prep["inputs_embeds"][:, start:end],
        attention_mask=prep["attention_mask"][:, :end],
        position_ids=prep["position_ids"][..., start:end],
        past_key_values=past,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )


def _layer_cache(cache_obj, layer_idx: int):
    if hasattr(cache_obj, "layers"):
        layer = cache_obj.layers[layer_idx]
        return layer.keys.detach().cpu(), layer.values.detach().cpu()
    return cache_obj.key_cache[layer_idx].detach().cpu(), cache_obj.value_cache[layer_idx].detach().cpu()


def _maxdiff(a, b):
    return float((a - b).abs().max().item())


def _summarize_token_diffs(out_one, out_mid, probe_abs: int, probe_rel_one: int):
    num_layers = len(out_one.hidden_states) - 1
    hidden_diffs = []
    k_diffs = []
    v_diffs = []
    for layer_idx in range(num_layers):
        # hidden_states[0] is embedding input; hidden_states[layer_idx+1] is post-layer output
        h_one = out_one.hidden_states[layer_idx + 1][0, probe_rel_one].detach().cpu()
        h_mid = out_mid.hidden_states[layer_idx + 1][0, -1].detach().cpu()
        hidden_diffs.append(_maxdiff(h_one, h_mid))
        k_one, v_one = _layer_cache(out_one.past_key_values, layer_idx)
        k_mid, v_mid = _layer_cache(out_mid.past_key_values, layer_idx)
        k_diffs.append(_maxdiff(k_one[0, :, probe_abs, :], k_mid[0, :, probe_abs, :]))
        v_diffs.append(_maxdiff(v_one[0, :, probe_abs, :], v_mid[0, :, probe_abs, :]))
    return hidden_diffs, k_diffs, v_diffs


@torch.no_grad()
def run_case(model, processor, token_id_map, device, steps, max_length, case):
    enc0 = encode_qwen_item(prefix_messages_with_images(steps[0]), processor, max_length, include_labels=False)
    enc1 = encode_qwen_item(prefix_messages_with_images(steps[1]), processor, max_length, include_labels=False)
    len0 = int(enc0["input_ids"].shape[0])
    latent1 = find_last_latent_state_index(enc1["input_ids"], token_id_map)
    prep0 = _prepare(model, enc0, device)
    prep1 = _prepare(model, enc1, device)

    reset_model_rope_state(model)
    out0 = _forward(model, prep0, 0, len0, None)
    out_one = _forward(model, prep1, len0, latent1, out0.past_key_values)

    reset_model_rope_state(model)
    out0b = _forward(model, prep0, 0, len0, None)

    first_hidden = None
    first_k = None
    first_v = None
    token_reports = []
    for split in range(len0 + 1, latent1 + 1):
        out_mid = _forward(model, prep1, len0, split, out0b.past_key_values)
        probe_abs = split - 1
        probe_rel_one = probe_abs - len0
        hidden_diffs, k_diffs, v_diffs = _summarize_token_diffs(out_one, out_mid, probe_abs, probe_rel_one)
        if first_hidden is None:
            for layer_idx, d in enumerate(hidden_diffs):
                if d != 0.0:
                    first_hidden = {"probe_abs": probe_abs, "probe_rel_one": probe_rel_one, "layer": layer_idx, "diff": d}
                    break
        if first_k is None:
            for layer_idx, d in enumerate(k_diffs):
                if d != 0.0:
                    first_k = {"probe_abs": probe_abs, "probe_rel_one": probe_rel_one, "layer": layer_idx, "diff": d}
                    break
        if first_v is None:
            for layer_idx, d in enumerate(v_diffs):
                if d != 0.0:
                    first_v = {"probe_abs": probe_abs, "probe_rel_one": probe_rel_one, "layer": layer_idx, "diff": d}
                    break
        if split in {len0 + 1, latent1, max(len0 + 1, latent1 - 1)}:
            token_reports.append({
                "probe_abs": probe_abs,
                "probe_rel_one": probe_rel_one,
                "hidden_diffs": hidden_diffs,
                "k_diffs": k_diffs,
                "v_diffs": v_diffs,
            })
        if first_hidden is not None and first_k is not None and first_v is not None and len(token_reports) >= 3:
            pass

    return {
        "case": case,
        "len0": len0,
        "latent1": latent1,
        "num_prelatent_suffix_tokens": latent1 - len0,
        "first_hidden_diff": first_hidden,
        "first_k_diff": first_k,
        "first_v_diff": first_v,
        "sample_token_reports": token_reports,
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
