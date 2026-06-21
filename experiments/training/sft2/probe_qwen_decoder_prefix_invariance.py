#!/usr/bin/env python3
"""Probe where Qwen2.5-VL prefix/full mismatch first appears."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.latent import add_special_tokens, special_token_ids
from nimloth.training.common.qwen_batch import encode_qwen_item
from nimloth.training.sft2.qwen_latent import reset_model_rope_state
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


def _capture_layer_outputs(module, call):
    captured: list[torch.Tensor] = []
    handles = []

    def hook(_mod, _inp, out):
        captured.append((out[0] if isinstance(out, tuple) else out).detach().cpu())

    for layer in module.layers:
        handles.append(layer.register_forward_hook(hook))
    try:
        result = call()
    finally:
        for handle in handles:
            handle.remove()
    return captured, result


def _pooler_output(x):
    if hasattr(x, "pooler_output"):
        return x.pooler_output
    return x


def _forward_wrapper_layers(model, enc: dict[str, torch.Tensor], device: torch.device) -> list[torch.Tensor]:
    reset_model_rope_state(model)
    batch = _batch_enc(enc)
    model_inputs = {k: v.to(device) for k, v in batch.items()}
    layers, _ = _capture_layer_outputs(
        model.model.language_model,
        lambda: model(**model_inputs, output_hidden_states=False, return_dict=True),
    )
    return layers


def _prepare_decoder_inputs(model, enc: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    reset_model_rope_state(model)
    batch = {k: v.to(device) for k, v in _batch_enc(enc).items()}
    inner = model.model
    inputs_embeds = inner.get_input_embeddings()(batch["input_ids"])

    if batch.get("pixel_values") is not None:
        image_features = _pooler_output(inner.get_image_features(batch["pixel_values"], batch.get("image_grid_thw")))
        image_embeds = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = inner.get_placeholder_mask(
            batch["input_ids"],
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if batch.get("pixel_values_videos") is not None:
        video_features = _pooler_output(
            inner.get_video_features(
                batch["pixel_values_videos"],
                batch.get("video_grid_thw"),
            )
        )
        video_embeds = torch.cat(video_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = inner.get_placeholder_mask(
            batch["input_ids"],
            inputs_embeds=inputs_embeds,
            video_features=video_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if hasattr(inner, "compute_3d_position_ids"):
        position_ids = inner.compute_3d_position_ids(
            input_ids=batch.get("input_ids"),
            image_grid_thw=batch.get("image_grid_thw"),
            video_grid_thw=batch.get("video_grid_thw"),
            inputs_embeds=inputs_embeds,
            attention_mask=batch.get("attention_mask"),
            past_key_values=None,
            second_per_grid_ts=batch.get("second_per_grid_ts"),
            mm_token_type_ids=batch.get("mm_token_type_ids"),
        )
    elif hasattr(inner, "get_rope_index"):
        position_ids, _ = inner.get_rope_index(
            input_ids=batch["input_ids"],
            image_grid_thw=batch.get("image_grid_thw"),
            video_grid_thw=batch.get("video_grid_thw"),
            attention_mask=batch.get("attention_mask"),
        )
    else:
        raise RuntimeError("Qwen2_5_VLModel has neither compute_3d_position_ids nor get_rope_index")
    if position_ids is None:
        raise RuntimeError("expected multimodal position_ids, got None")

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": batch.get("attention_mask"),
        "position_ids": position_ids,
    }


def _forward_decoder_layers(model, enc: dict[str, torch.Tensor], device: torch.device) -> list[torch.Tensor]:
    decoder_inputs = _prepare_decoder_inputs(model, enc, device)
    text_model = model.model.language_model
    layers, _ = _capture_layer_outputs(
        text_model,
        lambda: text_model(**decoder_inputs),
    )
    return layers


def _layer_diffs(prefix_layers: list[torch.Tensor], full_layers: list[torch.Tensor], prefix_len: int) -> tuple[list[float], int | None]:
    diffs: list[float] = []
    first_nonzero: int | None = None
    for idx, (prefix, full) in enumerate(zip(prefix_layers, full_layers, strict=True)):
        diff = float((prefix[0, :prefix_len] - full[0, :prefix_len]).abs().max().item())
        diffs.append(diff)
        if first_nonzero is None and diff > 1e-6:
            first_nonzero = idx
    return diffs, first_nonzero


def analyze_case(name: str, steps: list[TransitionSample], model, processor, device: torch.device, max_length: int) -> dict[str, Any]:
    prefix_enc = encode_qwen_item(prefix_messages_with_images(steps[0]), processor, max_length, include_labels=False)
    full_enc = encode_qwen_item(prefix_messages_with_images(steps[-1]), processor, max_length, include_labels=False)
    prefix_len = int(prefix_enc["input_ids"].shape[0])

    wrapper_prefix = _forward_wrapper_layers(model, prefix_enc, device)
    wrapper_full = _forward_wrapper_layers(model, full_enc, device)
    decoder_prefix = _forward_decoder_layers(model, prefix_enc, device)
    decoder_full = _forward_decoder_layers(model, full_enc, device)

    wrapper_diffs, wrapper_first = _layer_diffs(wrapper_prefix, wrapper_full, prefix_len)
    decoder_diffs, decoder_first = _layer_diffs(decoder_prefix, decoder_full, prefix_len)

    return {
        "case": name,
        "prefix_len": prefix_len,
        "full_len": int(full_enc["input_ids"].shape[0]),
        "input_ids_eq_full_prefix": bool(torch.equal(prefix_enc["input_ids"], full_enc["input_ids"][:prefix_len])),
        "position_ids_eq_full_prefix": None,
        "wrapper": {
            "num_layers": len(wrapper_diffs),
            "first_nonzero_layer": wrapper_first,
            "layer_max_abs_diffs": wrapper_diffs,
        },
        "decoder_only": {
            "num_layers": len(decoder_diffs),
            "first_nonzero_layer": decoder_first,
            "layer_max_abs_diffs": decoder_diffs,
        },
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
    _ = special_token_ids(processor.tokenizer)

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
