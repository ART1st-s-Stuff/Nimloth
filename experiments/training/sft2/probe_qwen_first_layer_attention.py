#!/usr/bin/env python3
"""Probe Qwen2.5-VL first decoder layer attention inputs for prefix/full mismatch."""

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

from nimloth.latent import add_special_tokens
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


def _pooler_output(x):
    if hasattr(x, "pooler_output"):
        return x.pooler_output
    return x


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
    else:
        position_ids, _ = inner.get_rope_index(
            input_ids=batch["input_ids"],
            image_grid_thw=batch.get("image_grid_thw"),
            video_grid_thw=batch.get("video_grid_thw"),
            attention_mask=batch.get("attention_mask"),
        )
    if position_ids is None:
        raise RuntimeError("expected multimodal position_ids, got None")

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": batch.get("attention_mask"),
        "position_ids": position_ids,
    }


def _capture_first_layer(model, decoder_inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    text_model = model.model.language_model
    layer = text_model.layers[0]
    attn = layer.self_attn
    captured: dict[str, torch.Tensor] = {}
    orig_forward = attn.forward

    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()
        captured["attn_input"] = hidden_states.detach().cpu()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        captured["q_proj_out"] = query_states.detach().cpu()
        captured["k_proj_out"] = key_states.detach().cpu()
        captured["v_proj_out"] = value_states.detach().cpu()

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        captured["cos"] = cos.detach().cpu()
        captured["sin"] = sin.detach().cpu()
        captured["position_ids"] = position_ids.detach().cpu() if position_ids is not None else None
        captured["attention_mask"] = attention_mask.detach().cpu() if attention_mask is not None else None
        captured["q_reshaped_pre_rope"] = query_states.detach().cpu()
        captured["k_reshaped_pre_rope"] = key_states.detach().cpu()
        captured["v_reshaped"] = value_states.detach().cpu()
        return orig_forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    attn.forward = MethodType(patched_forward, attn)
    try:
        text_model(**decoder_inputs)
    finally:
        attn.forward = orig_forward
    return captured


def _diff(a: torch.Tensor | None, b: torch.Tensor | None, prefix_len: int, *, head_layout: bool = False) -> float | None:
    if a is None or b is None:
        return None
    if head_layout:
        return float((a[..., :prefix_len, :] - b[..., :prefix_len, :]).abs().max().item())
    return float((a[:, :prefix_len, ...] - b[:, :prefix_len, ...]).abs().max().item())


def _seq_last_diff(a: torch.Tensor | None, b: torch.Tensor | None, prefix_len: int) -> float | None:
    if a is None or b is None:
        return None
    return float((a[..., :prefix_len] - b[..., :prefix_len]).abs().max().item())


def _seq_penultimate_diff(a: torch.Tensor | None, b: torch.Tensor | None, prefix_len: int) -> float | None:
    if a is None or b is None:
        return None
    return float((a[..., :prefix_len, :] - b[..., :prefix_len, :]).abs().max().item())


def _mask_diff(a: torch.Tensor | None, b: torch.Tensor | None, prefix_len: int) -> float | None:
    if a is None or b is None:
        return None
    if a.ndim == 4:
        return float((a[..., :prefix_len, :prefix_len] != b[..., :prefix_len, :prefix_len]).float().max().item())
    if a.ndim == 2:
        return float((a[:, :prefix_len] != b[:, :prefix_len]).float().max().item())
    return float((a - b).abs().max().item())


def analyze_case(name: str, steps: list[TransitionSample], model, processor, device: torch.device, max_length: int) -> dict[str, Any]:
    prefix_enc = encode_qwen_item(prefix_messages_with_images(steps[0]), processor, max_length, include_labels=False)
    full_enc = encode_qwen_item(prefix_messages_with_images(steps[-1]), processor, max_length, include_labels=False)
    prefix_len = int(prefix_enc["input_ids"].shape[0])

    prefix_dec = _prepare_decoder_inputs(model, prefix_enc, device)
    full_dec = _prepare_decoder_inputs(model, full_enc, device)
    prefix_cap = _capture_first_layer(model, prefix_dec)
    full_cap = _capture_first_layer(model, full_dec)

    return {
        "case": name,
        "prefix_len": prefix_len,
        "full_len": int(full_enc["input_ids"].shape[0]),
        "inputs_embeds_prefix_max_diff": _diff(prefix_dec["inputs_embeds"].cpu(), full_dec["inputs_embeds"].cpu(), prefix_len),
        "position_ids_prefix_max_diff": _seq_last_diff(prefix_dec["position_ids"].cpu(), full_dec["position_ids"].cpu(), prefix_len),
        "first_layer": {
            "attn_input_max_diff": _diff(prefix_cap.get("attn_input"), full_cap.get("attn_input"), prefix_len),
            "q_proj_max_diff": _diff(prefix_cap.get("q_proj_out"), full_cap.get("q_proj_out"), prefix_len),
            "k_proj_max_diff": _diff(prefix_cap.get("k_proj_out"), full_cap.get("k_proj_out"), prefix_len),
            "v_proj_max_diff": _diff(prefix_cap.get("v_proj_out"), full_cap.get("v_proj_out"), prefix_len),
            "attention_mask_prefix_diff": _mask_diff(prefix_cap.get("attention_mask"), full_cap.get("attention_mask"), prefix_len),
            "position_ids_prefix_diff": _seq_last_diff(
                prefix_cap.get("position_ids"),
                full_cap.get("position_ids"),
                prefix_len,
            ),
            "cos_prefix_max_diff": _seq_penultimate_diff(prefix_cap.get("cos"), full_cap.get("cos"), prefix_len),
            "sin_prefix_max_diff": _seq_penultimate_diff(prefix_cap.get("sin"), full_cap.get("sin"), prefix_len),
            "q_reshaped_pre_rope_max_diff": _diff(prefix_cap.get("q_reshaped_pre_rope"), full_cap.get("q_reshaped_pre_rope"), prefix_len, head_layout=True),
            "k_reshaped_pre_rope_max_diff": _diff(prefix_cap.get("k_reshaped_pre_rope"), full_cap.get("k_reshaped_pre_rope"), prefix_len, head_layout=True),
            "v_reshaped_max_diff": _diff(prefix_cap.get("v_reshaped"), full_cap.get("v_reshaped"), prefix_len, head_layout=True),
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
