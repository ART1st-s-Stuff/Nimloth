#!/usr/bin/env python3
"""Compare legacy per-prefix forward vs packed KV trajectory on one record."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.latent import add_special_tokens, special_token_ids
from nimloth.training.sft2.packed_trajectory import (
    TrajectoryStepResult,
    _reset_model_rope_state,
    assert_trajectory_latents_equivalent,
    kv_forward_trajectory,
    legacy_forward_trajectory,
)
from nimloth.wm.dataset import expand_record_transitions, load_jsonl_records


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--train-jsonl", type=Path, required=True)
    ap.add_argument("--record-index", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=12000)
    ap.add_argument("--atol", type=float, default=1e-2)
    return ap.parse_args()


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

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    model.to(device).eval()

    legacy = legacy_forward_trajectory(
        model, transitions, processor, token_id_map, device, max_length=args.max_length
    )
    legacy_cpu = [
        TrajectoryStepResult(
            step_index=step.step_index,
            current_latent=step.current_latent.detach().cpu(),
            enc=step.enc,
            item=step.item,
        )
        for step in legacy
    ]
    del legacy
    torch.cuda.empty_cache()
    _reset_model_rope_state(model)

    packed = kv_forward_trajectory(
        model, transitions, processor, token_id_map, device, max_length=args.max_length
    )

    per_step: list[dict[str, float | int]] = []
    max_diff = 0.0
    for leg, pack in zip(legacy_cpu, packed, strict=True):
        diff = float((leg.current_latent - pack.current_latent.detach().cpu()).abs().max().item())
        max_diff = max(max_diff, diff)
        per_step.append({"step": leg.step_index, "max_diff": diff})

    print("num_steps", len(transitions))
    print("per_step", per_step)
    try:
        assert_trajectory_latents_equivalent(legacy_cpu, packed, atol=args.atol)
        print("passed", True, "max_diff", max_diff)
        return 0
    except AssertionError as exc:
        print("passed", False, "max_diff", max_diff, "error", str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
