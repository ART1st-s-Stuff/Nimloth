#!/usr/bin/env python3
"""Compare legacy per-prefix training tensors vs trajectory-once packed forward."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.latent import add_special_tokens, special_token_ids
from nimloth.training.sft2.trajectory_equiv import legacy_record_losses, packed_record_losses
from nimloth.wm.dataset import load_jsonl_records


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--train-jsonl", type=Path, required=True)
    ap.add_argument("--max-records", type=int, default=3)
    ap.add_argument("--max-length", type=int, default=12000)
    ap.add_argument("--atol-latent", type=float, default=1e-2)
    ap.add_argument("--atol-loss", type=float, default=1e-3)
    return ap.parse_args()


@torch.no_grad()
def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = 602112
    add_special_tokens(processor.tokenizer)
    token_id_map = special_token_ids(processor.tokenizer)

    from nimloth.wm import LatentWMPredictor, LeWMConfig, StateProjector, ValueHead

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    model.to(device).eval()
    dtype = next(model.parameters()).dtype
    state_proj = StateProjector(model.config.hidden_size, 128).to(device=device, dtype=dtype).eval()
    wm_predictor = LatentWMPredictor.create(LeWMConfig(emb_dim=128)).to(device).eval()
    value_head = ValueHead(128).to(device=device, dtype=dtype).eval()

    passed = True
    for record in load_jsonl_records(args.train_jsonl, max_records=args.max_records):
        legacy = legacy_record_losses(
            model, processor, token_id_map, device, record, args.max_length, state_proj, wm_predictor, value_head
        )
        packed = packed_record_losses(
            model, processor, token_id_map, device, record, args.max_length, state_proj, wm_predictor, value_head
        )
        latent_diff = float((legacy["current"] - packed["current"]).abs().max().item())
        lm_diff = float(abs(float(legacy["lm_loss"]) - float(packed["lm_loss"])))
        wm_diff = float(abs(float(legacy["wm_loss"]) - float(packed["wm_loss"])))
        value_diff = float(abs(float(legacy["value_loss"]) - float(packed["value_loss"])))
        total_diff = float(abs(float(legacy["total_loss"]) - float(packed["total_loss"])))
        ok = latent_diff <= args.atol_latent and max(lm_diff, wm_diff, value_diff, total_diff) <= args.atol_loss
        passed = passed and ok
        print(
            record.get("id"),
            {
                "latent_max_diff": latent_diff,
                "lm_diff": lm_diff,
                "wm_diff": wm_diff,
                "value_diff": value_diff,
                "total_diff": total_diff,
                "passed": ok,
            },
        )
    print("passed", passed)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
