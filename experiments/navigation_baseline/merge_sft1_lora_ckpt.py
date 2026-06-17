#!/usr/bin/env python3
"""Merge a LoRA adapter checkpoint into hf_merged for VAGEN eval."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", type=Path, required=True)
    ap.add_argument("--adapter-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.adapter_dir, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    base.resize_token_embeddings(len(processor.tokenizer))
    merged = PeftModel.from_pretrained(base, args.adapter_dir).merge_and_unload()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.out_dir, safe_serialization=True)
    processor.save_pretrained(args.out_dir)
    print(f"merged -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
