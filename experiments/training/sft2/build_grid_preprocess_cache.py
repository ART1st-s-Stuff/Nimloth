#!/usr/bin/env python3
"""Build k16 compact Qwen current/next-prefix cache for grid SFT2."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoProcessor

from nimloth.latent import add_special_tokens
from nimloth.training.sft2.preprocess_cache import (
    DEFAULT_MIN_PIXELS,
    build_compact_transition_preprocess_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft1-checkpoint", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=12000)
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(str(args.sft1_checkpoint), trust_remote_code=True)
    if add_special_tokens(processor.tokenizer, latent_token_count=16) != 0:
        raise ValueError("completed SFT1 tokenizer must already contain all 16 query tokens")
    processor.image_processor.min_pixels = DEFAULT_MIN_PIXELS
    processor.image_processor.max_pixels = args.max_pixels
    common = dict(
        model_path=args.sft1_checkpoint,
        processor=processor,
        max_length=args.max_length,
        max_pixels=args.max_pixels,
        min_pixels=DEFAULT_MIN_PIXELS,
        preprocess_workers=args.workers,
        force=args.force,
        latent_token_count=16,
        mask_latent_query_labels=True,
        image_dtype="bfloat16",
        image_shard_size=128,
        transition_shard_size=256,
    )
    build_compact_transition_preprocess_cache(
        jsonl_path=args.train_jsonl,
        cache_dir=args.cache_root / "train",
        success_only=False,
        **common,
    )
    build_compact_transition_preprocess_cache(
        jsonl_path=args.val_jsonl,
        cache_dir=args.cache_root / "val",
        success_only=False,
        **common,
    )


if __name__ == "__main__":
    main()
