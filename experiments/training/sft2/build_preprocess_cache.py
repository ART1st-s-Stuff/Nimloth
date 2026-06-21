#!/usr/bin/env python
"""Prebuild SFT2 preprocess cache before launching DDP training."""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoProcessor

from nimloth.latent import add_special_tokens
from nimloth.training.sft2.cli import parse_sft2_args
from nimloth.training.sft2.preprocess_cache import (
    DEFAULT_MIN_PIXELS,
    build_trajectory_preprocess_cache,
    build_transition_preprocess_cache,
)


def main(argv: list[str] | None = None) -> int:
    args = parse_sft2_args(argv)
    if args.preprocess_cache_dir is None:
        print(json.dumps({"preprocess_cache": "disabled"}))
        return 0

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.image_processor.min_pixels = DEFAULT_MIN_PIXELS
    processor.image_processor.max_pixels = args.max_pixels
    add_special_tokens(processor.tokenizer)

    cache_root = Path(args.preprocess_cache_dir)
    build_kwargs = dict(
        model_path=Path(args.model),
        processor=processor,
        max_length=args.max_length,
        max_pixels=args.max_pixels,
        min_pixels=DEFAULT_MIN_PIXELS,
        preprocess_workers=args.preprocess_workers,
        force=args.force_rebuild_cache,
    )

    if args.packed_forward:
        build_trajectory_preprocess_cache(
            jsonl_path=Path(args.train_jsonl),
            cache_dir=cache_root / "train_trajectory",
            max_records=args.max_train_records,
            success_only=args.success_only,
            **build_kwargs,
        )
        build_trajectory_preprocess_cache(
            jsonl_path=Path(args.val_jsonl),
            cache_dir=cache_root / "val_trajectory",
            max_records=args.max_val_records,
            success_only=False,
            **build_kwargs,
        )
    else:
        build_transition_preprocess_cache(
            jsonl_path=Path(args.train_jsonl),
            cache_dir=cache_root / "train",
            max_records=args.max_train_records,
            success_only=args.success_only,
            value_gamma=args.value_gamma,
            **build_kwargs,
        )
        build_transition_preprocess_cache(
            jsonl_path=Path(args.val_jsonl),
            cache_dir=cache_root / "val",
            max_records=args.max_val_records,
            success_only=False,
            value_gamma=args.value_gamma,
            **build_kwargs,
        )
    print(json.dumps({"preprocess_cache": "ready", "dir": str(cache_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
