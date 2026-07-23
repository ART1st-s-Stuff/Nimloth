#!/usr/bin/env python3
"""Validate read-only reuse of the historical k=16 Qwen/DINO cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor

from nimloth.backbone import CachedDINOGridTargets, DINOV2_LARGE_IDENTITY
from nimloth.latent import add_special_tokens
from nimloth.rollout.transitions import (
    TransitionContextIndex,
    TransitionJsonlDataset,
    transition_training_item,
)
from nimloth.training.sft2.data.samplers import OnlineHistoryBatchSampler
from nimloth.util.cache import (
    COMPACT_CACHE_FORMAT_V1,
    DEFAULT_MIN_PIXELS,
    LEGACY_TRANSITION_EXPANSION_VERSION,
    CachedTransitionDataset,
    CompactCachedTransitionCollator,
    cache_fingerprint,
    encode_transition_item,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft1-checkpoint", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=12000)
    parser.add_argument("--max-pixels", type=int, default=100352)
    return parser.parse_args()


def _assert_encoding_equal(
    cached: dict[str, torch.Tensor],
    fresh: dict[str, torch.Tensor],
    *,
    source: str,
) -> None:
    if set(cached) != set(fresh):
        raise ValueError(
            f"{source} tensor keys differ: {sorted(cached)} != {sorted(fresh)}"
        )
    for key in cached:
        expected = fresh[key]
        if key == "pixel_values":
            expected = expected.to(dtype=cached[key].dtype)
        if not torch.equal(cached[key].cpu(), expected.cpu()):
            raise ValueError(f"{source} tensor differs: {key}")


def _selected_indices(samples) -> list[int]:
    terminal = [
        index
        for index, sample in enumerate(samples)
        if index + 1 == len(samples)
        or samples[index + 1].record_id != sample.record_id
    ]
    candidates = [0, len(samples) // 2, terminal[0], terminal[-1]]
    return sorted(set(candidates))


def _validate_split(
    *,
    split: str,
    jsonl_path: Path,
    cache_dir: Path,
    processor: Any,
    model_path: Path,
    max_length: int,
    max_pixels: int,
) -> dict[str, Any]:
    samples = TransitionJsonlDataset(jsonl_path).samples
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    expected_base_fingerprint = cache_fingerprint(
        jsonl_path,
        max_length=max_length,
        max_pixels=max_pixels,
        min_pixels=DEFAULT_MIN_PIXELS,
        vocab_size=len(processor.tokenizer),
        value_gamma=1.0,
        latent_token_count=16,
        mask_latent_query_labels=True,
        cache_format=COMPACT_CACHE_FORMAT_V1,
        image_dtype="bfloat16",
        processor_source=str(model_path.resolve()),
        transition_expansion_version=LEGACY_TRANSITION_EXPANSION_VERSION,
    )
    if manifest.get("base_fingerprint") != expected_base_fingerprint:
        raise ValueError(f"{split} v1 cache fingerprint does not match inputs")
    if int(manifest.get("count", -1)) != len(samples):
        raise ValueError(f"{split} cache transition count mismatch")
    if not all(
        sample.next_prefix_messages is not None
        and sample.next_prefix_image_paths is not None
        for sample in samples
    ):
        raise ValueError(f"{split} contains a transition without a next prompt")

    image_index = json.loads(
        (cache_dir / "image_index.json").read_text(encoding="utf-8")
    )
    cached_paths = {
        str(Path(location["path"]).resolve())
        for location in image_index["images"]
    }
    required_paths = {
        str(Path(path).resolve())
        for sample in samples
        for path in (*sample.prefix_image_paths, sample.next_image_path)
    }
    if required_paths - cached_paths:
        raise ValueError(
            f"{split} cache misses {len(required_paths - cached_paths)} images"
        )

    dataset = CachedTransitionDataset(
        cache_dir,
        samples,
        processor=processor,
        max_length=max_length,
        latent_token_count=16,
        mask_latent_query_labels=True,
    )
    collator = CompactCachedTransitionCollator(cache_dir)
    for index in _selected_indices(samples):
        sample = samples[index]
        cached_entry = dataset[TransitionContextIndex(index, 1, True)]
        materialized = collator([cached_entry])
        fresh = encode_transition_item(
            transition_training_item(sample),
            processor,
            max_length,
            latent_token_count=16,
            mask_latent_query_labels=True,
        )
        _assert_encoding_equal(
            materialized["current_enc_rows"][0],
            fresh["current_enc"],
            source=f"{split}:{index}:current",
        )
        cached_next = materialized["next_enc_rows"][0]
        if cached_next is None or fresh["next_enc"] is None:
            raise ValueError(f"{split}:{index} next encoding is missing")
        _assert_encoding_equal(
            cached_next,
            fresh["next_enc"],
            source=f"{split}:{index}:next",
        )

    sampler = OnlineHistoryBatchSampler(
        samples,
        history_size=4,
        batch_size=1,
        shuffle=False,
        pad_to_equal_batches=False,
    )
    if sampler.window_count != len(samples):
        raise ValueError(
            f"{split} sampler dropped transitions: "
            f"{sampler.window_count} != {len(samples)}"
        )
    return {
        "records": len({sample.record_id for sample in samples}),
        "transitions": len(samples),
        "cache_images": len(cached_paths),
        "sampler_current_steps": sampler.window_count,
        "tensor_equivalence_indices": _selected_indices(samples),
        "base_fingerprint": expected_base_fingerprint,
    }


def main() -> None:
    args = _parse_args()
    processor = AutoProcessor.from_pretrained(
        args.sft1_checkpoint,
        trust_remote_code=True,
    )
    if add_special_tokens(processor.tokenizer, latent_token_count=16) != 0:
        raise ValueError("SFT1 tokenizer does not already contain all 16 query tokens")
    processor.image_processor.min_pixels = DEFAULT_MIN_PIXELS
    processor.image_processor.max_pixels = args.max_pixels

    result = {
        split: _validate_split(
            split=split,
            jsonl_path=args.records_root / f"{split}_all.jsonl",
            cache_dir=args.cache_root / split,
            processor=processor,
            model_path=args.sft1_checkpoint,
            max_length=args.max_length,
            max_pixels=args.max_pixels,
        )
        for split in ("train", "val")
    }
    dino_targets = CachedDINOGridTargets.from_cache_root(
        args.cache_root,
        identity=DINOV2_LARGE_IDENTITY,
        grid_size=4,
    )
    result["dino_cache_fingerprint"] = dino_targets.cache_fingerprint
    result["status"] = "compatible_read_only_v1"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
