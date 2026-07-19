#!/usr/bin/env python3
"""Benchmark online frozen-DINO CLS extraction against a validated sidecar."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from nimloth.backbone.dino import CachedDINOEncoder, FrozenDINOEncoder


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dino-cache-dir", type=Path, required=True)
    ap.add_argument("--dino-model", default="facebook/dinov2-large")
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--batch-sizes", default="1,2,4,8,16")
    ap.add_argument("--repeats", type=int, default=3)
    return ap.parse_args()


def _elapsed(encoder, paths: list[str], batch_size: int, repeats: int, device: torch.device) -> float:
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(repeats):
        for offset in range(0, len(paths), batch_size):
            encoder.encode_image_paths(paths[offset : offset + batch_size], device=device)
    torch.cuda.synchronize(device)
    return time.perf_counter() - started


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("DINO cache benchmark requires CUDA")
    device = torch.device("cuda:0")
    online = FrozenDINOEncoder.from_pretrained(
        args.dino_model,
        device=device,
        dtype=torch.bfloat16,
    )
    cached = CachedDINOEncoder.from_cache_root(args.dino_cache_dir, identity=online.identity)
    image_index = json.loads(
        (args.dino_cache_dir / "train" / "image_index.json").read_text(encoding="utf-8")
    )
    paths = [entry["path"] for entry in image_index["images"][: args.samples]]
    if not paths:
        raise ValueError("benchmark requires at least one cached image")

    # Warm up kernels and filesystem pages, then prove target equivalence before timing.
    online_features = online.encode_image_paths(paths[:2], device=device)
    cached_features = cached.encode_image_paths(paths[:2], device=device)
    if not torch.equal(online_features, cached_features):
        raise ValueError(
            "online/cache benchmark target mismatch: "
            f"max_abs={float((online_features - cached_features).abs().max().item())}"
        )

    results = []
    for batch_size in (int(value) for value in args.batch_sizes.split(",")):
        if batch_size < 1:
            raise ValueError("batch sizes must be positive")
        online_seconds = _elapsed(online, paths, batch_size, args.repeats, device)
        cached_seconds = _elapsed(cached, paths, batch_size, args.repeats, device)
        image_count = len(paths) * args.repeats
        results.append(
            {
                "batch_size": batch_size,
                "images": image_count,
                "online_seconds": online_seconds,
                "cached_seconds": cached_seconds,
                "online_images_per_second": image_count / online_seconds,
                "cached_images_per_second": image_count / cached_seconds,
                "teacher_only_speedup": online_seconds / cached_seconds,
            }
        )
    print(json.dumps({"dino_cache_benchmark": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
