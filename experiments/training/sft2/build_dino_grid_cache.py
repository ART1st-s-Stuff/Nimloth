#!/usr/bin/env python3
"""Build exact float32 DINOv2 pooled 4x4 targets beside compact Qwen caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nimloth.backbone.dino import (
    DEFAULT_DINO_MODEL,
    CachedDINOGridEncoder,
    FrozenDINOEncoder,
    build_dino_grid_feature_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--dino-model", default=DEFAULT_DINO_MODEL)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.grid_size != 4:
        parser.error("the approved SFT2 representation requires grid-size=4")
    if not torch.cuda.is_available():
        raise RuntimeError("DINO grid cache build requires CUDA")
    device = torch.device("cuda")
    encoder = FrozenDINOEncoder.from_pretrained(
        args.dino_model,
        device=device,
        dtype=torch.bfloat16,
    )
    manifests = build_dino_grid_feature_cache(
        cache_root=args.cache_root,
        encoder=encoder,
        device=device,
        grid_size=args.grid_size,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        force=args.force,
    )
    cached = CachedDINOGridEncoder.from_cache_root(
        args.cache_root,
        identity=encoder.identity,
        grid_size=args.grid_size,
    )
    checked = 0
    for split in ("train", "val"):
        index = json.loads((args.cache_root / split / "image_index.json").read_text(encoding="utf-8"))
        paths = [entry["path"] for entry in index["images"][:2]]
        if not paths:
            continue
        online = encoder.encode_image_paths_grid(paths, device=device, grid_size=args.grid_size)
        restored = cached.encode_image_paths_grid(paths, device=device, grid_size=args.grid_size)
        if not torch.equal(online.float(), restored.float()):
            delta = float((online.float() - restored.float()).abs().max().item())
            raise ValueError(f"DINO grid cache online/restored mismatch for {split}: max_delta={delta}")
        checked += len(paths)
    print(
        json.dumps(
            {
                "manifests": {split: manifest["fingerprint"] for split, manifest in manifests.items()},
                "cache_fingerprint": cached.cache_fingerprint,
                "bitwise_checked": checked,
            }
        )
    )


if __name__ == "__main__":
    main()
