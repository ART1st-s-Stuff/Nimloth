#!/usr/bin/env python3
"""Build validated frozen-DINO CLS sidecars for an existing compact SFT2 cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nimloth.backbone.dino import (
    CachedDINOEncoder,
    FrozenDINOEncoder,
    build_dino_feature_cache,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preprocess-cache-dir", type=Path, required=True)
    ap.add_argument("--dino-model", default="facebook/dinov2-large")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--shard-size", type=int, default=1024)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify-samples", type=int, default=8)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("production DINO target cache must be built with the CUDA BF16 training runtime")
    device = torch.device("cuda:0")
    encoder = FrozenDINOEncoder.from_pretrained(
        args.dino_model,
        device=device,
        dtype=torch.bfloat16,
    )
    manifests = build_dino_feature_cache(
        cache_root=args.preprocess_cache_dir,
        encoder=encoder,
        device=device,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        force=args.force,
    )
    cached = CachedDINOEncoder.from_cache_root(
        args.preprocess_cache_dir,
        identity=encoder.identity,
    )

    verify_paths = []
    for split in ("train", "val"):
        index = json.loads(
            (args.preprocess_cache_dir / split / "image_index.json").read_text(encoding="utf-8")
        )
        verify_paths.extend(entry["path"] for entry in index["images"][: args.verify_samples])
    if verify_paths:
        online = encoder.encode_image_paths(verify_paths, device=device)
        offline = cached.encode_image_paths(verify_paths, device=device)
        if not torch.equal(online, offline):
            max_abs = float((online - offline).abs().max().item())
            raise ValueError(f"cached DINO targets are not bitwise equal to online BF16 targets: max_abs={max_abs}")
    print(
        json.dumps(
            {
                "dino_cache": "ready",
                "root": str(args.preprocess_cache_dir),
                "identity": encoder.identity.__dict__,
                "fingerprint": cached.cache_fingerprint,
                "splits": {key: value["count"] for key, value in manifests.items()},
                "bitwise_verify_samples": len(verify_paths),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
