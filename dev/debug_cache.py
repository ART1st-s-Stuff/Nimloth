#!/usr/bin/env python3
"""Debug cache loading."""
import sys
sys.path.insert(0, '.')
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from omegaconf import OmegaConf
from pathlib import Path
import torch

wm_cfg = OmegaConf.load('configs/wm/lewm_qwen25vl_8b.yaml')
run_dir = Path('datasets/ai2thor/test/2026-04-24_14-47-16')
cache_path = run_dir / f"{run_dir.name}.latents.{wm_cfg.name}.pt"

print(f"Cache path: {cache_path}")
print(f"Cache exists: {cache_path.exists()}")

if cache_path.exists():
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    latents = payload.get("latents", {})
    print(f"Cache loaded: {len(latents)} entries")

    # Test image path from manifest
    test_path = "datasets/ai2thor/train/2026-04-24_14-47-16/images/floorplan1_ep0000_step0000.png"
    print(f"Test path: {test_path}")
    print(f"Test path in cache: {test_path in latents}")

    # Check a few keys
    keys = list(latents.keys())[:5]
    print(f"Sample keys: {keys}")

    # Check what format the keys are in
    import json
    from src.data.dataset import read_worker_manifests
    samples = read_worker_manifests(run_dir)
    image_paths = sorted(set(str(s["image_path"]) for s in samples if "image_path" in s))
    print(f"\nManifest has {len(image_paths)} images")
    print(f"First manifest path: {image_paths[0]}")
    print(f"First manifest path in cache: {image_paths[0] in latents}")

    # Check overlap
    cache_keys = set(latents.keys())
    manifest_paths = set(image_paths)
    overlap = cache_keys & manifest_paths
    print(f"\nOverlap: {len(overlap)} images")
else:
    print("Cache does not exist!")