"""Test WMDataset init and item access."""
from pathlib import Path
import time
import torch

from hydra import initialize_config_dir, compose

config_dir = Path("/home/jincai_guo/atst/flower/configs").resolve()
with initialize_config_dir(config_dir=str(config_dir), version_base=None):
    cfg = compose(config_name="config", overrides=["wm=cfm_dinov2m"])
    wm_cfg = cfg.wm
    dataset_cfg = cfg.dataset
    
    cache_path = Path("/home/jincai_guo/atst/flower/datasets/ai2thor/val/2026-04-24_14-47-16/2026-04-24_14-47-16.latents.cfm_dinov2m")
    print(f"Cache exists: {cache_path.exists()}, size: {cache_path.stat().st_size / 1e9:.1f} GB")
    
    print("Loading cache directly (not via WMDataset warmup)...")
    start = time.time()
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    elapsed = time.time() - start
    print(f"Cache load time: {elapsed:.1f}s")
    
    latents = payload.get("latents", {}) if isinstance(payload, dict) else {}
    latent_dim = payload.get("latent_dim", "unknown") if isinstance(payload, dict) else "unknown"
    print(f"Latent dim from cache: {latent_dim}")
    print(f"Number of cached latents: {len(latents)}")
    
    # Check a sample latent
    if latents:
        sample_key = list(latents.keys())[0]
        sample_val = latents[sample_key]
        print(f"Sample latent key: {sample_key}, shape: {sample_val.shape}")

print("\nDone!")
