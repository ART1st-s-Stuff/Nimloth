"""Test dataset warmup."""
import time
from pathlib import Path
import torch
from hydra import initialize_config_dir, compose

from src.data.semantic_dataset import SemanticAlignDataset
from src.train.manifest_resolver import resolve_manifest_for_split
from src.train.latent_cache import infer_latent_cache_path_from_manifest

print("=== Testing warmup ===")

config_dir = Path("/home/jincai_guo/atst/flower/configs").resolve()
with initialize_config_dir(config_dir=str(config_dir), version_base=None):
    cfg = compose(config_name="config", overrides=["wm=cfm_dinov2m"])
    wm_cfg = cfg.wm
    dataset_cfg = cfg.dataset
    train_cfg = cfg.pipeline.train.semantic_align
    
    manifests_cfg = dict(dataset_cfg.get("manifests", {}))
    eval_split = str(train_cfg.get("eval_split", "val"))
    
    resolved_eval_manifest = resolve_manifest_for_split(
        manifests_cfg=manifests_cfg,
        split=eval_split,
        outputs_root=str(train_cfg.operation.outputs_root),
        dataset_name=str(dataset_cfg.name),
    )
    resolved_cache = infer_latent_cache_path_from_manifest(
        str(resolved_eval_manifest), str(wm_cfg.name)
    )
    
    print(f"Manifest: {resolved_eval_manifest}")
    print(f"Cache: {resolved_cache}")
    
    # Check cache is single file (not dir)
    cache_path = Path(resolved_cache) if resolved_cache else None
    print(f"Cache is_file: {cache_path.is_file() if cache_path else 'None'}")
    print(f"Cache is_dir: {cache_path.is_dir() if cache_path else 'None'}")
    
    print("\nCreating dataset...")
    start = time.time()
    dataset = SemanticAlignDataset(
        manifest_path=str(resolved_eval_manifest),
        latent_dim=int(wm_cfg.latent_dim),
        action_dim=int(dataset_cfg.action_dim),
        history_len=int(wm_cfg.history_len),
        image_encoder=None,
        latent_cache_path=str(resolved_cache) if resolved_cache else None,
    )
    print(f"Dataset init took {time.time() - start:.1f}s")
    print(f"Dataset size: {len(dataset)}")
    print(f"Latent cache size: {len(dataset._wm_dataset._latent_cache)}")

print("\nDone!")
