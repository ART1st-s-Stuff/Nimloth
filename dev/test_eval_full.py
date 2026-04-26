"""直接测试 val split dataset iteration without warmup."""
import sys
import time
from pathlib import Path
import torch
from hydra import initialize_config_dir, compose
from torch.utils.data import DataLoader

from src.data.semantic_dataset import SemanticAlignDataset
from src.train.train_semantic_align import _collate_semantic_batch

print("=== Testing val dataset without warmup ===")

config_dir = Path("/home/jincai_guo/atst/flower/configs").resolve()
with initialize_config_dir(config_dir=str(config_dir), version_base=None):
    cfg = compose(config_name="config", overrides=["wm=cfm_dinov2m"])
    wm_cfg = cfg.wm
    dataset_cfg = cfg.dataset
    train_cfg = cfg.pipeline.train.semantic_align
    
    manifests_cfg = dict(dataset_cfg.get("manifests", {}))
    eval_split = str(train_cfg.get("eval_split", "val"))
    
    from src.train.manifest_resolver import resolve_manifest_for_split
    from src.train.latent_cache import infer_latent_cache_path_from_manifest
    
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
    print(f"Cache exists: {Path(resolved_cache).exists() if resolved_cache else 'None'}")
    
    # Create dataset WITHOUT encoder (will use lazy loading from cache)
    print("\nCreating dataset (no encoder)...")
    start = time.time()
    dataset = SemanticAlignDataset(
        manifest_path=str(resolved_eval_manifest),
        latent_dim=int(wm_cfg.latent_dim),
        action_dim=int(dataset_cfg.action_dim),
        history_len=int(wm_cfg.history_len),
        image_encoder=None,  # No encoder - lazy loading
        latent_cache_path=str(resolved_cache) if resolved_cache else None,
    )
    print(f"Dataset created in {time.time() - start:.1f}s")
    print(f"Dataset size: {len(dataset)}")
    
    # Check if warmup happened
    wm = dataset._wm_dataset
    print(f"Latent cache size: {len(wm._latent_cache)}")
    
    # Test one item
    print("\nTesting __getitem__...")
    start = time.time()
    try:
        item = dataset[0]
        print(f"Item 0 loaded in {time.time() - start:.1f}s")
        print(f"  keys: {list(item.keys())}")
        print(f"  z_t shape: {item['z_t'].shape}")
    except Exception as e:
        print(f"Error loading item 0: {e}")

print("\nDone!")
