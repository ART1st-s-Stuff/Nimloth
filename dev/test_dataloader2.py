"""Test DataLoader iteration speed - simpler version."""
import sys
import time
import torch
from hydra import initialize_config_dir, compose
from torch.utils.data import DataLoader
from pathlib import Path

from src.data.semantic_dataset import SemanticAlignDataset
from src.train.train_semantic_align import _collate_semantic_batch

print("Starting simple DataLoader test...")

config_dir = Path("/home/jincai_guo/atst/flower/configs").resolve()
with initialize_config_dir(config_dir=str(config_dir), version_base=None):
    cfg = compose(config_name="config", overrides=["wm=cfm_dinov2m"])
    wm_cfg = cfg.wm
    dataset_cfg = cfg.dataset
    train_cfg = cfg.pipeline.train.semantic_align
    
    print(f"wm.latent_dim: {wm_cfg.latent_dim}, wm.hidden_dim: {wm_cfg.hidden_dim}")
    
    # Get manifest path directly
    from src.train.manifest_resolver import resolve_manifest_for_split
    from src.train.latent_cache import infer_latent_cache_path_from_manifest
    
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
    print(f"Loading dataset...")
    
    dataset = SemanticAlignDataset(
        manifest_path=str(resolved_eval_manifest),
        latent_dim=int(wm_cfg.latent_dim),
        action_dim=int(dataset_cfg.action_dim),
        history_len=int(wm_cfg.history_len),
        image_encoder=None,  # No encoder for val split
        positive_k=int(train_cfg.positive_k),
        negative_gap=int(train_cfg.negative_gap),
        latent_cache_path=str(resolved_cache) if resolved_cache else None,
    )
    print(f"Dataset size: {len(dataset)}")
    
    # Use small batch size and workers=0 for test
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_semantic_batch,
    )
    
    print("Starting iteration...")
    start = time.time()
    for i, batch in enumerate(loader):
        if i >= 2:
            break
        elapsed = time.time() - start
        print(f"Batch {i}: z_t shape={batch['z_t'].shape}, took {elapsed:.2f}s")
        start = time.time()

print("Done!")
