"""Test eval loop - dataset creation and iteration."""
import time
from pathlib import Path
import torch
from hydra import initialize_config_dir, compose
from torch.utils.data import DataLoader

from src.data.semantic_dataset import SemanticAlignDataset
from src.train.train_semantic_align import _collate_semantic_batch
from src.train.manifest_resolver import resolve_manifest_for_split
from src.train.latent_cache import infer_latent_cache_path_from_manifest

print("=== Testing dataset + DataLoader ===")

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
    
    print("Creating dataset...")
    ds_start = time.time()
    dataset = SemanticAlignDataset(
        manifest_path=str(resolved_eval_manifest),
        latent_dim=int(wm_cfg.latent_dim),
        action_dim=int(dataset_cfg.action_dim),
        history_len=int(wm_cfg.history_len),
        image_encoder=None,
        latent_cache_path=str(resolved_cache) if resolved_cache else None,
    )
    print(f"Dataset created in {time.time() - ds_start:.1f}s")
    print(f"Dataset size: {len(dataset)}")
    
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=_collate_semantic_batch)
    print(f"DataLoader ready")
    
    # Test 5 iterations
    print("\nTesting 5 iterations...")
    iterator = iter(loader)
    for i in range(5):
        t_start = time.time()
        batch = next(iterator)
        elapsed = time.time() - t_start
        print(f"  Batch {i}: {elapsed:.3f}s, z_t={batch['z_t'].shape}")

print("\n=== Done ===")
