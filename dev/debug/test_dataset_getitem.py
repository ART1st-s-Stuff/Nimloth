"""Test dataset __getitem__ directly."""
from pathlib import Path
import time
import torch
from hydra import initialize_config_dir, compose

config_dir = Path("/home/jincai_guo/atst/flower/configs").resolve()
with initialize_config_dir(config_dir=str(config_dir), version_base=None):
    cfg = compose(config_name="config", overrides=["wm=cfm_dinov2m"])
    wm_cfg = cfg.wm
    dataset_cfg = cfg.dataset
    train_cfg = cfg.pipeline.train.semantic_align
    
    from src.train.manifest_resolver import resolve_manifest_for_split
    from src.train.latent_cache import infer_latent_cache_path_from_manifest
    from src.data.semantic_dataset import SemanticAlignDataset
    
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
    
    print(f"Creating dataset...")
    dataset = SemanticAlignDataset(
        manifest_path=str(resolved_eval_manifest),
        latent_dim=int(wm_cfg.latent_dim),
        action_dim=int(dataset_cfg.action_dim),
        history_len=int(wm_cfg.history_len),
        image_encoder=None,
        positive_k=int(train_cfg.positive_k),
        negative_gap=int(train_cfg.negative_gap),
        latent_cache_path=str(resolved_cache) if resolved_cache else None,
    )
    print(f"Dataset size: {len(dataset)}")
    
    # Test individual items
    print("\nTesting individual item access:")
    for i in [0, 100, 1000, 10000]:
        start = time.time()
        item = dataset[i]
        elapsed = time.time() - start
        print(f"  Item {i}: {elapsed:.2f}s, keys={list(item.keys())}")
        if 'z_t' in item:
            print(f"    z_t shape: {item['z_t'].shape}")

print("\nDone!")
