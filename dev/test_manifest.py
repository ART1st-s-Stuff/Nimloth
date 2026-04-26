"""Test manifest resolution and data loading."""
from pathlib import Path
from hydra import initialize_config_dir, compose

config_dir = Path("/home/jincai_guo/atst/flower/configs").resolve()
with initialize_config_dir(config_dir=str(config_dir), version_base=None):
    cfg = compose(config_name="config", overrides=["wm=cfm_dinov2m"])
    dataset_cfg = cfg.dataset
    train_cfg = cfg.pipeline.train.semantic_align
    
    manifests_cfg = dict(dataset_cfg.get("manifests", {}))
    print("Manifests config:")
    for k, v in manifests_cfg.items():
        print(f"  {k}: {v}")
    
    from src.train.manifest_resolver import resolve_manifest_for_split
    from src.train.latent_cache import infer_latent_cache_path_from_manifest
    
    eval_split = str(train_cfg.get("eval_split", "val"))
    resolved_eval_manifest = resolve_manifest_for_split(
        manifests_cfg=manifests_cfg,
        split=eval_split,
        outputs_root=str(train_cfg.operation.outputs_root),
        dataset_name=str(dataset_cfg.name),
    )
    print(f"\nResolved manifest for '{eval_split}': {resolved_eval_manifest}")
    print(f"Exists: {resolved_eval_manifest.exists()}")
    
    resolved_cache = infer_latent_cache_path_from_manifest(
        str(resolved_eval_manifest), str(cfg.wm.name)
    )
    print(f"\nLatent cache: {resolved_cache}")
    if resolved_cache:
        print(f"Cache exists: {Path(resolved_cache).exists()}")

print("\nDone!")
