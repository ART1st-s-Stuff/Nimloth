"""Test eval loop with timing."""
import time
from pathlib import Path
import torch
from hydra import initialize_config_dir, compose
from torch.utils.data import DataLoader

from src.data.semantic_dataset import SemanticAlignDataset
from src.train.train_semantic_align import DeltaProjector, _collate_semantic_batch
from src.train.manifest_resolver import resolve_manifest_for_split
from src.train.latent_cache import infer_latent_cache_path_from_manifest
from src.vlm.qwen_adapter import QwenVLMAdapter
from src.vlm.semantic_state import SemanticStateGenerator

print("=== Testing eval loop with model + VLM ===")

config_dir = Path("/home/jincai_guo/atst/flower/configs").resolve()
with initialize_config_dir(config_dir=str(config_dir), version_base=None):
    cfg = compose(config_name="config", overrides=["wm=cfm_dinov2m"])
    wm_cfg = cfg.wm
    dataset_cfg = cfg.dataset
    train_cfg = cfg.pipeline.train.semantic_align
    
    device = torch.device(str(train_cfg.device))
    
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
    
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=_collate_semantic_batch)
    print(f"DataLoader ready, {len(loader)} batches")
    
    # Load model
    print("\nLoading model...")
    model = DeltaProjector(latent_dim=int(wm_cfg.latent_dim), hidden_dim=int(wm_cfg.hidden_dim)).to(device)
    ckpt_path = Path("/home/jincai_guo/atst/flower/models/semantic_align/cfm_dinov2m/2026-04-26_16-05-48/checkpoint_batch14000.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["projector"])
    model.eval()
    print("Model loaded.")
    
    # VLM adapter
    adapter = QwenVLMAdapter(
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        latent_dim=int(wm_cfg.latent_dim),
        enabled=False,
        fallback_enabled=True,
        max_new_tokens=128,
    )
    semantic_generator = SemanticStateGenerator(vlm_adapter=adapter)
    print(f"VLM init_error: {adapter.init_error}")
    
    # Run 10 batches with timing
    print("\n=== Running 10 batches ===")
    iterator = iter(loader)
    total_start = time.time()
    
    for batch_idx in range(10):
        batch_start = time.time()
        batch = next(iterator)
        
        z_t = batch["z_t"].to(device)
        z_t_pos = batch["z_t_pos"].to(device)
        z_t_neg = batch["z_t_neg"].to(device)
        
        model_start = time.time()
        pred_pos = model(z_t=z_t, z_tp=z_t_pos)
        pred_neg = model(z_t=z_t, z_tp=z_t_neg)
        model_time = time.time() - model_start
        
        vlm_start = time.time()
        out_t = semantic_generator.infer(
            image_path=batch["image_path"][0],
            history_image_paths=[batch["image_path"][0]],
            task_text=batch["task_text"][0],
            env_context=batch["env_context"][0],
        )
        vlm_time = time.time() - vlm_start
        
        total_time = time.time() - batch_start
        print(f"Batch {batch_idx}: total={total_time:.3f}s, model={model_time:.3f}s, vlm={vlm_time:.3f}s")
    
    print(f"\nTotal time: {time.time() - total_start:.1f}s")
    print("Done!")

print("\n=== All done ===")
