"""Minimal test of eval loop."""
from pathlib import Path
import time
import torch
from hydra import initialize_config_dir, compose
from torch.utils.data import DataLoader

from src.data.semantic_dataset import SemanticAlignDataset
from src.train.train_semantic_align import DeltaProjector, _collate_semantic_batch
from src.train.manifest_resolver import resolve_manifest_for_split
from src.train.latent_cache import infer_latent_cache_path_from_manifest
from src.vlm.qwen_adapter import QwenVLMAdapter
from src.vlm.semantic_state import SemanticStateGenerator

print("Starting minimal eval test...")

config_dir = Path("/home/jincai_guo/atst/flower/configs").resolve()
with initialize_config_dir(config_dir=str(config_dir), version_base=None):
    cfg = compose(config_name="config", overrides=["wm=cfm_dinov2m"])
    wm_cfg = cfg.wm
    dataset_cfg = cfg.dataset
    train_cfg = cfg.pipeline.train.semantic_align
    vlm_cfg = cfg.vlm
    
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
    
    print(f"Creating dataset...")
    dataset = SemanticAlignDataset(
        manifest_path=str(resolved_eval_manifest),
        latent_dim=int(wm_cfg.latent_dim),
        action_dim=int(dataset_cfg.action_dim),
        history_len=int(wm_cfg.history_len),
        image_encoder=None,  # No encoder - rely on cache
        positive_k=int(train_cfg.positive_k),
        negative_gap=int(train_cfg.negative_gap),
        latent_cache_path=str(resolved_cache) if resolved_cache else None,
    )
    print(f"Dataset size: {len(dataset)}")
    
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_semantic_batch,
    )
    
    # Load model and checkpoint
    print("Loading model...")
    model = DeltaProjector(latent_dim=int(wm_cfg.latent_dim), hidden_dim=int(wm_cfg.hidden_dim)).to(device)
    ckpt_path = Path("/home/jincai_guo/atst/flower/models/semantic_align/cfm_dinov2m/2026-04-26_16-05-48/checkpoint_batch14000.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["projector"])
    model.eval()
    print("Model loaded.")
    
    # Create VLM adapter with fallback mode
    print("Creating VLM adapter...")
    adapter = QwenVLMAdapter(
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        latent_dim=int(wm_cfg.latent_dim),
        enabled=False,  # Disabled - will use fallback
        fallback_enabled=True,
        max_new_tokens=128,
    )
    print(f"Adapter init_error: {adapter.init_error}")
    
    semantic_generator = SemanticStateGenerator(vlm_adapter=adapter)
    
    # Run just 3 batches
    print("\nStarting eval loop (3 batches)...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= 3:
                break
            start = time.time()
            z_t = batch["z_t"].to(device)
            z_t_pos = batch["z_t_pos"].to(device)
            z_t_neg = batch["z_t_neg"].to(device)
            
            pred_pos = model(z_t=z_t, z_tp=z_t_pos)
            pred_neg = model(z_t=z_t, z_tp=z_t_neg)
            
            # Test VLM fallback (just 1 sample per batch)
            out_t = semantic_generator.infer(
                image_path=batch["image_path"][0],
                history_image_paths=[batch["image_path"][0]],
                task_text=batch["task_text"][0],
                env_context=batch["env_context"][0],
            )
            
            elapsed = time.time() - start
            print(f"Batch {batch_idx}: z_t={z_t.shape}, pred_pos={pred_pos.shape}, took {elapsed:.2f}s")
            print(f"  s_t shape: {out_t.s_t.shape}, cot: {out_t.cot_text[:50]}...")

print("\nDone!")
