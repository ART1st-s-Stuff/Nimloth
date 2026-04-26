"""使用小数据集快速验证 semantic align 评估流程。"""
import sys
import time
import json
from pathlib import Path
import torch
from hydra import initialize_config_dir, compose

from src.data.semantic_dataset import SemanticAlignDataset
from src.train.train_semantic_align import DeltaProjector, _collate_semantic_batch
from src.vlm.qwen_adapter import QwenVLMAdapter
from src.vlm.semantic_state import SemanticStateGenerator
from src.utils.io import ensure_dir, write_json

print("=== Small dataset semantic align eval ===")

config_dir = Path("/home/jincai_guo/atst/flower/configs").resolve()
with initialize_config_dir(config_dir=str(config_dir), version_base=None):
    cfg = compose(config_name="config", overrides=["wm=cfm_dinov2m"])
    wm_cfg = cfg.wm
    dataset_cfg = cfg.dataset
    train_cfg = cfg.pipeline.train.semantic_align
    
    device = torch.device(str(train_cfg.device))
    
    # 使用小数据集：只取前 100 个样本的 manifest
    train_manifest = Path("/home/jincai_guo/atst/flower/datasets/ai2thor/train/2026-04-24_14-47-16")
    small_manifest_dir = Path("/tmp/semantic_align_test_manifest")
    small_manifest_dir.mkdir(exist_ok=True)
    
    # 创建一个小 manifest（只取前 100 条）
    all_rows = []
    for wf in sorted(train_manifest.glob("manifest_worker_*.jsonl"))[:2]:  # 只取前 2 个 worker 文件
        lines = wf.read_text().strip().split("\n")
        for line in lines[:50]:  # 每个文件只取前 50 条
            all_rows.append(json.loads(line))
    
    small_manifest_path = small_manifest_dir / "test_manifest.jsonl"
    small_manifest_path.write_text("\n".join(json.dumps(r) for r in all_rows))
    print(f"Created small manifest with {len(all_rows)} samples at {small_manifest_path}")
    
    # 创建小 latent cache
    small_cache_dir = small_manifest_dir / "latents"
    small_cache_dir.mkdir(exist_ok=True)
    cache_path = Path("/home/jincai_guo/atst/flower/datasets/ai2thor/train/2026-04-24_14-47-16/2026-04-24_14-47-16.latents.cfm_dinov2m")
    
    print(f"Loading original cache to extract subset...")
    full_cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    latents = full_cache.get("latents", {})
    
    small_latents = {}
    for row in all_rows[:20]:  # 只缓存前 20 个 latents
        img_path = row.get("image_path", "")
        if img_path in latents:
            small_latents[img_path] = latents[img_path]
    
    small_cache_path = small_cache_dir / "test_latents.pt"
    torch.save({"latents": small_latents, "latent_dim": 6144}, small_cache_path)
    print(f"Created small cache with {len(small_latents)} latents at {small_cache_path}")
    
    # 创建 dataset
    print("\nCreating dataset...")
    dataset = SemanticAlignDataset(
        manifest_path=str(small_manifest_path),
        latent_dim=int(wm_cfg.latent_dim),
        action_dim=int(dataset_cfg.action_dim),
        history_len=int(wm_cfg.history_len),
        image_encoder=None,
        latent_cache_path=str(small_cache_dir),  # 使用小缓存目录
    )
    print(f"Dataset size: {len(dataset)}")
    
    # Load model
    print("\nLoading model...")
    model = DeltaProjector(latent_dim=int(wm_cfg.latent_dim), hidden_dim=int(wm_cfg.hidden_dim)).to(device)
    ckpt_path = Path("/home/jincai_guo/atst/flower/models/semantic_align/cfm_dinov2m/2026-04-26_16-05-48/checkpoint_batch14000.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["projector"])
    model.eval()
    print("Model loaded.")
    
    # VLM adapter with fallback
    adapter = QwenVLMAdapter(
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        latent_dim=int(wm_cfg.latent_dim),
        enabled=False,  # Use fallback
        fallback_enabled=True,
        max_new_tokens=128,
    )
    print(f"VLM adapter init_error: {adapter.init_error}")
    
    semantic_generator = SemanticStateGenerator(vlm_adapter=adapter)
    
    # Run eval on small dataset
    print("\n=== Running eval ===")
    same_intent_sims = []
    diff_intent_sims = []
    temporal_smooths = []
    
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=_collate_semantic_batch)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            print(f"\nBatch {batch_idx}:")
            z_t = batch["z_t"].to(device)
            z_t_pos = batch["z_t_pos"].to(device)
            z_t_neg = batch["z_t_neg"].to(device)
            
            print(f"  z_t shape: {z_t.shape}")
            
            pred_pos = model(z_t=z_t, z_tp=z_t_pos)
            pred_neg = model(z_t=z_t, z_tp=z_t_neg)
            
            print(f"  pred_pos shape: {pred_pos.shape}")
            
            for i in range(z_t.size(0)):
                start = time.time()
                out_t = semantic_generator.infer(
                    image_path=batch["image_path"][i],
                    history_image_paths=[batch["image_path"][i]],
                    task_text=batch["task_text"][i],
                    env_context=batch["env_context"][i],
                )
                elapsed = time.time() - start
                
                s_t = torch.nn.functional.normalize(out_t.s_t, dim=0)
                sim_same = torch.sum(s_t * torch.nn.functional.normalize(pred_pos[i].cpu(), dim=0)).item()
                sim_diff = torch.sum(s_t * torch.nn.functional.normalize(pred_neg[i].cpu(), dim=0)).item()
                
                same_intent_sims.append(float(sim_same))
                diff_intent_sims.append(float(sim_diff))
                temporal_smooths.append(float(torch.mean((out_t.s_t - out_t.s_t) ** 2).item()))
                
                print(f"  Sample {i}: sim_same={sim_same:.4f}, sim_diff={sim_diff:.4f}, vlm_time={elapsed:.2f}s")
            
            if batch_idx >= 2:
                break
    
    # Summary
    metrics = {
        "same_intent_similarity_mean": sum(same_intent_sims) / max(1, len(same_intent_sims)),
        "diff_intent_similarity_mean": sum(diff_intent_sims) / max(1, len(diff_intent_sims)),
        "temporal_smooth_mse_mean": sum(temporal_smooths) / max(1, len(temporal_smooths)),
        "sample_count": len(same_intent_sims),
        "note": "small test dataset (20 samples from 2 worker manifests)",
    }
    
    print("\n=== Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    # Save
    out_path = Path("/tmp/semantic_align_test_metrics.json")
    write_json(out_path, metrics)
    print(f"\nSaved to {out_path}")

print("\n=== Done ===")
