"""使用小数据集 + 预加载 latents（修复路径问题）。"""
import sys
import time
import json
from pathlib import Path
import torch
from hydra import initialize_config_dir, compose
from torch.utils.data import DataLoader

from src.data.semantic_dataset import SemanticAlignDataset
from src.train.train_semantic_align import DeltaProjector, _collate_semantic_batch
from src.vlm.qwen_adapter import QwenVLMAdapter
from src.vlm.semantic_state import SemanticStateGenerator
from src.utils.io import write_json

print("=== Small dataset semantic align eval (fixed paths) ===")

config_dir = Path("/home/jincai_guo/atst/flower/configs").resolve()
with initialize_config_dir(config_dir=str(config_dir), version_base=None):
    cfg = compose(config_name="config", overrides=["wm=cfm_dinov2m"])
    wm_cfg = cfg.wm
    dataset_cfg = cfg.dataset
    train_cfg = cfg.pipeline.train.semantic_align
    
    device = torch.device(str(train_cfg.device))
    
    # 加载原始 latent cache
    cache_path = Path("/home/jincai_guo/atst/flower/datasets/ai2thor/train/2026-04-24_14-47-16/2026-04-24_14-47-16.latents.cfm_dinov2m")
    print(f"Loading cache...")
    full_cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    latents = full_cache.get("latents", {})
    print(f"Loaded {len(latents)} latents")
    
    # 创建小 manifest（使用原始相对路径）
    train_manifest = Path("/home/jincai_guo/atst/flower/datasets/ai2thor/train/2026-04-24_14-47-16")
    all_rows = []
    for wf in sorted(train_manifest.glob("manifest_worker_*.jsonl"))[:1]:  # 只取前 1 个 worker
        lines = wf.read_text().strip().split("\n")
        for line in lines[:20]:  # 只取前 20 条
            all_rows.append(json.loads(line))
    
    small_manifest_dir = Path("/tmp/semantic_align_test_manifest3")
    small_manifest_dir.mkdir(exist_ok=True)
    
    # 保持原始相对路径
    small_manifest_path = small_manifest_dir / "test_manifest.jsonl"
    small_manifest_path.write_text("\n".join(json.dumps(r) for r in all_rows))
    print(f"Created small manifest with {len(all_rows)} samples")
    
    # 构建 cache（使用相对路径作为 key）
    small_latents = {}
    for row in all_rows:
        img_path = row.get("image_path", "")
        if img_path in latents:
            small_latents[img_path] = latents[img_path]
    
    small_cache_path = small_manifest_dir / "test_latents.pt"
    torch.save({"latents": small_latents, "latent_dim": 6144}, small_cache_path)
    print(f"Created small cache with {len(small_latents)} latents")
    
    # 创建 dataset（禁用 encoder，强制使用缓存）
    print("\nCreating dataset...")
    dataset = SemanticAlignDataset(
        manifest_path=str(small_manifest_path),
        latent_dim=int(wm_cfg.latent_dim),
        action_dim=int(dataset_cfg.action_dim),
        history_len=int(wm_cfg.history_len),
        image_encoder=None,  # 不使用 encoder
        latent_cache_path=str(small_cache_path),  # 单文件缓存
    )
    print(f"Dataset size: {len(dataset)}")
    
    # 手动 warmup：将 latents 直接写入 _latent_cache
    print("Pre-loading latents...")
    dataset._wm_dataset._latent_cache = small_latents.copy()
    print(f"Warmed up with {len(small_latents)} latents")
    
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
        enabled=False,
        fallback_enabled=True,
        max_new_tokens=128,
    )
    semantic_generator = SemanticStateGenerator(vlm_adapter=adapter)
    print(f"VLM adapter init_error: {adapter.init_error}")
    
    # Run eval
    print("\n=== Running eval ===")
    same_intent_sims = []
    diff_intent_sims = []
    
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=_collate_semantic_batch)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            z_t = batch["z_t"].to(device)
            z_t_pos = batch["z_t_pos"].to(device)
            z_t_neg = batch["z_t_neg"].to(device)
            
            pred_pos = model(z_t=z_t, z_tp=z_t_pos)
            pred_neg = model(z_t=z_t, z_tp=z_t_neg)
            
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
                
                print(f"  Batch {batch_idx}, Sample {i}: sim_same={sim_same:.4f}, sim_diff={sim_diff:.4f}, vlm_time={elapsed:.3f}s")
            
            if batch_idx >= 4:
                break
    
    metrics = {
        "same_intent_similarity_mean": sum(same_intent_sims) / max(1, len(same_intent_sims)),
        "diff_intent_similarity_mean": sum(diff_intent_sims) / max(1, len(diff_intent_sims)),
        "sample_count": len(same_intent_sims),
        "note": "small test (5 batches, fallback VLM)",
    }
    
    print("\n=== Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    out_path = Path("/tmp/semantic_align_test_metrics.json")
    write_json(out_path, metrics)
    print(f"\nSaved to {out_path}")

print("\n=== Done ===")
