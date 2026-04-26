"""Test full eval loop with progress tracking."""
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

print("=== Testing eval loop ===")

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
    
    print("Creating dataset...")
    start = time.time()
    dataset = SemanticAlignDataset(
        manifest_path=str(resolved_eval_manifest),
        latent_dim=int(wm_cfg.latent_dim),
        action_dim=int(dataset_cfg.action_dim),
        history_len=int(wm_cfg.history_len),
        image_encoder=None,
        latent_cache_path=str(resolved_cache) if resolved_cache else None,
    )
    print(f"Dataset created in {time.time() - start:.1f}s")
    print(f"Dataset size: {len(dataset)}")
    print(f"Latent cache size: {len(dataset._wm_dataset._latent_cache)}")
    
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=_collate_semantic_batch)
    print(f"DataLoader created, batches: {len(loader)}")
    
    # Load model
    print("\nLoading model...")
    model = DeltaProjector(latent_dim=int(wm_cfg.latent_dim), hidden_dim=int(wm_cfg.hidden_dim)).to(device)
    ckpt_path = Path("/home/jincai_guo/atst/flower/models/semantic_align/cfm_dinov2m/2026-04-26_16-05-48/checkpoint_batch14000.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["projector"])
    model.eval()
    print("Model loaded.")
    
    # VLM adapter
    print("Creating VLM adapter...")
    adapter = QwenVLMAdapter(
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        latent_dim=int(wm_cfg.latent_dim),
        enabled=False,  # Fallback mode
        fallback_enabled=True,
        max_new_tokens=128,
    )
    print(f"VLM adapter init_error: {adapter.init_error}")
    
    semantic_generator = SemanticStateGenerator(vlm_adapter=adapter)
    
    # Run eval loop
    print("\n=== Running eval loop ===")
    same_intent_sims = []
    diff_intent_sims = []
    
    batch_times = []
    with torch.no_grad():
        iterator = iter(loader)
        for batch_idx in range(10):  # 只跑 10 个 batch
            batch_start = time.time()
            try:
                batch = next(iterator)
            except StopIteration:
                print("DataLoader exhausted")
                break
            
            z_t = batch["z_t"].to(device)
            z_t_pos = batch["z_t_pos"].to(device)
            z_t_neg = batch["z_t_neg"].to(device)
            
            pred_pos = model(z_t=z_t, z_tp=z_t_pos)
            pred_neg = model(z_t=z_t, z_tp=z_t_neg)
            
            # VLM inference (只处理第一个样本)
            start = time.time()
            out_t = semantic_generator.infer(
                image_path=batch["image_path"][0],
                history_image_paths=[batch["image_path"][0]],
                task_text=batch["task_text"][0],
                env_context=batch["env_context"][0],
            )
            vlm_time = time.time() - start
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            s_t = torch.nn.functional.normalize(out_t.s_t, dim=0)
            sim_same = torch.sum(s_t * torch.nn.functional.normalize(pred_pos[0].cpu(), dim=0)).item()
            sim_diff = torch.sum(s_t * torch.nn.functional.normalize(pred_neg[0].cpu(), dim=0)).item()
            
            same_intent_sims.append(float(sim_same))
            diff_intent_sims.append(float(sim_diff))
            
            print(f"Batch {batch_idx}: time={batch_time:.2f}s (model={batch_time - vlm_time:.2f}s, vlm={vlm_time:.2f}s), sim_same={sim_same:.4f}, sim_diff={sim_diff:.4f}")
    
    print(f"\n=== Summary ===")
    print(f"Avg batch time: {sum(batch_times) / len(batch_times):.2f}s")
    print(f"Avg same_sim: {sum(same_intent_sims) / len(same_intent_sims):.4f}")
    print(f"Avg diff_sim: {sum(diff_intent_sims) / len(diff_intent_sims):.4f}")

print("\n=== Done ===")
