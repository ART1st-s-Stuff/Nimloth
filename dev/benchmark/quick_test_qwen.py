#!/usr/bin/env python3
"""快速测试 Qwen dataset - 使用已有的缓存。"""
import sys
sys.path.insert(0, '.')
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from omegaconf import OmegaConf
from pathlib import Path

wm_cfg = OmegaConf.load('configs/wm/lewm_qwen25vl_8b.yaml')
dataset_cfg = OmegaConf.load('configs/dataset/ai2thor.yaml')

print(f"WM Config: latent_dim={wm_cfg.latent_dim}, num_patches={getattr(wm_cfg.encoder, 'num_patches', 16)}")
print(f"Expected z_history shape: [history_len={wm_cfg.history_len}, num_patches=16, token_dim={wm_cfg.latent_dim//16}]")

# 创建 dataset（使用现有缓存，不重新编码）
run_dir = Path('datasets/ai2thor/test/2026-04-24_14-47-16')

from src.train.latent_cache import build_wm_dataset_with_cache

# 使用 encoder=None 直接加载缓存
dataset, _ = build_wm_dataset_with_cache(
    run_dir=run_dir,
    wm_name=str(wm_cfg.name),
    latent_dim=int(wm_cfg.latent_dim),
    action_dim=int(dataset_cfg.action_dim),
    history_len=int(wm_cfg.history_len),
    temporal_stride=1,
    image_encoder=None,  # 不编码，使用现有缓存
    encoder_num_workers=0,
    encoder_batch_size=32,
    expected_num_patches=int(getattr(wm_cfg.encoder, 'num_patches', 0)),
    expected_token_dim=int(wm_cfg.latent_dim) // int(getattr(wm_cfg.encoder, 'num_patches', 1)) if int(getattr(wm_cfg.encoder, 'num_patches', 0)) > 0 else 0,
    lazy_mode=False,
    chunk_mode=False,
)

print(f"Dataset created: {len(dataset._training_indices)} training indices")
print(f"Latent cache size: {len(dataset._latent_cache)}")

# 限制样本数用于测试
max_samples = 50
dataset._training_indices = dataset._training_indices[:max_samples]
print(f"Limited to {max_samples} samples")

# 测试获取样本
print("\n测试获取样本...")
try:
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"z_history shape: {sample['z_history'].shape}")
    print(f"Expected: [history_len=4, num_patches=16, token_dim=8] = [4, 16, 8]")

    expected_shape = (4, 16, 8)
    if sample['z_history'].shape == expected_shape:
        print("\n✅ SUCCESS: Sample shape matches expected [B,H,P,D]")
    else:
        print(f"\n❌ FAILURE: Sample shape {sample['z_history'].shape} != expected {expected_shape}")
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n测试完成!")