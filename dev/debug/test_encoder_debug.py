#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from omegaconf import OmegaConf
from pathlib import Path

wm_cfg = OmegaConf.load('configs/wm/lewm_qwen25vl_8b.yaml')
dataset_cfg = OmegaConf.load('configs/dataset/ai2thor.yaml')

from src.wm.encoder import build_wm_image_encoder
encoder = build_wm_image_encoder(wm_cfg)

from src.train.latent_cache import build_wm_dataset_with_cache

run_dir = Path('datasets/ai2thor/test/2026-04-24_14-47-16')
dataset, cache_dir = build_wm_dataset_with_cache(
    run_dir=run_dir,
    wm_name=str(wm_cfg.name),
    latent_dim=int(wm_cfg.latent_dim),
    action_dim=int(dataset_cfg.action_dim),
    history_len=int(wm_cfg.history_len),
    temporal_stride=1,
    image_encoder=encoder,
    encoder_num_workers=2,
    encoder_batch_size=16,
    expected_num_patches=int(getattr(wm_cfg.encoder, 'num_patches', 0)),
    expected_token_dim=int(wm_cfg.latent_dim) // int(getattr(wm_cfg.encoder, 'num_patches', 1)) if int(getattr(wm_cfg.encoder, 'num_patches', 0)) > 0 else 0,
    lazy_mode=False,
    chunk_mode=False,
)

unique_paths = {str(sample['image_path']) for sample in dataset.samples if 'image_path' in sample}
print('Total images:', len(unique_paths))

# 测试编码器
test_path = list(unique_paths)[0]
print('Test path:', test_path)
import time
start = time.time()
result = encoder.encode_image_path(test_path)
print('Encode time:', time.time() - start)
print('Result shape:', result.z.shape)