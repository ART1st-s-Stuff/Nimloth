#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from omegaconf import OmegaConf
from pathlib import Path

wm_cfg = OmegaConf.load('configs/wm/lewm_qwen25vl_8b.yaml')
dataset_cfg = OmegaConf.load('configs/dataset/ai2thor.yaml')

from src.wm.encoder import build_wm_image_encoder
encoder = build_wm_image_encoder(wm_cfg)

# 模拟预编码前 100 张图片
test_img = 'datasets/ai2thor/train/2026-04-24_14-47-16/images/floorplan1_ep0000_step0000.png'

import time
start = time.time()
for i in range(100):
    result = encoder.encode_image_path(test_img)
elapsed = time.time() - start
print(f'Encoded 100 images in {elapsed:.3f}s ({elapsed/100*1000:.1f}ms per image)')

# 估算 512523 张图片需要的时间
estimated_total = (512523 / 16) * (elapsed / 100) / 60  # minutes
print(f'Estimated total pre-encoding time: {estimated_total:.1f} minutes')