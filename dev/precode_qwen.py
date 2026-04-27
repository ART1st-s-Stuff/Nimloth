#!/usr/bin/env python3
"""预编码足够数量的 Qwen latent 用于测试。"""
import sys
sys.path.insert(0, '.')
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from omegaconf import OmegaConf
from pathlib import Path
import torch
import time

wm_cfg = OmegaConf.load('configs/wm/lewm_qwen25vl_8b.yaml')

print(f"WM Config: latent_dim={wm_cfg.latent_dim}")

from src.wm.encoder import build_wm_image_encoder
encoder = build_wm_image_encoder(wm_cfg)

# 缓存路径
run_dir = Path('datasets/ai2thor/test/2026-04-24_14-47-16')
cache_path = run_dir / f"{run_dir.name}.latents.{wm_cfg.name}.pt"

# 加载现有缓存
existing_latents = {}
if cache_path.exists():
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    existing_latents = payload.get("latents", {})
    print(f"发现现有缓存: {len(existing_latents)} 张图片")

# 获取 manifest 中的图片路径
from src.data.dataset import read_worker_manifests
samples = read_worker_manifests(run_dir)
image_paths = sorted(set(str(s["image_path"]) for s in samples if "image_path" in s))
print(f"Manifest 中共有 {len(image_paths)} 张图片")

# 找出缺失的图片
missing_paths = [p for p in image_paths if p not in existing_latents]
print(f"缺失 {len(missing_paths)} 张图片")

# 预编码 200 张图片用于测试（应该足够覆盖前 50 个样本的需求）
num_to_encode = 200
if len(missing_paths) > num_to_encode:
    missing_paths = missing_paths[:num_to_encode]

print(f"将编码 {len(missing_paths)} 张图片...")

# 编码
new_latents = {}
batch_size = 32
for start_idx in range(0, len(missing_paths), batch_size):
    batch = missing_paths[start_idx:start_idx + batch_size]
    t0 = time.time()
    outputs = encoder.encode_image_paths(batch)
    for path, output in zip(batch, outputs, strict=True):
        new_latents[path] = output.z.detach().cpu()
    done = start_idx + len(batch)
    print(f"  {done}/{len(missing_paths)} ({time.time()-t0:.2f}s)")

# 合并缓存
existing_latents.update(new_latents)

# 保存
print(f"保存缓存到 {cache_path}...")
torch.save({"latent_dim": wm_cfg.latent_dim, "latents": existing_latents}, cache_path)
print(f"完成! 共 {len(existing_latents)} 张图片")