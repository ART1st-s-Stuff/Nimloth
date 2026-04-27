#!/usr/bin/env python3
"""Quick training smoke test - writes progress to stderr for immediate visibility."""

import torch, sys, os, time
sys.path.insert(0, '.')
os.chdir('/home/jincai_guo/atst/flower')

from hydra import compose, initialize_config_dir
from pathlib import Path
from torch.utils.data import DataLoader
from src.data.semantic_dataset import SemanticAlignDataset
from src.wm.encoders import build_wm_image_encoder
from src.vlm.qwen_adapter import QwenVLMAdapter
from src.vlm.semantic_state import SemanticStateGenerator
from src.vlm.semantic_align import DeltaProjector, SemanticAlignModel

def _collate(batch):
    return {
        'z_t': torch.stack([item['z_t'] for item in batch], dim=0),
        'z_t_pos': torch.stack([item['z_t_pos'] for item in batch], dim=0),
        'z_t_neg': torch.stack([item['z_t_neg'] for item in batch], dim=0),
        'image_path': [str(item['image_path']) for item in batch],
        'pos_image_path': [str(item['pos_image_path']) for item in batch],
        'task_text': [str(item['task_text']) for item in batch],
        'env_context': [str(item['env_context']) for item in batch],
    }

cfg_path = Path('configs').absolute()
with initialize_config_dir(version_base=None, config_dir=str(cfg_path)):
    cfg = compose(config_name='config')
    wm_cfg = cfg.wm
    dataset_cfg = cfg.dataset

    cache_path = '/home/jincai_guo/atst/flower/datasets/ai2thor/train/2026-04-24_14-47-16/2026-04-24_14-47-16.latents.cfm_dinov2m'
    manifest_path = 'datasets/ai2thor/train/2026-04-24_14-47-16'

    encoder = build_wm_image_encoder(wm_cfg=wm_cfg)
    dataset = SemanticAlignDataset(
        manifest_path=manifest_path,
        latent_dim=int(wm_cfg.latent_dim),
        action_dim=int(dataset_cfg.action_dim),
        history_len=int(wm_cfg.history_len),
        image_encoder=encoder,
        latent_cache_path=cache_path,
        positive_k=1, negative_gap=6,
    )
    dataset._wm_dataset.disable_encoder_after_warmup()

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=_collate)

    vlm_adapter = QwenVLMAdapter(model_name='Qwen/Qwen2.5-VL-7B-Instruct',
                                   latent_dim=int(wm_cfg.latent_dim),
                                   enabled=False, fallback_enabled=True, max_new_tokens=128)
    semantic_gen = SemanticStateGenerator(vlm_adapter=vlm_adapter)
    device = torch.device('cuda')
    projector = DeltaProjector(latent_dim=6144, hidden_dim=512).to(device)
    optimizer = torch.optim.Adam(projector.parameters(), lr=0.001)
    align_model = SemanticAlignModel(projector=projector, semantic_generator=semantic_gen,
                                     optimizer=optimizer, device=device, temporal_weight=0.2)
    align_model._temperature = 0.07

    print(f'Dataset: {len(dataset)}, batches: {len(loader)}', flush=True)

    t0 = time.time()
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= 100:
            break
        metrics = align_model.train_step(batch)
        if batch_idx % 20 == 0:
            print(f'batch {batch_idx}: loss={metrics["loss"]:.4f}, elapsed={time.time()-t0:.0f}s', flush=True)

    elapsed = time.time() - t0
    print(f'Done: 100 batches in {elapsed:.1f}s ({elapsed/100:.2f}s/batch)', flush=True)