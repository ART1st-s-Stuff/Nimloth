#!/usr/bin/env python3
"""Phase 3 semantic alignment training - proper version with checkpoint saving."""

import torch, sys, os, time
sys.path.insert(0, '.')
os.chdir('/home/jincai_guo/atst/flower')

from hydra import compose, initialize_config_dir
from pathlib import Path
from torch.utils.data import DataLoader
from src.data.semantic_dataset import SemanticAlignDataset
from src.wm.encoder import build_wm_image_encoder
from src.vlm.qwen_adapter import QwenVLMAdapter
from src.vlm.semantic_state import SemanticStateGenerator
from src.vlm.semantic_align import DeltaProjector, SemanticAlignModel
from src.utils.run_output import build_run_output_dir, write_run_status

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

# Disable wandb
os.environ['WANDB_MODE'] = 'disabled'
os.environ.pop('WANDB_API_KEY', None)

cfg_path = Path('configs').absolute()
with initialize_config_dir(version_base=None, config_dir=str(cfg_path)):
    cfg = compose(config_name='config')
    wm_cfg = cfg.wm
    dataset_cfg = cfg.dataset
    train_cfg = cfg.pipeline.train.semantic_align

    cache_path = '/home/jincai_guo/atst/flower/datasets/ai2thor/train/2026-04-24_14-47-16/2026-04-24_14-47-16.latents.cfm_dinov2m'
    manifest_path = 'datasets/ai2thor/train/2026-04-24_14-47-16'

    print('Building encoder...', file=sys.stderr, flush=True)
    encoder = build_wm_image_encoder(wm_cfg=wm_cfg)
    print('Creating dataset...', file=sys.stderr, flush=True)
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

    print(f'Starting training: {len(dataset)} samples, {len(loader)} batches, {train_cfg.epochs} epochs', file=sys.stderr, flush=True)

    # Setup output directory
    run_dir = build_run_output_dir(['models', 'semantic_align', 'cfm_dinov2m'])
    write_run_status(run_dir, 'running')
    print(f'Output: {run_dir}', file=sys.stderr, flush=True)

    total_epochs = int(train_cfg.epochs)
    for epoch in range(total_epochs):
        t0 = time.time()
        running_loss, running_nce, running_temporal = 0.0, 0.0, 0.0
        for batch_idx, batch in enumerate(loader):
            step_metrics = align_model.train_step(batch)
            running_loss += step_metrics['loss']
            running_nce += step_metrics['loss_nce']
            running_temporal += step_metrics['loss_temporal']
            # Save checkpoint every 2000 batches
            if batch_idx > 0 and batch_idx % 2000 == 0:
                ckpt = run_dir / f'checkpoint_batch{batch_idx}.pt'
                torch.save({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'projector': projector.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': running_loss / batch_idx,
                }, ckpt)
                print(f'  checkpoint at batch {batch_idx}', file=sys.stderr, flush=True)
            if batch_idx % 1000 == 0:
                elapsed = time.time() - t0
                print(f'epoch={epoch+1} batch={batch_idx}/{len(loader)} loss={step_metrics["loss"]:.4f} elapsed={elapsed:.0f}s', file=sys.stderr, flush=True)

        avg_loss = running_loss / max(1, len(loader))
        avg_nce = running_nce / max(1, len(loader))
        avg_temporal = running_temporal / max(1, len(loader))
        epoch_time = time.time() - t0
        print(f'epoch={epoch+1} done: loss={avg_loss:.4f} nce={avg_nce:.4f} temporal={avg_temporal:.4f} time={epoch_time:.1f}s', file=sys.stderr, flush=True)

    # Save final model
    final_ckpt = run_dir / 'semantic_projector.pt'
    torch.save(projector.state_dict(), final_ckpt)
    print(f'Saved: {final_ckpt}', file=sys.stderr, flush=True)

    # Save metrics
    metrics = {'loss': avg_loss, 'loss_nce': avg_nce, 'loss_temporal': avg_temporal, 'vlm_init_error': vlm_adapter.init_error}
    import json
    (run_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))

    write_run_status(run_dir, 'completed')
    print(f'Training complete! Final loss={avg_loss:.4f}', file=sys.stderr, flush=True)