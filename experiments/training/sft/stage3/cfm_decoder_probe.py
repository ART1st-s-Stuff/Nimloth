"""Matched frozen-state/DINO spatial CFM readout; no upstream model is loaded."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.training.sft.stage3.frozen_wm_diagnostic import (
    FrozenTrajectoryCache, file_sha256, _load_shard, _validate_record)
from experiments.training.sft.stage3.render_frozen_wm_predictions import load_image_index, join_cache_images
from nimloth.recon.cfm.model import CFMConfig, SpatialConditionedFlowUNet
from nimloth.recon.cfm.flow import conditional_flow_matching_loss, condition_sensitivity

SCHEMA = "paired_spatial_cfm_v1"
PROVENANCE = ("stage2_checkpoint", "stage2_policy_fingerprint", "stage2_config_sha256",
              "stage2_commit_marker_sha256", "stage2_grid_config_sha256", "stage2_projector_sha256",
              "dino_cache_fingerprint", "source_commit")


def load_image_uint8(path: Path) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("RGB").resize((128, 128), Image.Resampling.BICUBIC)
        return torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1).contiguous()


class PairedObservationDataset:
    """Unique trajectory/time rows, verified sealed features and exact RGB joins."""
    def __init__(self, cache_dir: Path, jsonl: Path, split: str, condition: str = "state"):
        if condition not in {"state", "dino"} or split not in {"train", "eval"}:
            raise ValueError("invalid condition or split")
        self.cache = FrozenTrajectoryCache(Path(cache_dir))
        manifest = self.cache.manifest
        provenance = manifest['identity']
        if (manifest['grid_tokens'], manifest['state_dim']) != (64, 1024):
            raise ValueError("CFM requires exact K64 x 1024 cache")
        if provenance.get('split') != split or provenance.get('split_sha256') != file_sha256(jsonl):
            raise ValueError("cache/JSONL split identity mismatch")
        if any(not provenance.get(key) for key in PROVENANCE):
            raise ValueError("cache missing frozen Stage2/DINO provenance")
        images, _ = load_image_index(jsonl)
        joined = join_cache_images(self.cache, images, prediction_horizon=manifest['prediction_horizon'])
        features = {}
        for shard in manifest['shards']:
            shard_path = self.cache.directory / shard['path']
            payload = _load_shard(shard_path)
            for record in payload['records']:
                _validate_record(record, horizon=manifest['prediction_horizon'], path=shard_path)
                key = record['trajectory_id']
                if key in features:
                    raise ValueError('duplicate trajectory in cache shards')
                features[key] = record['states' if condition == 'state' else 'dino'].detach()
            del payload, record
        if set(features) != {row['trajectory'] for row in joined}:
            raise ValueError('cache shard/index identity mismatch')
        self.keys, paths, condition_rows = [], [], []
        for entry in sorted(joined, key=lambda item: item['trajectory']):
            feature = features.pop(entry['trajectory'])
            if len(feature) != entry['state_count']:
                raise ValueError('cache shard/index state count mismatch')
            condition_rows.append(feature)
            for step, path in enumerate(entry['image_paths']):
                self.keys.append((entry['trajectory'], step))
                paths.append(path)
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("duplicate observation identity")
        self.conditions = torch.cat(condition_rows).float().contiguous()
        del condition_rows
        self.images = torch.stack([load_image_uint8(path) for path in paths])
        self.image_hashes = tuple(file_sha256(path) for path in paths)
        rows = list(zip(self.keys, self.image_hashes, strict=True))
        self.identity = {'cache_manifest_sha256': self.cache.manifest_sha256,
            'split_sha256': file_sha256(jsonl), 'split': split, 'condition': condition,
            'observation_count': len(self.keys), 'rows_sha256': hashlib.sha256(
                json.dumps(rows, separators=(',', ':')).encode()).hexdigest(),
            'provenance': {key: provenance[key] for key in PROVENANCE},
            'rgb_preprocessing': 'RGB_full_resize_bicubic_128_uint8_to_minus1_plus1'}

    def batch(self, indices, device):
        return (self.conditions[indices].flatten(1).to(device),
                self.images[indices].to(device=device, dtype=torch.float32).div(127.5).sub(1.))


def validate_pair(train: PairedObservationDataset, evaluation: PairedObservationDataset):
    if train.identity['provenance'] != evaluation.identity['provenance']:
        raise ValueError("train/eval frozen feature provenance differs")
    if {key[0] for key in train.keys} & {key[0] for key in evaluation.keys}:
        raise ValueError("train/eval trajectory overlap")
    # Pixel duplicates are reported, not relabeled as independent held-out scenes.
    return {'shared_image_hashes': len(set(train.image_hashes) & set(evaluation.image_hashes))}


def load_decoder(checkpoint: Path, device='cpu'):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload.get('schema') != SCHEMA:
        raise ValueError("unsupported decoder checkpoint")
    model = SpatialConditionedFlowUNet(CFMConfig(**payload['config']))
    model.load_state_dict(payload['model'], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, payload


def save_decoder(path, model, optimizer, step, identity, row_rng, flow_rng):
    pending = path.with_suffix('.tmp')
    torch.save({'schema': SCHEMA, 'config': model.config.to_metadata(),
        'identity': identity, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'step': step, 'row_rng': row_rng.get_state(), 'flow_rng': flow_rng.get_state(),
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, pending)
    pending.replace(path)


def train(args):
    train_set = PairedObservationDataset(args.train_cache, args.train_jsonl, 'train', args.condition)
    eval_set = PairedObservationDataset(args.eval_cache, args.eval_jsonl, 'eval', args.condition)
    overlap = validate_pair(train_set, eval_set)
    identity = {'train': train_set.identity, 'eval': eval_set.identity, 'seed': args.seed,
        'steps': args.steps, 'batch': args.batch, 'lr': 1e-4, 'weight_decay': 1e-4,
        'clip_grad_norm': 1., 'overlap': overlap, 'decoder_family': 'spatial_grid_v1'}
    if args.preflight_only:
        print(json.dumps(identity), flush=True)
        return identity
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=bool(args.resume))
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = SpatialConditionedFlowUNet(CFMConfig(token_count=64)).to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != 17565571:
        raise ValueError("spatial CFM architecture parameter count changed")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    row_rng = torch.Generator().manual_seed(args.seed)
    flow_rng = torch.Generator(device=device).manual_seed(args.seed + 1)
    start = 0
    if args.resume:
        restored, payload = load_decoder(args.resume, device)
        if payload['identity'] != identity:
            raise ValueError("decoder resume identity changed")
        model.load_state_dict(restored.state_dict())
        del restored
        optimizer.load_state_dict(payload['optimizer'])
        row_rng.set_state(payload['row_rng'])
        flow_rng.set_state(payload['flow_rng'])
        torch.set_rng_state(payload['torch_rng'])
        if payload['cuda_rng']:
            torch.cuda.set_rng_state_all(payload['cuda_rng'])
        start = payload['step']
    if start >= args.steps:
        raise ValueError("checkpoint already reached training budget")
    (args.output / 'run.json').write_text(json.dumps(identity, indent=2) + '\n')
    # Same deterministic subset and row/noise schedule in both decoder arms.
    eval_indices = torch.randperm(len(eval_set.keys), generator=torch.Generator().manual_seed(args.seed))[:256]
    with (args.output / 'metrics.jsonl').open('a') as log:
        for step in range(start + 1, args.steps + 1):
            begin = time.monotonic()
            model.train()
            indices = torch.randint(len(train_set.keys), (args.batch,), generator=row_rng)
            condition, images = train_set.batch(indices, device)
            optimizer.zero_grad(set_to_none=True)
            loss = conditional_flow_matching_loss(model, images, condition, generator=flow_rng)
            if not torch.isfinite(loss):
                raise ValueError("non-finite CFM loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            metrics = {'step': step, 'flow_loss': float(loss.detach()),
                'grad_norm': float(norm), 'seconds': time.monotonic() - begin}
            if step % args.eval_interval == 0 or step == args.steps:
                metrics['eval'] = condition_sensitivity(model,
                    eval_set.conditions[eval_indices].flatten(1), eval_set.images[eval_indices],
                    device, batch_size=args.batch, seed=args.seed + 2)
            log.write(json.dumps(metrics) + '\n')
            log.flush()
            if step == 1 or step % 100 == 0:
                print(json.dumps(metrics), flush=True)
            if step % args.save_interval == 0 or step == args.steps:
                save_decoder(args.output / 'latest.pt', model, optimizer, step, identity, row_rng, flow_rng)
    os.link(args.output / 'latest.pt', args.output / 'final.pt')
    (args.output / 'COMPLETE').write_text(json.dumps({'step': args.steps,
        'checkpoint': 'final.pt', 'sha256': file_sha256(args.output / 'final.pt')}) + '\n')
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=('state', 'dino'), required=True)
    for name in ('train-cache', 'eval-cache', 'train-jsonl', 'eval-jsonl', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=20260921)
    parser.add_argument('--eval-interval', type=int, default=1000)
    parser.add_argument('--save-interval', type=int, default=1000)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if min(args.steps, args.batch, args.eval_interval, args.save_interval) < 1:
        parser.error('steps, batch and intervals must be positive')
    train(args)


if __name__ == '__main__':
    main()
