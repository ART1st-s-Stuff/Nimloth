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
    FrozenTrajectoryCache,
    _load_shard,
    _validate_record,
    file_sha256,
)
from experiments.training.sft.stage3.render_frozen_wm_predictions import (
    join_cache_images,
    load_image_index,
)
from nimloth.recon.cfm.flow import (
    condition_sensitivity,
    conditional_flow_matching_loss,
    spatial_cls_condition_sensitivity,
)
from nimloth.recon.cfm.model import (
    CFMConfig,
    SpatialCLSCFMConfig,
    SpatialCLSConditionedFlowUNet,
    SpatialConditionedFlowUNet,
)
from nimloth.wm.layout import GridStateLayout

SCHEMA = "paired_spatial_cfm_v1"
CLS_SCHEMA = "paired_spatial_cls_cfm_v1"
DECODER_FAMILIES = ("spatial_grid_v1", "spatial_cls_grid_v1")
PROVENANCE = ("stage2_checkpoint", "stage2_policy_fingerprint", "stage2_config_sha256",
              "stage2_commit_marker_sha256", "stage2_grid_config_sha256", "stage2_projector_sha256",
              "dino_cache_fingerprint", "source_commit")


def load_image_uint8(path: Path) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("RGB").resize((128, 128), Image.Resampling.BICUBIC)
        return torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1).contiguous()


class PairedObservationDataset:
    """Unique trajectory/time rows, verified sealed features and exact RGB joins."""
    def __init__(
        self,
        cache_dir: Path,
        jsonl: Path,
        split: str,
        condition: str = "state",
        decoder_family: str = "spatial_grid_v1",
    ):
        if condition not in {"state", "dino"} or split not in {"train", "eval"}:
            raise ValueError("invalid condition or split")
        if decoder_family not in DECODER_FAMILIES:
            raise ValueError(f"unsupported decoder family: {decoder_family}")
        self.cache = FrozenTrajectoryCache(Path(cache_dir))
        manifest = self.cache.manifest
        provenance = manifest['identity']
        expected_tokens = 65 if decoder_family == "spatial_cls_grid_v1" else 64
        if (manifest['grid_tokens'], manifest['state_dim']) != (expected_tokens, 1024):
            raise ValueError(
                f"{decoder_family} requires exact K{expected_tokens} x 1024 cache"
            )
        if provenance.get('split') != split or provenance.get('split_sha256') != file_sha256(jsonl):
            raise ValueError("cache/JSONL split identity mismatch")
        if any(not provenance.get(key) for key in PROVENANCE):
            raise ValueError("cache missing frozen Stage2/DINO provenance")
        state_layout = None
        if decoder_family == "spatial_cls_grid_v1":
            expected_layout = GridStateLayout(
                spatial_grid_size=8,
                global_tokens=1,
                global_role="dino_cls",
            )
            try:
                state_layout = GridStateLayout.from_metadata(
                    provenance.get("state_layout") or {}
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "spatial+CLS cache is missing the exact K64+K1 state layout"
                ) from error
            if state_layout != expected_layout:
                raise ValueError(
                    "spatial+CLS cache is missing the exact K64+K1 state layout"
                )
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
        if tuple(self.conditions.shape[1:]) != (expected_tokens, 1024):
            raise ValueError("sealed cache condition tensor does not match decoder layout")
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
        if decoder_family == "spatial_cls_grid_v1":
            self.identity['decoder_family'] = decoder_family
            self.identity['state_layout'] = state_layout.metadata()

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


def decoder_family_from_payload(payload: dict) -> str:
    """Return the checkpoint family only when schema and identity agree."""

    expected = {
        SCHEMA: "spatial_grid_v1",
        CLS_SCHEMA: "spatial_cls_grid_v1",
    }.get(payload.get("schema"))
    if expected is None:
        raise ValueError("unsupported decoder checkpoint")
    actual = payload.get("identity", {}).get("decoder_family")
    # Historical K64 checkpoints predate the explicit family field. Their
    # immutable schema already identifies the spatial-only architecture.
    if actual is None and payload.get("schema") == SCHEMA:
        actual = "spatial_grid_v1"
    if actual != expected:
        raise ValueError(
            "decoder checkpoint schema/family mismatch: "
            f"schema requires {expected!r}, identity records {actual!r}"
        )
    return expected


def load_decoder(checkpoint: Path, device='cpu'):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    family = decoder_family_from_payload(payload)
    if family == "spatial_grid_v1":
        model = SpatialConditionedFlowUNet(CFMConfig(**payload['config']))
    else:
        model = SpatialCLSConditionedFlowUNet(
            SpatialCLSCFMConfig(**payload['config'])
        )
    model.load_state_dict(payload['model'], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, payload


def save_decoder(
    path,
    model,
    optimizer,
    step,
    identity,
    row_rng,
    flow_rng,
    *,
    best_val=float("inf"),
):
    pending = path.with_suffix('.tmp')
    schema = CLS_SCHEMA if model.decoder_family == "spatial_cls_grid_v1" else SCHEMA
    if identity.get("decoder_family") != model.decoder_family:
        raise ValueError("decoder identity does not match model family")
    torch.save({'schema': schema, 'config': model.config.to_metadata(),
        'identity': identity, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'step': step, 'best_val': float(best_val),
        'row_rng': row_rng.get_state(), 'flow_rng': flow_rng.get_state(),
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, pending)
    pending.replace(path)


def train(args):
    train_set = PairedObservationDataset(
        args.train_cache,
        args.train_jsonl,
        'train',
        args.condition,
        args.decoder_family,
    )
    eval_set = PairedObservationDataset(
        args.eval_cache,
        args.eval_jsonl,
        'eval',
        args.condition,
        args.decoder_family,
    )
    overlap = validate_pair(train_set, eval_set)
    if (
        args.decoder_family == 'spatial_cls_grid_v1'
        and overlap['shared_image_hashes']
    ):
        raise ValueError(
            "spatial+CLS decoder train/eval splits contain identical RGB images"
        )
    identity = {'train': train_set.identity, 'eval': eval_set.identity, 'seed': args.seed,
        'steps': args.steps, 'batch': args.batch, 'lr': 1e-4, 'weight_decay': 1e-4,
        'clip_grad_norm': 1., 'overlap': overlap,
        'decoder_family': args.decoder_family}
    if args.decoder_family == 'spatial_cls_grid_v1':
        identity.update({
            'trainable_modules': ['SpatialCLSConditionedFlowUNet'],
            'frozen_modules': [
                'Qwen', 'StateProjector', 'WorldModel', 'ValueHead', 'OutcomeHead'
            ],
            'fit_split': 'train',
            'validation_split': 'eval',
        })
    if args.preflight_only:
        print(json.dumps(identity), flush=True)
        return identity
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=bool(args.resume))
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if args.decoder_family == 'spatial_cls_grid_v1':
        model = SpatialCLSConditionedFlowUNet(SpatialCLSCFMConfig()).to(device)
    else:
        model = SpatialConditionedFlowUNet(CFMConfig(token_count=64)).to(device)
        if sum(parameter.numel() for parameter in model.parameters()) != 17565571:
            raise ValueError("spatial CFM architecture parameter count changed")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    row_rng = torch.Generator().manual_seed(args.seed)
    flow_rng = torch.Generator(device=device).manual_seed(args.seed + 1)
    start = 0
    best_val = float('inf')
    if args.resume:
        if (args.output / 'COMPLETE').exists():
            raise ValueError("cannot resume a completed decoder run")
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
        best_val = float(payload.get('best_val', float('inf')))
        run_path = args.output / 'run.json'
        if run_path.is_file() and json.loads(run_path.read_text()) != identity:
            raise ValueError("decoder resume output identity changed")
        log_path = args.output / 'metrics.jsonl'
        if log_path.is_file():
            logged_steps = [
                int(json.loads(line)['step'])
                for line in log_path.read_text().splitlines()
                if line.strip()
            ]
            if not logged_steps or logged_steps != list(range(1, logged_steps[-1] + 1)):
                raise ValueError("decoder resume log is not contiguous")
            if logged_steps[-1] != start:
                raise ValueError("decoder resume checkpoint/log step mismatch")
    if start > args.steps:
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
                sensitivity = (
                    spatial_cls_condition_sensitivity
                    if args.decoder_family == 'spatial_cls_grid_v1'
                    else condition_sensitivity
                )
                metrics['eval'] = sensitivity(
                    model,
                    eval_set.conditions[eval_indices].flatten(1),
                    eval_set.images[eval_indices],
                    device,
                    batch_size=args.batch,
                    seed=args.seed + 2,
                )
                if metrics['eval']['correct_flow_mse'] < best_val:
                    best_val = metrics['eval']['correct_flow_mse']
                    save_decoder(
                        args.output / 'best.pt', model, optimizer, step,
                        identity, row_rng, flow_rng, best_val=best_val,
                    )
                metrics['best_val_correct_flow_mse'] = best_val
            log.write(json.dumps(metrics) + '\n')
            log.flush()
            if step == 1 or step % 100 == 0:
                print(json.dumps(metrics), flush=True)
            if step % args.save_interval == 0 or step == args.steps:
                save_decoder(
                    args.output / 'latest.pt', model, optimizer, step,
                    identity, row_rng, flow_rng, best_val=best_val,
                )
    latest = args.output / 'latest.pt'
    final = args.output / 'final.pt'
    if not latest.is_file():
        raise FileNotFoundError("decoder completion requires latest.pt")
    if final.exists():
        if not final.is_file() or not os.path.samefile(latest, final):
            raise FileExistsError("final.pt does not identify latest.pt")
    else:
        os.link(latest, final)
    completion = json.dumps({'step': args.steps,
        'checkpoint': 'final.pt', 'sha256': file_sha256(args.output / 'final.pt'),
        'best_checkpoint': 'best.pt', 'best_sha256': file_sha256(args.output / 'best.pt'),
        'best_val_correct_flow_mse': best_val}) + '\n'
    complete_path = args.output / 'COMPLETE'
    complete_pending = args.output / 'COMPLETE.tmp'
    complete_pending.write_text(completion)
    complete_pending.replace(complete_path)
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=('state', 'dino'), required=True)
    parser.add_argument('--decoder-family', choices=DECODER_FAMILIES,
                        default='spatial_grid_v1')
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
