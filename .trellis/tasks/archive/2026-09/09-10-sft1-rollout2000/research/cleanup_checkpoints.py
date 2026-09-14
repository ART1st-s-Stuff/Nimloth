"""Remove this run's committed step saves only after successful verified completion."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil


def durable_json(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def reject_links(path):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f'symlink path: {path}')
    if any(p.is_symlink() for p in path.rglob('*')):
        raise ValueError(f'symlink inside checkpoint: {path}')


def validate_completed_state(state, path):
    for key in ('optimizer', 'scheduler', 'identity'):
        if not isinstance(state.get(key), dict) or not state[key]:
            raise ValueError(f'missing {key}: {path}')
    if (not isinstance(state.get('epoch'), int) or state['epoch'] < 1 or state.get('training_stage') != 'format'
            or state.get('format_objective') != 'format_answer_ce_v2'
            or state.get('latent_token_count') is not None or state.get('latent_query_mode') is not None
            or state.get('mask_latent_query_labels') is not None
            or state.get('world_size') != 8 or not state.get('lora')
            or not math.isfinite(state['best_val']) or state['step'] < 1):
        raise ValueError(f'invalid completed state: {path}')
    from nimloth.training.sft.stage1.convergence import ConvergenceState
    convergence = ConvergenceState.from_state_dict(state.get('convergence_state') or {})
    if convergence.last_epoch != state['epoch']:
        raise ValueError(f'epoch convergence cursor mismatch: {path}')
    rng = state.get('rank_rng_states')
    required = {'python', 'numpy', 'torch_cpu', 'torch_cuda'}
    if (not isinstance(rng, list) or len(rng) != 8
            or any(not isinstance(item, dict) or not required.issubset(item) for item in rng)):
        raise ValueError(f'incomplete epoch RNG state: {path}')


def validate_checkpoint(path):
    import torch
    from safetensors import safe_open
    reject_links(path)
    state = torch.load(path / 'training_state.pt', map_location='cpu', weights_only=False)
    validate_completed_state(state, path)
    for name in ('adapter_config.json', 'tokenizer_config.json', 'preprocessor_config.json'):
        json.loads((path / name).read_text())
    weights = list(path.glob('*.safetensors'))
    if not weights:
        raise ValueError(f'missing weights: {path}')
    for weight in weights:
        with safe_open(weight, framework='pt', device='cpu') as handle:
            keys = list(handle.keys())
            if not keys:
                raise ValueError(f'empty weights: {weight}')
            for key in keys:
                tensor = handle.get_tensor(key)
                if not torch.isfinite(tensor).all().item():
                    raise ValueError(f'nonfinite weight: {weight}:{key}')
    return {k: state[k] for k in ('epoch', 'step', 'identity')}


def validate_step_checkpoint(path, step, identity):
    import torch
    state = torch.load(path/'training_state.pt', map_location='cpu', weights_only=False)
    if (state.get('step') != step or state.get('identity') != identity
            or state.get('resume_schema') != 'nimloth_early_stage_resume_v1'
            or state.get('training_stage') != 'format'
            or state.get('format_objective') != 'format_answer_ce_v2'):
        raise ValueError(f'step checkpoint identity mismatch: {path}')


def cleanup(run, validator=validate_checkpoint, through_epoch=None, step_validator=validate_step_checkpoint):
    run = Path(run).absolute()
    reject_links(run)
    metadata = json.loads((run / 'launch.json').read_text())
    if metadata['run_output'] != str(run):
        raise ValueError('run ownership mismatch')
    if through_epoch is None:
        if (run / 'TRAIN_SUCCEEDED').read_text().strip() != '0':
            raise ValueError('training did not succeed')
        final = validator(run / 'final')
        completed_epoch = final.get('epoch')
    else:
        completed_epoch = through_epoch
    if not isinstance(completed_epoch, int) or completed_epoch < 1:
        raise ValueError('invalid completed epoch')
    epoch_name = f'epoch_{completed_epoch:03d}'
    epoch = validator(run / epoch_name)
    if through_epoch is None and epoch != final:
        raise ValueError('epoch/final state mismatch')
    marker = json.loads((run / epoch_name / 'COMMITTED').read_text())
    if marker != {'epoch': completed_epoch, 'step': epoch['step']}:
        raise ValueError('epoch marker mismatch')
    suffix = '' if through_epoch is None else f'_epoch_{completed_epoch:03d}'
    manifest_path = run / f'cleanup{suffix}_manifest.json'
    complete_path = run / f'cleanup{suffix}_complete.json'
    if complete_path.exists():
        return
    candidates = []
    for path in sorted(run.iterdir()):
        if not re.fullmatch(r'resume_step_[0-9]{8}', path.name):
            continue
        step = int(path.name.rsplit('_', 1)[1])
        if through_epoch is not None and step > epoch['step']:
            continue
        if not path.is_dir():
            raise ValueError(f'not a checkpoint directory: {path}')
        reject_links(path)
        marker = json.loads((path / 'COMMITTED').read_text())
        step = int(path.name.rsplit('_', 1)[1])
        if (marker.get('step') != step or marker.get('schema') != 'nimloth_early_stage_resume_v1'
                or step <= 0 or step > epoch['step']):
            raise ValueError(f'invalid step marker: {path}')
        step_validator(path, step, epoch['identity'])
        files = []
        for f in sorted(path.rglob('*')):
            if f.is_file():
                digest = hashlib.sha256()
                with f.open('rb') as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                        digest.update(block)
                files.append({'path': str(f.relative_to(run)), 'size': f.stat().st_size, 'sha256': digest.hexdigest()})
        candidates.append({'path': str(path), 'files': files})
    durable_json(manifest_path, {'retained': [epoch_name, 'final', 'best'], 'verified_state': epoch, 'removed_candidates': candidates})
    for candidate in candidates:
        path = Path(candidate['path'])
        reject_links(path)
        shutil.rmtree(path)
    durable_json(complete_path, {'removed': [c['path'] for c in candidates]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--through-epoch', type=int)
    args = parser.parse_args()
    cleanup(args.run, through_epoch=args.through_epoch)
