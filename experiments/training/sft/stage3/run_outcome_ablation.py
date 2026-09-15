"""Bounded remote-host controller for the reviewed eight-rank outcome A/B run.

Default is command-only dry run. Execute only on the authorized training host;
this controller never establishes SSH or retries a phase. Explicit cleanup flags
permit only verified intermediate checkpoints created by this run.
"""
from __future__ import annotations

import argparse
import csv
import math
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import re

ARM_SECONDS = 12 * 3600


def command(args, arm, phase, port):
    output = args.run_root / (arm + ('_canary' if phase != 'formal' else ''))
    values = {
        'model': args.model, 'train-jsonl': args.train, 'val-jsonl': args.val,
        'preprocess-cache-dir': args.preprocess, 'preprocess-cache-processor-source': args.model,
        'dino-grid-cache': args.dino, 'output-dir': output, 'epochs': 1,
        'distributed-strategy': 'fsdp', 'fsdp-wrap-granularity': getattr(args, 'fsdp_wrap_granularity', 'linear'), 'batch-size': 1, 'grad-accum': 8, 'seed': 42, 'history-size': 1, 'prediction-horizon': 4,
        'grid-size': 8, 'latent-token-count': 64, 'llm-tune': 'full', 'vision-tune': 'full',
        'query-tune': 'selected_rows', 'query-lr': 1e-4, 'protocol-lr': 2e-5,
        'lr-qwen-start': 2e-6, 'lr-qwen-peak': 2e-6, 'state-proj-lr': 8e-5,
        'wm-predictor-lr': 3e-4, 'value-head-lr': 1e-4, 'outcome-head-lr': 1e-4,
        'lambda-outcome': int(arm == 'treatment'), 'max-length': args.max_length,
        'checkpoint-interval-steps': 10, 'checkpoint-keep-last': 2,
        'checkpoint-interval-minutes': 0, 'step-timing-interval': 1,
        'step-timing-sample-interval': getattr(args, 'step_timing_sample_interval', 10) if phase == 'formal' else 1,
        'wandb-run-name': f'{args.run_root.name}_{arm}_{phase}',
    }
    result = [str(args.python), '-m', 'torch.distributed.run', '--nproc_per_node=8',
              f'--master_port={port}', '-m', 'nimloth.training.sft.stage3',
              '--config', str(args.worktree / 'configs/training/sft2/action_outcome_k64_h1_t4.yaml')]
    for key, value in values.items():
        result.extend(['--' + key, str(value)])
    result.extend(['--outcome-head', '--require-prebuilt-cache', '--step-timing', '--deduplicate-epoch-checkpoints'])
    if phase == 'formal':
        if arm == 'control' and getattr(args, 'resume_control_from', None):
            result.extend(['--resume', '--resume-from', str(args.resume_control_from)])
        result.extend(['--outcome-eval-dir', str(args.run_root / (arm + '_evaluation'))])
    else:
        result.extend(['--stop-after-steps', '1' if phase == 'canary' else '2'])
        if phase == 'resume':
            result.extend(['--resume', '--resume-from', str(output / 'stop_step_000001')])
        elif arm == 'treatment':
            result.append('--diagnose-outcome-gradients')
    return result


def free_port():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        return listener.getsockname()[1]


def resources(args, *, required_gib=None):
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'],
                            check=True, capture_output=True, text=True, timeout=20).stdout
    devices = [line.split(',')[0].strip() for line in result.splitlines() if line.strip()]
    if devices != [str(index) for index in range(8)]:
        raise RuntimeError('exactly eight visible physical GPUs are required')
    processes = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'],
                               check=True, capture_output=True, text=True, timeout=20).stdout.strip()
    if processes:
        raise RuntimeError(f'existing GPU compute processes must remain untouched: {processes}')
    free = shutil.disk_usage(args.run_root.parent).free / 1024**3
    required = args.min_free_gib if required_gib is None else required_gib
    if free < required:
        raise RuntimeError(f'insufficient disk: {free:.1f} GiB free, require {required:.1f}')
    return {'gpu_query': result, 'free_gib': free, 'required_gib': required}


def terminate_group(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=20)


def run_process(argv, *, cwd, environment, log_path, timeout):
    start = time.monotonic()
    with log_path.open('x') as log:
        process = subprocess.Popen(argv, cwd=cwd, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
            if code:
                raise RuntimeError(f'phase exited {code}; see {log_path}')
        except BaseException:
            terminate_group(process)
            raise
    return time.monotonic() - start


def verify_phase(args, arm, phase):
    root = args.run_root / (arm + ('_canary' if phase != 'formal' else ''))
    if phase != 'formal':
        step = 1 if phase == 'canary' else 2
        checkpoint = root / f'stop_step_{step:06d}'
        marker = json.loads((checkpoint / 'STOPPED').read_text())
        if marker['step'] != step or marker['epoch_complete']:
            raise RuntimeError('canary stop marker is inconsistent')
    else:
        checkpoint = root / 'epoch_001'
        for split in ('train', 'eval'):
            manifest = json.loads((args.run_root / (arm + '_evaluation') / f'epoch_001_{split}.complete.json').read_text())
            if manifest.get('status') != 'complete' or len(manifest['files']) != 8:
                raise RuntimeError('formal export is incomplete')
    with (root / 'train_step_log.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError('training produced no step metrics')
    for row in rows:
        for required_metric in ('total_loss', 'wm_mse', 'dino_grid_mse', 'value_mc_mse'):
            if not row.get(required_metric):
                raise RuntimeError(f'missing required training metric: {required_metric}')
        if arm == 'treatment' and not row.get('outcome_bce'):
            raise RuntimeError('missing treatment outcome BCE')
        for key in ('total_loss', 'wm_mse', 'dino_grid_mse', 'value_mc_mse', 'lm_ce', 'outcome_bce'):
            if row.get(key) and not math.isfinite(float(row[key])):
                raise RuntimeError(f'non-finite {key} in training log')
    required = ['training_state.pt' , 'state_proj.pt', 'outcome_head.pt', 'wm_predictor/predictor.pt',
                'value_head/value_head.pt', *[f'history_cache_rank_{rank:03d}.pt' for rank in range(8)]]
    for relative in required:
        if not (checkpoint / relative).is_file():
            raise RuntimeError(f'incomplete checkpoint: {checkpoint / relative}')
    metadata = checkpoint_metadata(args, checkpoint)
    if (metadata.get('epoch') != 1 or not metadata.get('has_optimizer')
            or metadata.get('epoch_complete') is not (phase == 'formal')
            or (phase != 'formal' and metadata.get('step') != step)):
        raise RuntimeError('checkpoint training state disagrees with completed phase')
    if not (checkpoint / 'selected_token_rows.pt').is_file():
        raise RuntimeError('checkpoint missing exact selected token rows')
    return str(checkpoint)


def checkpoint_bytes(checkpoint):
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise RuntimeError('checkpoint must be a real directory')
    files = list(checkpoint.rglob('*'))
    if any(path.is_symlink() for path in files):
        raise RuntimeError('checkpoint tree contains a symlink')
    return sum(path.stat().st_size for path in files if path.is_file())


def checkpoint_metadata(args, checkpoint):
    script = """import json,sys,torch
from pathlib import Path
p=Path(sys.argv[1]); s=torch.load(p/'training_state.pt',map_location='cpu',weights_only=False)
keys=('step','epoch','epoch_complete','micro_step_in_epoch','training_invariants')
r={k:s.get(k) for k in keys};r['has_optimizer']=s.get('optimizer') is not None
print(json.dumps(r))
"""
    result = subprocess.run([str(args.python), '-c', script, str(checkpoint)],
                            check=True, capture_output=True, text=True, timeout=120)
    return json.loads(result.stdout)


def cleanup_checkpoint(args, checkpoint, controller, *, expected_step, final_step, expected_invariants):
    """Delete only complete intermediate children of this run after audit flush."""
    checkpoint = Path(checkpoint)
    allowed_parents = {args.run_root / arm for arm in ('control','treatment','control_canary','treatment_canary')}
    if checkpoint.parent not in allowed_parents or checkpoint.parent.is_symlink():
        raise RuntimeError('cleanup path is outside the owned run')
    if not re.fullmatch(r'(?:stop_step|step)_[0-9]{6,}', checkpoint.name):
        raise RuntimeError('only intermediate checkpoints may be cleaned')
    checkpoint_bytes(checkpoint)
    required = ['training_state.pt','state_proj.pt','outcome_head.pt','wm_predictor/predictor.pt','value_head/value_head.pt',
                *[f'history_cache_rank_{rank:03d}.pt' for rank in range(8)]]
    if any(not (checkpoint / name).is_file() for name in required):
        raise RuntimeError('refusing incomplete intermediate cleanup')
    metadata = checkpoint_metadata(args, checkpoint)
    if metadata['step'] != expected_step or expected_step > final_step or metadata['epoch_complete'] is not False or not metadata['has_optimizer']:
        raise RuntimeError('intermediate metadata does not match validated recovery boundary')
    if not expected_invariants or metadata.get('training_invariants') != expected_invariants:
        raise RuntimeError('foreign intermediate training identity')
    if checkpoint.name.startswith('stop_'):
        stopped = json.loads((checkpoint/'STOPPED').read_text())
        if stopped['step'] != expected_step or stopped['epoch_complete']:
            raise RuntimeError('stopped marker mismatch')
    hashes = {}
    for path in sorted(checkpoint.rglob('*')):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(4*1024**2), b''):
                    digest.update(block)
            hashes[str(path.relative_to(checkpoint))] = {'sha256':digest.hexdigest(),'bytes':path.stat().st_size}
    audit = controller / (checkpoint.parent.name + '_' + checkpoint.name + '_cleanup.json')
    with audit.open('x') as stream:
        json.dump({'checkpoint':str(checkpoint),'metadata':metadata,'files':hashes,'validated_final_step':final_step},stream,indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    shutil.rmtree(checkpoint)


def phases(args):
    if getattr(args, 'resume_control_from', None):
        return [('control', 'formal'), ('treatment', 'formal')]
    return [('control', 'canary'), ('control', 'resume'), ('treatment', 'canary'),
            ('treatment', 'resume'), ('control', 'formal'), ('treatment', 'formal')]


def semantic_command(argv):
    """Compare every recorded argument except explicit operational overrides."""
    ignored = {'--output-dir', '--outcome-eval-dir', '--wandb-run-name', '--resume-from',
               '--step-timing-interval', '--step-timing-sample-interval', '--fsdp-wrap-granularity'}
    result = []
    index = 0
    while index < len(argv):
        value = argv[index]
        if value in ignored:
            index += 2
            continue
        if value in {'--resume', '--step-timing'} or value.startswith('--master_port='):
            index += 1
            continue
        result.append(value)
        index += 1
    return result


def validate_control_resume(args):
    checkpoint = args.resume_control_from
    if checkpoint.parent.name != 'control' or not re.fullmatch(r'step_[0-9]{6,}', checkpoint.name):
        raise RuntimeError('resume source must be a control intermediate checkpoint')
    checkpoint_bytes(checkpoint)
    old_root = checkpoint.parent.parent
    if old_root.resolve() == args.run_root.resolve():
        raise RuntimeError('resume requires a new output root')
    # Refuse a live old controller or trainer, including a controller interrupted
    # before it could update progress.json. Do not rely on stale status alone.
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            tokens = proc.read_bytes().decode().split('\0')
        except (OSError, UnicodeDecodeError):
            continue
        if int(proc.parent.name) == os.getpid():
            continue
        if any(value == str(old_root) or value.startswith(str(old_root) + '/')
               for token in tokens for value in [token.split('=', 1)[-1]]):
            raise RuntimeError('old run still has a live process')
    record = json.loads((old_root / 'controller/progress.json').read_text())
    expected_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for name, path in [('train', args.train), ('val', args.val)]}
    expected_command = semantic_command(command(args, 'control', 'formal', 29500))
    chain = []
    seen = set()
    current, root = record, old_root
    while True:
        identity = str(root.resolve())
        if identity in seen:
            raise RuntimeError('cyclic resume provenance')
        seen.add(identity)
        if current.get('dataset_sha256') != expected_hashes:
            raise RuntimeError('resume dataset hashes differ')
        if any(p['arm'] == 'treatment' and p['phase'] == 'formal' for p in current['phases']):
            raise RuntimeError('treatment formal phase has already started')
        formal = [p for p in current['phases'] if p['arm'] == 'control' and p['phase'] == 'formal']
        if len(formal) != 1 or formal[0]['status'] == 'complete':
            raise RuntimeError('expected one interrupted control formal phase')
        if semantic_command(formal[0]['argv']) != expected_command:
            raise RuntimeError('resume changes semantic training arguments')
        chain.append((current, formal[0]))
        inherited = current.get('resume_control')
        if not inherited:
            completed = {(p['arm'], p['phase']) for p in current['phases'] if p['status'] == 'complete'}
            if not {(a, p) for a in ('control', 'treatment') for p in ('canary', 'resume')} <= completed:
                raise RuntimeError('all four original canary/resume gates must be complete')
            break
        source = Path(inherited['source'])
        if source.parent.name != 'control' or not re.fullmatch(r'step_[0-9]{6,}', source.name):
            raise RuntimeError('invalid inherited control source')
        argv = formal[0]['argv']
        if '--resume-from' not in argv or Path(argv[argv.index('--resume-from') + 1]) != source:
            raise RuntimeError('inherited checkpoint differs from actual command')
        root = source.parent.parent
        original = json.loads((root / 'controller/progress.json').read_text())
        if original != inherited['original_progress']:
            raise RuntimeError('inherited progress differs from source record')
        if checkpoint_metadata(args, source) != inherited['metadata']:
            raise RuntimeError('inherited checkpoint metadata changed')
        if inherited.get('training_state_sha256') and file_sha256(source / 'training_state.pt') != inherited['training_state_sha256']:
            raise RuntimeError('inherited checkpoint hash changed')
        current = original
    required = ['training_state.pt', 'state_proj.pt', 'selected_token_rows.pt', 'outcome_head.pt', 'vision_ema.pt',
                'wm_predictor/predictor.pt', 'value_head/value_head.pt',
                *[f'history_cache_rank_{r:03d}.pt' for r in range(8)]]
    if any(not (checkpoint / name).is_file() for name in required):
        raise RuntimeError('incomplete control recovery checkpoint')
    index = checkpoint / 'model.safetensors.index.json'
    if index.is_file():
        shards = set(json.loads(index.read_text())['weight_map'].values())
        if not shards or any(not (checkpoint / shard).is_file() for shard in shards):
            raise RuntimeError('incomplete HF shards')
    elif not (checkpoint / 'model.safetensors').is_file():
        raise RuntimeError('missing HF model weights')
    metadata = checkpoint_metadata(args, checkpoint)
    if (metadata.get('epoch') != 1 or metadata.get('epoch_complete') is not False
            or metadata.get('step') != int(checkpoint.name[5:])
            or not metadata.get('has_optimizer') or not metadata.get('training_invariants')
            or not metadata.get('micro_step_in_epoch')):
        raise RuntimeError('invalid control optimizer/recovery metadata')
    # Reconstruct the original budget from the root. Charge all wall time since
    # its formal start, including interruptions; inherited counters can only
    # increase this conservative amount, never reset it.
    original, first_formal = chain[-1]
    consumed = {arm: max((p.get('arm_consumed_seconds', 0) for p in original['phases']
                         if p['arm'] == arm and p['status'] == 'complete'), default=0)
                for arm in ('control', 'treatment')}
    consumed['control'] += max(0, time.time() - first_formal['started_at'])
    for item, _ in chain:
        inherited = item.get('resume_control') or {}
        for arm, value in inherited.get('consumed_seconds', {}).items():
            if arm not in consumed or not math.isfinite(value) or value < 0:
                raise RuntimeError('invalid inherited budget')
            consumed[arm] = max(consumed[arm], value)
    if any(value >= ARM_SECONDS for value in consumed.values()):
        raise TimeoutError('original arm budget exhausted')
    return {'source': str(checkpoint), 'metadata': metadata, 'consumed_seconds': consumed,
            'training_state_sha256': file_sha256(checkpoint / 'training_state.pt'),
            'original_progress': record}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def execute(args):
    if args.run_root.exists():
        raise FileExistsError(args.run_root)
    status = subprocess.check_output(['git', 'status', '--porcelain', '--ignore-submodules=untracked'], cwd=args.worktree, text=True)
    if status.strip():
        raise RuntimeError('remote worktree must be clean and committed')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.worktree, text=True).strip()
    if commit != args.commit:
        raise RuntimeError('source commit mismatch')
    for path in (args.python, args.model, args.train, args.val, args.preprocess, args.dino):
        if not path.exists():
            raise FileNotFoundError(path)
    resume = validate_control_resume(args) if getattr(args, 'resume_control_from', None) else None
    resources(args)
    args.run_root.mkdir()
    controller = args.run_root / 'controller'
    controller.mkdir()
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', TOKENIZERS_PARALLELISM='false')
    environment['PYTHONPATH'] = str(args.worktree / 'src')
    record = {'commit': commit, 'status': 'running', 'phases': [], 'hard_seconds_per_arm': ARM_SECONDS,
              'retention': {'rolling_keep_last':2, 'cleanup_validated_canaries':args.cleanup_validated_canaries, 'cleanup_validated_intermediates':args.cleanup_validated_intermediates, 'deduplicate_epoch_checkpoints':True},
              'dataset_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in [('train', args.train), ('val', args.val)]}}
    record['resume_control'] = resume
    consumed_seconds = dict(resume['consumed_seconds']) if resume else {}
    measured_checkpoint_gib = checkpoint_bytes(args.resume_control_from) / 1024**3 if resume else None
    try:
        for arm, phase in phases(args):
            phase_started = time.monotonic()
            remaining = ARM_SECONDS - consumed_seconds.get(arm, 0.0)
            if remaining <= 0:
                raise TimeoutError(f'{arm} total deadline expired')
            required = (max(120.0, 3 * measured_checkpoint_gib) if phase == "formal" and measured_checkpoint_gib is not None
                        else max(40.0, measured_checkpoint_gib or 40.0))
            snapshot = resources(args, required_gib=required)
            argv = command(args, arm, phase, free_port())
            entry = {'arm':arm, 'phase':phase, 'argv':argv, 'resources':snapshot,
                     'environment': {key:environment[key] for key in ('CUDA_VISIBLE_DEVICES','TOKENIZERS_PARALLELISM','PYTHONPATH')},
                     'started_at':time.time(), 'status':'running'}
            record['phases'].append(entry)
            (controller / 'progress.json').write_text(json.dumps(record, indent=2))
            remaining -= time.monotonic() - phase_started
            if remaining <= 0:
                raise TimeoutError(f'{arm} total deadline expired during preflight')
            elapsed = run_process(argv, cwd=args.worktree, environment=environment,
                                  log_path=controller / f'{arm}_{phase}.log',
                                  timeout=min(remaining, 900) if phase != 'formal' else remaining)
            entry.update(status='complete', elapsed_seconds=elapsed, checkpoint=verify_phase(args, arm, phase))
            size = checkpoint_bytes(Path(entry['checkpoint'])) / 1024**3
            measured_checkpoint_gib = max(measured_checkpoint_gib or 0, size)
            entry['checkpoint_gib'] = size
            if phase == 'resume' and args.cleanup_validated_canaries:
                validated = [item for item in record['phases'] if item['arm'] == arm and item['status'] == 'complete']
                if {item['phase'] for item in validated} != {'canary', 'resume'}:
                    raise RuntimeError('canary cleanup requires both successful phases')
                resume_metadata = checkpoint_metadata(args, Path(entry['checkpoint']))
                for step in (1, 2):
                    checkpoint = args.run_root / (arm+'_canary') / f'stop_step_{step:06d}'
                    cleanup_checkpoint(args, checkpoint, controller, expected_step=step, final_step=2, expected_invariants=resume_metadata["training_invariants"])
            if phase == 'formal' and args.cleanup_validated_intermediates:
                final_meta = checkpoint_metadata(args, Path(entry['checkpoint']))
                for checkpoint in sorted((args.run_root/arm).glob('step_*')):
                    if re.fullmatch(r'step_[0-9]{6,}', checkpoint.name):
                        cleanup_checkpoint(args, checkpoint, controller, expected_step=int(checkpoint.name[5:]), final_step=final_meta['step'], expected_invariants=final_meta['training_invariants'])
            consumed_seconds[arm] = consumed_seconds.get(arm, 0.0) + time.monotonic() - phase_started
            entry['arm_consumed_seconds'] = consumed_seconds[arm]
            if consumed_seconds[arm] > ARM_SECONDS:
                raise TimeoutError(f'{arm} total deadline exceeded')
            (controller / 'progress.json').write_text(json.dumps(record, indent=2))
        record['status'] = 'complete'
        (controller / 'COMPLETE').write_text(json.dumps(record, indent=2))
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        if record['phases'] and record['phases'][-1]['status'] == 'running':
            record['phases'][-1].update(status='failed', error=failure)
        record.update(status='failed', error=failure)
        (controller / 'FAILED').write_text(json.dumps(record, indent=2))
        raise
    finally:
        (controller / 'progress.json').write_text(json.dumps(record, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('python','worktree','model','train','val','preprocess','dino','run-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--min-free-gib', type=float, default=160.0,
                        help='Initial free-space gate (at least160GiB). Later gates use measured checkpoint size.')
    parser.add_argument('--cleanup-validated-canaries', action='store_true')
    parser.add_argument('--cleanup-validated-intermediates', action='store_true')
    parser.add_argument('--max-length', type=int, default=12000)
    parser.add_argument('--resume-control-from', type=Path)
    parser.add_argument('--fsdp-wrap-granularity', choices=('linear', 'block'), default='linear')
    parser.add_argument('--step-timing-sample-interval', type=int, default=10)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args(argv)
    if args.min_free_gib < 160 or args.max_length < 1 or args.step_timing_sample_interval < 1:
        parser.error('initial disk budget must be at least160GiB and max length positive')
    if args.execute:
        def interrupted(_signum, _frame):
            raise KeyboardInterrupt("controller termination requested")
        previous = signal.signal(signal.SIGTERM, interrupted)
        try:
            execute(args)
        finally:
            signal.signal(signal.SIGTERM, previous)
    else:
        plan = {'mode':'dry_run', 'commands':[command(args, arm, phase, 29500+index)
                for index, (arm, phase) in enumerate(phases(args))],
                'disk_warning':'Initial160GiB gate; each formal requires max(120GiB,3*measured checkpoint size). Without explicit validated-canary cleanup retained tests may exceed available disk.'}
        if args.preflight:
            plan['resources'] = resources(args)
        print(json.dumps(plan, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
