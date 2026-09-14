"""Bounded remote-host controller for the reviewed eight-rank outcome A/B run.

Default is command-only dry run. Execute only on the authorized training host;
this controller never establishes SSH, deletes checkpoints, or retries a phase.
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

ARM_SECONDS = 12 * 3600


def command(args, arm, phase, port):
    output = args.run_root / (arm + ('_canary' if phase != 'formal' else ''))
    values = {
        'model': args.model, 'train-jsonl': args.train, 'val-jsonl': args.val,
        'preprocess-cache-dir': args.preprocess, 'preprocess-cache-processor-source': args.model,
        'dino-grid-cache': args.dino, 'output-dir': output, 'epochs': 1,
        'batch-size': 1, 'grad-accum': 8, 'seed': 42, 'history-size': 1, 'prediction-horizon': 4,
        'grid-size': 8, 'latent-token-count': 64, 'llm-tune': 'full', 'vision-tune': 'full',
        'query-tune': 'selected_rows', 'query-lr': 1e-4, 'protocol-lr': 2e-5,
        'lr-qwen-start': 2e-6, 'lr-qwen-peak': 2e-6, 'state-proj-lr': 8e-5,
        'wm-predictor-lr': 3e-4, 'value-head-lr': 1e-4, 'outcome-head-lr': 1e-4,
        'lambda-outcome': int(arm == 'treatment'), 'max-length': args.max_length,
        'checkpoint-interval-steps': 10, 'checkpoint-keep-last': 2,
        'checkpoint-interval-minutes': 0, 'step-timing-interval': 1,
        'wandb-run-name': f'{args.run_root.name}_{arm}_{phase}',
    }
    result = [str(args.python), '-m', 'torch.distributed.run', '--nproc_per_node=8',
              f'--master_port={port}', '-m', 'nimloth.training.sft.stage3',
              '--config', str(args.worktree / 'configs/training/sft2/action_outcome_k64_h1_t4.yaml')]
    for key, value in values.items():
        result.extend(['--' + key, str(value)])
    result.extend(['--outcome-head', '--require-prebuilt-cache', '--step-timing'])
    if phase == 'formal':
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


def resources(args):
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
    if free < args.min_free_gib:
        raise RuntimeError(f'insufficient disk: {free:.1f} GiB free, require {args.min_free_gib:.1f}')
    return {'gpu_query': result, 'free_gib': free}


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
        for key in ('total_loss', 'wm_mse', 'dino_grid_mse', 'value_mc_mse', 'lm_ce', 'outcome_bce'):
            if row.get(key) and not math.isfinite(float(row[key])):
                raise RuntimeError(f'non-finite {key} in training log')
    required = ['training_state.pt' , 'state_proj.pt', 'outcome_head.pt', 'wm_predictor/predictor.pt',
                'value_head/value_head.pt', *[f'history_cache_rank_{rank:03d}.pt' for rank in range(8)]]
    for relative in required:
        if not (checkpoint / relative).is_file():
            raise RuntimeError(f'incomplete checkpoint: {checkpoint / relative}')
    return str(checkpoint)


def execute(args):
    if args.run_root.exists():
        raise FileExistsError(args.run_root)
    status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=args.worktree, text=True)
    if status.strip():
        raise RuntimeError('remote worktree must be clean and committed')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.worktree, text=True).strip()
    if commit != args.commit:
        raise RuntimeError('source commit mismatch')
    for path in (args.python, args.model, args.train, args.val, args.preprocess, args.dino):
        if not path.exists():
            raise FileNotFoundError(path)
    resources(args)
    args.run_root.mkdir()
    controller = args.run_root / 'controller'
    controller.mkdir()
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', TOKENIZERS_PARALLELISM='false')
    environment['PYTHONPATH'] = str(args.worktree / 'src')
    record = {'commit': commit, 'status': 'running', 'phases': [], 'hard_seconds_per_arm': ARM_SECONDS,
              'retention': 'no controller deletion; trainer rolling steps keep_last=2',
              'dataset_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in [('train', args.train), ('val', args.val)]}}
    deadlines = {}
    try:
        for arm, phase in [('control','canary'), ('control','resume'), ('treatment','canary'), ('treatment','resume'), ('control','formal'), ('treatment','formal')]:
            deadlines.setdefault(arm, time.monotonic() + ARM_SECONDS)
            remaining = deadlines[arm] - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f'{arm} total deadline expired')
            snapshot = resources(args)
            argv = command(args, arm, phase, free_port())
            entry = {'arm':arm, 'phase':phase, 'argv':argv, 'resources':snapshot,
                     'environment': {key:environment[key] for key in ('CUDA_VISIBLE_DEVICES','TOKENIZERS_PARALLELISM','PYTHONPATH')},
                     'started_at':time.time(), 'status':'running'}
            record['phases'].append(entry)
            (controller / 'progress.json').write_text(json.dumps(record, indent=2))
            elapsed = run_process(argv, cwd=args.worktree, environment=environment,
                                  log_path=controller / f'{arm}_{phase}.log',
                                  timeout=min(remaining, 900) if phase != 'formal' else remaining)
            entry.update(status='complete', elapsed_seconds=elapsed, checkpoint=verify_phase(args, arm, phase))
            (controller / 'progress.json').write_text(json.dumps(record, indent=2))
        record['status'] = 'complete'
        (controller / 'COMPLETE').write_text(json.dumps(record, indent=2))
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        (controller / 'FAILED').write_text(json.dumps(record, indent=2))
        raise
    finally:
        (controller / 'progress.json').write_text(json.dumps(record, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('python','worktree','model','train','val','preprocess','dino','run-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--min-free-gib', type=float, required=True,
                        help='Reviewed space budget for remaining phases, including all retained checkpoints.')
    parser.add_argument('--max-length', type=int, default=12000)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args(argv)
    if args.min_free_gib <= 0 or args.max_length < 1:
        parser.error('disk budget and max length must be positive')
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
                for index, (arm, phase) in enumerate([('control','canary'),('control','resume'),('treatment','canary'),('treatment','resume'),('control','formal'),('treatment','formal')])],
                'disk_warning':'Four canary stop checkpoints plus independent epoch/best/final and rolling checkpoints must fit; this controller deletes none.'}
        if args.preflight:
            plan['resources'] = resources(args)
        print(json.dumps(plan, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
