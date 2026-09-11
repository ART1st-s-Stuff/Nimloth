"""Portable Stage 1 preparation, cache and distributed training command."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str]) -> int:
    """Forward interruption to this command's process group and reap its launcher."""
    child = subprocess.Popen(command, start_new_session=True)
    previous = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Workflow received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        return child.wait()
    except BaseException:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-model', required=True, type=Path)
    parser.add_argument('--train-jsonl', required=True, type=Path)
    parser.add_argument('--val-jsonl', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--nproc-per-node', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    args, overrides = parser.parse_known_args(argv)
    if args.nproc_per_node < 1:
        parser.error('--nproc-per-node must be positive')
    # Paths are owned by this workflow; model/data replacement on resume is forbidden.
    reserved = {'--model', '--train-jsonl', '--val-jsonl', '--output-dir', '--config',
                '--cache-dir', '--cache-only', '--rebuild-cache', '--require-prebuilt-cache', '--no-cache'}
    if any(flag.split('=')[0] in reserved for flag in overrides):
        parser.error('Training overrides cannot replace workflow-owned paths/cache options')
    root = args.output_dir.resolve()
    contract = {name: str(getattr(args, name).resolve()) for name in
                ('source_model', 'train_jsonl', 'val_jsonl', 'config')}
    # Absolute diagnostic step caps are stop controls, not objective identity.
    # Removing or raising one must allow continuation from the same saved state.
    identity_overrides = []
    index = 0
    while index < len(overrides):
        flag = overrides[index]
        if flag == '--max-optimizer-steps':
            if index + 1 >= len(overrides):
                parser.error('--max-optimizer-steps requires a value')
            index += 2
            continue
        if not flag.startswith('--max-optimizer-steps='):
            identity_overrides.append(flag)
        index += 1
    contract.update(overrides=identity_overrides, nproc_per_node=args.nproc_per_node)
    contract["input_sha256"] = {name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
                                for name in ("train_jsonl", "val_jsonl", "config")}
    from nimloth.rollout.fresh import policy_artifact_fingerprint
    contract['source_model_fingerprint'] = policy_artifact_fingerprint(args.source_model)
    manifest_path = root / 'workflow.json'
    if args.resume:
        contract["base_fingerprint"] = policy_artifact_fingerprint(root / "base")
        saved = json.loads(manifest_path.read_text())
        if saved != contract:
            raise ValueError('Resume workflow parameters differ from the prepared run')
    else:
        from .initialization import initialize_model
        from .preparation import prepare_records
        root.mkdir(parents=True, exist_ok=False)
        train = prepare_records(args.train_jsonl, root / 'data/train.jsonl')
        val = prepare_records(args.val_jsonl, root / 'data/val.jsonl')
        train_ids = {json.loads(line)['id'] for line in (root / 'data/train.jsonl').read_text().splitlines()}
        val_ids = {json.loads(line)['id'] for line in (root / 'data/val.jsonl').read_text().splitlines()}
        if train_ids & val_ids:
            raise ValueError('Training and validation records overlap')
        initialize_model(args.source_model, root / 'base')
        contract["base_fingerprint"] = policy_artifact_fingerprint(root / "base")
        (root / 'preparation.json').write_text(json.dumps({'train': train, 'val': val}, indent=2))
    common = ['--config', str(args.config.resolve()), '--model', str(root / 'base'),
              '--train-jsonl', str(root / 'data/train.jsonl'), '--val-jsonl', str(root / 'data/val.jsonl'),
              '--output-dir', str(root / 'train'), '--cache-dir', str(root / 'cache'), *overrides]
    module = 'nimloth.training.sft.stage1'
    if not args.resume:
        cache_command = [sys.executable, '-m', module, *common, '--cache-only']
        code = run_command(cache_command)
        if code:
            raise subprocess.CalledProcessError(code, cache_command)
        manifest_path.write_text(json.dumps(contract, indent=2) + '\n')
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
               f'--nproc-per-node={args.nproc_per_node}', '-m', module, *common,
               '--require-prebuilt-cache']
    if args.resume:
        command.append('--resume')
    # Inherit the caller's resource/environment selection; no server-specific defaults.
    return run_command(command)


if __name__ == '__main__':
    raise SystemExit(main())
