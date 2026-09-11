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


def validate_prepared_splits(train_path: Path, val_path: Path, format_path: Path) -> None:
    from .preparation import semantic_source_identity, validated_source_identity

    def records_by_id(path: Path) -> tuple[dict[str, dict], dict[tuple, dict]]:
        by_id: dict[str, dict] = {}
        by_source: dict[tuple, dict] = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            record_id = str(record['id'])
            if record_id in by_id:
                raise ValueError(f'Duplicate record ID in {path}: {record_id}')
            source_key = semantic_source_identity(validated_source_identity(record))
            if source_key in by_source:
                raise ValueError(
                    f'Duplicate semantic source identity in {path}: {record_id}'
                )
            by_id[record_id] = record
            by_source[source_key] = record
        return by_id, by_source

    train_records, train_sources = records_by_id(train_path)
    val_records, val_sources = records_by_id(val_path)
    format_eval_records, format_eval_sources = records_by_id(format_path)
    train_ids = set(train_records)
    val_ids = set(val_records)
    format_eval_ids = set(format_eval_records)
    if train_ids & val_ids:
        raise ValueError('Training and validation records overlap')
    if train_ids & format_eval_ids:
        raise ValueError('Training and format-eval records overlap')
    if not val_ids <= format_eval_ids:
        raise ValueError('Successful validation records are absent from format-eval')
    if any(
        val_records[record_id] != format_eval_records[record_id]
        for record_id in val_ids
    ):
        raise ValueError('Validation and format-eval record contents disagree')
    if set(train_sources) & set(val_sources):
        raise ValueError('Training and validation semantic sources overlap')
    if set(train_sources) & set(format_eval_sources):
        raise ValueError('Training and format-eval semantic sources overlap')
    train_seeds = {source_key[1] for source_key in train_sources}
    val_seeds = {source_key[1] for source_key in val_sources}
    format_eval_seeds = {source_key[1] for source_key in format_eval_sources}
    if train_seeds & val_seeds:
        raise ValueError('Training and validation source seeds overlap')
    if train_seeds & format_eval_seeds:
        raise ValueError('Training and format-eval source seeds overlap')
    if not val_seeds <= format_eval_seeds:
        raise ValueError('Successful validation seeds are absent from format-eval')
    if not set(val_sources) <= set(format_eval_sources):
        raise ValueError(
            'Successful validation semantic sources are absent from format-eval'
        )
    if any(
        val_sources[source_key] != format_eval_sources[source_key]
        for source_key in val_sources
    ):
        raise ValueError(
            'Validation and format-eval semantic source contents disagree'
        )

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-model', required=True, type=Path)
    parser.add_argument('--train-jsonl', required=True, type=Path)
    parser.add_argument('--val-jsonl', required=True, type=Path)
    parser.add_argument('--format-eval-jsonl', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--nproc-per-node', type=int, default=8)
    parser.add_argument(
        '--success-only',
        action='store_true',
        help='require boolean success and train/validate only on successful records',
    )
    parser.add_argument('--resume', action='store_true')
    args, overrides = parser.parse_known_args(argv)
    if args.nproc_per_node < 1:
        parser.error('--nproc-per-node must be positive')
    if not args.success_only:
        parser.error('standard Stage 1 workflow requires --success-only')
    # Paths are owned by this workflow; model/data replacement on resume is forbidden.
    reserved = {'--model', '--train-jsonl', '--val-jsonl', '--format-eval-jsonl',
                '--output-dir', '--config',
                '--cache-dir', '--cache-only', '--rebuild-cache', '--require-prebuilt-cache', '--no-cache'}
    if any(flag.split('=')[0] in reserved for flag in overrides):
        parser.error('Training overrides cannot replace workflow-owned paths/cache options')
    root = args.output_dir.resolve()
    contract = {name: str(getattr(args, name).resolve()) for name in
                ('source_model', 'train_jsonl', 'val_jsonl', 'format_eval_jsonl',
                 'config')}
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
    contract.update(
        overrides=identity_overrides,
        nproc_per_node=args.nproc_per_node,
        success_only=True,
    )
    contract["input_sha256"] = {name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
                                for name in ("train_jsonl", "val_jsonl",
                                             "format_eval_jsonl", "config")}
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
        train = prepare_records(
            args.train_jsonl,
            root / 'data/train.jsonl',
            success_only=True,
            require_all_actions=True,
        )
        val = prepare_records(
            args.val_jsonl,
            root / 'data/val.jsonl',
            success_only=True,
            require_all_actions=True,
        )
        format_eval = prepare_records(
            args.format_eval_jsonl,
            root / 'data/format_eval.jsonl',
        )
        validate_prepared_splits(
            root / 'data/train.jsonl',
            root / 'data/val.jsonl',
            root / 'data/format_eval.jsonl',
        )
        initialize_model(args.source_model, root / 'base')
        contract["base_fingerprint"] = policy_artifact_fingerprint(root / "base")
        (root / 'preparation.json').write_text(
            json.dumps(
                {'train': train, 'val': val, 'format_eval': format_eval}, indent=2
            )
        )
    common = ['--config', str(args.config.resolve()), '--model', str(root / 'base'),
              '--train-jsonl', str(root / 'data/train.jsonl'), '--val-jsonl', str(root / 'data/val.jsonl'),
              '--format-eval-jsonl', str(root / 'data/format_eval.jsonl'),
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
