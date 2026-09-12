"""Resume the bounded test without resetting its clock or initialization."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import time

from run_test import (argument, checkpoints_to_evaluate, run_phase, utc_now,
                      validate_audit, validate_contract)


def remaining_budget(original, now):
    total, reserve = validate_contract(original)
    events = [json.loads(line) for line in
              (Path(original['root']) / 'test_controller/events.jsonl').read_text().splitlines()]
    starts = [e for e in events if e['event'] == 'test_started']
    if len(starts) != 1 or starts[0]['total_seconds'] != total:
        raise ValueError('missing or ambiguous original budget anchor')
    start = datetime.fromisoformat(starts[0]['time'].replace('Z', '+00:00')).timestamp()
    remaining = start + total - now
    if remaining <= 60:
        raise TimeoutError('original six-hour budget exhausted')
    return remaining, reserve


def resume_argv(original, checkpoint):
    validate_contract(original)
    return [v for v in original['train_argv'] if v != '--save-initial-checkpoint'] + [
        '--resume']


def validate_marker(marker, state):
    if (marker.get('step') != state['step']
            or marker.get('schema') != state.get('resume_schema')
            or not marker.get('schema')):
        raise ValueError('checkpoint marker/state mismatch')


def validate_checkpoint(root, checkpoint):
    import torch
    from nimloth.training.sft.stage1.checkpoint import validate_resume_state, find_latest_resume_dir
    if checkpoint.resolve().parent != (root / 'train').resolve():
        raise ValueError('resume checkpoint outside original run')
    if find_latest_resume_dir(root / 'train').resolve() != checkpoint.resolve():
        raise ValueError('trainer resume selection differs from requested checkpoint')
    marker = json.loads((checkpoint / 'COMMITTED').read_text())
    state = torch.load(checkpoint / 'training_state.pt', map_location='cpu', weights_only=False)
    reference = torch.load(root / 'train/epoch_001/training_state.pt',
                           map_location='cpu', weights_only=False)
    validate_marker(marker, state)
    if not 0 < state['epoch'] <= 2 or state['step'] < reference['step']:
        raise ValueError('resume cursor outside approved continuation')
    if not state.get('optimizer') or not state.get('scheduler'):
        raise ValueError('optimizer/scheduler missing')
    for rank in range(8):
        validate_resume_state(state, expected_identity=reference['identity'], rank=rank, world=8)
    for filename in ('slot_projector.pt', 'adapter_model.safetensors', 'grid_state_config.json'):
        if not (checkpoint / filename).is_file():
            raise ValueError(f'missing checkpoint component: {filename}')
    checkpoints_to_evaluate(root / 'train')  # Requires original step-zero snapshot.
    return marker


def evaluation_complete(output, checkpoint, root):
    if not output.exists():
        return False
    if not (output / 'COMPLETED').is_file():
        raise ValueError(f'preserve incomplete evaluation: {output}')
    report = json.loads((output / 'metrics.json').read_text())
    provenance = report['provenance']
    expected = {'checkpoint': str(checkpoint.resolve()), 'model': str((root / 'base').resolve()),
                'world_size': 8, 'grid_size': 8,
                'checkpoint_commit_marker': json.loads((checkpoint / 'COMMITTED').read_text())}
    for key, file in [('train_jsonl_sha256', root / 'data/train.jsonl'),
                      ('val_jsonl_sha256', root / 'data/val.jsonl'),
                      ('projector_sha256', checkpoint / 'slot_projector.pt'),
                      ('adapter_sha256', checkpoint / 'adapter_model.safetensors')]:
        expected[key] = hashlib.sha256(file.read_bytes()).hexdigest()
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError('completed evaluation provenance mismatch')
    if not report.get('metrics'):
        raise ValueError('completed evaluation has no metrics')
    return True


def main():
    from run_query_gate import validate_checkout, wait_selected_gpus_idle
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    continuation = json.loads(args.contract.read_text())
    original = json.loads(Path(continuation['original_contract']).read_text())
    root, checkout = Path(original['root']), Path(original['checkout'])
    checkpoint = Path(continuation['resume_checkpoint'])
    validate_checkout(checkout, continuation['commit'])
    optimized = '03962702'
    import subprocess
    subprocess.run(['git', 'merge-base', '--is-ancestor', optimized, continuation['commit']],
                   cwd=checkout, check=True)
    validate_audit(original)
    marker = validate_checkpoint(root, checkpoint)
    remaining, reserve = remaining_budget(original, time.time())
    logs = root / continuation['controller_name']
    if logs.parent != root or not logs.name.startswith('resume_controller_') or logs.exists():
        raise ValueError('continuation requires unique controller directory')
    if args.check_only:
        print(json.dumps({'preflight': 'passed', 'checkpoint': marker, 'remaining_seconds': remaining}))
        return
    logs.mkdir()
    def event(kind, **fields):
        with (logs / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps({'time': utc_now(), 'event': kind, **fields}) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    env = os.environ.copy()
    env.update(original.get('env', {}))
    env.update(PYTHONPATH=str(checkout / 'src'), CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    deadline = time.monotonic() + remaining
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    event('resume_started', commit=continuation['commit'], checkpoint=str(checkpoint),
          marker=marker, remaining_seconds=remaining, pid=os.getpid())
    try:
        wait_selected_gpus_idle(0, 8)
        train_end = deadline - reserve
        status = 'evaluation_reserve_reached'
        if time.monotonic() < train_end - 300:
            status = run_phase(resume_argv(original, checkpoint), phase='train', checkout=checkout,
                               env=env, logs=logs, deadline=train_end, event=event,
                               training_run=root / 'train', soft_deadline=train_end - 300)
        completed = []
        for checkpoint in checkpoints_to_evaluate(root / 'train'):
            output = root / 'evaluation' / checkpoint.name
            if not evaluation_complete(output, checkpoint, root):
                if time.monotonic() >= deadline - 60:
                    raise TimeoutError('original budget exhausted before evaluations')
                wait_selected_gpus_idle(min(30, deadline - time.monotonic() - 40), 8)
                argv = [value.replace('{checkpoint}', str(checkpoint)).replace('{output}', str(output))
                        for value in original['eval_argv_template']]
                run_phase(argv, phase='eval_' + checkpoint.name, checkout=checkout, env=env,
                          logs=logs, deadline=deadline, event=event)
                if not evaluation_complete(output, checkpoint, root):
                    raise RuntimeError('missing completed evaluation')
            completed.append(str(output))
        result = {'training_status': status, 'evaluations': completed}
        (logs / 'COMPLETED.json').write_text(json.dumps(result, indent=2) + '\n')
        event('test_completed', **result)
    except BaseException as error:
        event('test_stopped', error=repr(error))
        (logs / 'STOPPED.json').write_text(json.dumps({'error': repr(error)}) + '\n')
        raise


if __name__ == '__main__':
    main()
