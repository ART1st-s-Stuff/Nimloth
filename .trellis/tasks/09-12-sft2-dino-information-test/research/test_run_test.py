import hashlib
import json
import sys
from pathlib import Path
import time

import pytest
from run_test import validate_contract, validate_audit, checkpoints_to_evaluate, run_phase


def contract(root):
    inputs = ['--model', str(root / 'base'), '--train-jsonl', str(root / 'data/train.jsonl'),
              '--val-jsonl', str(root / 'data/val.jsonl'), '--dino-cache-root', str(root / 'dino_cache')]
    return {'root': str(root), 'train_argv': inputs + [
        '--nproc-per-node=8', '-m', 'nimloth.training.sft.stage2',
        '--epochs', '2', '--grid-size', '8', '--distributed-strategy', 'fsdp',
        '--save-initial-checkpoint', '--output-dir', str(root / 'train')],
        'eval_argv_template': inputs + ['--grid-size', '8', '--nproc-per-node=8', '{checkpoint}', '{output}'],
        'expected_records': {'train': 3, 'val': 2}}


def test_budget_and_fresh_training_guards(tmp_path):
    value = contract(tmp_path)
    assert validate_contract(value) == (21600, 3600)
    with pytest.raises(ValueError, match='budget'):
        validate_contract({**value, 'total_seconds': 21601})
    with pytest.raises(ValueError, match='two epochs'):
        validate_contract({**value, 'train_argv': value['train_argv'] + ['--until-converged']})
    with pytest.raises(ValueError, match='fresh test'):
        validate_contract({**value, 'train_argv': value['train_argv'] + ['--resume']})


def test_audit_detects_changed_inputs(tmp_path):
    value = contract(tmp_path)
    (tmp_path / 'data').mkdir()
    splits = {}
    for split, count in value['expected_records'].items():
        path = tmp_path / 'data' / (split + '.jsonl')
        path.write_text('{}\n' * count)
        splits[split] = dict(checked=count, records=count, jsonl=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), max_length=100,
            successful_lm_answers=1)
    audit = dict(status='passed', grid_size=8, query_count=64,
                 model=str(tmp_path / 'base'), max_length=20000, splits=splits)
    (tmp_path / 'input_audit.json').write_text(json.dumps(audit))
    validate_audit(value)
    (tmp_path / 'data/val.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='changed or invalid val'):
        validate_audit(value)


def test_checkpoint_selection_keeps_initial_epochs_and_newest_step(tmp_path):
    for name, epoch, step in [('epoch_000', 0, 0), ('epoch_001', 1, 10),
                              ('resume_step_00000005', 1, 5), ('resume_step_00000015', 2, 15)]:
        path = tmp_path / name
        path.mkdir()
        (path / 'COMMITTED').write_text(json.dumps({'epoch': epoch, 'step': step}))
    (tmp_path / 'epoch_002').mkdir()  # partial write is never evaluated
    assert [p.name for p in checkpoints_to_evaluate(tmp_path)] == [
        'epoch_000', 'epoch_001', 'resume_step_00000015']


def test_failed_phase_is_not_retried(tmp_path):
    events = []
    with pytest.raises(RuntimeError, match='failed with code 3'):
        run_phase([sys.executable, '-c', 'print("failure evidence"); raise SystemExit(3)'],
                  phase='failure', checkout=tmp_path, env={}, logs=tmp_path,
                  deadline=time.monotonic() + 60, event=lambda *a, **k: events.append((a, k)))
    assert sum(a[0] == 'phase_started' for a, _k in events) == 1
    assert 'failure evidence' in (tmp_path / 'failure.log').read_text()
