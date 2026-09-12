import hashlib
import json
import sys
from pathlib import Path
import time

import pytest
from run_test import validate_contract, validate_audit, checkpoints_to_evaluate, run_phase


def contract(root):
    inputs_root = root / 'inputs'
    output_root = root / 'fresh'
    inputs = ['--model', str(inputs_root / 'base'), '--train-jsonl', str(inputs_root / 'data/train.jsonl'),
              '--val-jsonl', str(inputs_root / 'data/val.jsonl'), '--dino-cache-root', str(inputs_root / 'dino_cache')]
    return {'root': str(output_root), 'input_root': str(inputs_root),
        'train_argv': inputs + [
        '--nproc-per-node=8', '-m', 'nimloth.training.sft.stage2',
        '--until-converged', '--convergence-min-epochs', '2',
        '--convergence-patience-epochs', '2',
        '--convergence-min-relative-improvement', '.01',
        '--grid-size', '8', '--distributed-strategy', 'fsdp',
        '--save-initial-checkpoint', '--output-dir', str(output_root / 'train'),
        '--lr', '5e-5', '--projector-lr', '5e-5', '--query-token-lr', '5e-5',
        '--protocol-token-lr', '1e-5', '--embedding-master-dtype', 'float32',
        '--resume-save-steps', '10'],
        'eval_argv_template': inputs + ['--grid-size', '8', '--nproc-per-node=8', '{checkpoint}', '{output}'],
        'expected_records': {'train': 3, 'val': 2}}


def test_budget_and_fresh_training_guards(tmp_path):
    value = contract(tmp_path)
    assert validate_contract(value) == (21600, 3600)
    with pytest.raises(ValueError, match='budget'):
        validate_contract({**value, 'total_seconds': 21601})
    with pytest.raises(ValueError, match='convergence mode'):
        validate_contract({**value, 'train_argv': value['train_argv'] + ['--epochs', '2']})
    with pytest.raises(ValueError, match='forbids'):
        validate_contract({**value, 'train_argv': value['train_argv'] + ['--resume']})


def test_audit_detects_changed_inputs(tmp_path):
    value = contract(tmp_path)
    input_root = Path(value['input_root'])
    (input_root / 'data').mkdir(parents=True)
    splits = {}
    for split, count in value['expected_records'].items():
        path = input_root / 'data' / (split + '.jsonl')
        path.write_text('{}\n' * count)
        splits[split] = dict(checked=count, records=count, jsonl=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), max_length=100,
            successful_lm_answers=1)
    audit = dict(status='passed', grid_size=8, query_count=64, train_max_sample_index=0,
                 model=str(input_root / 'base'), max_length=20000, splits=splits)
    (input_root / 'input_audit.json').write_text(json.dumps(audit))
    validate_audit(value)
    (input_root / 'data/val.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='changed or invalid val'):
        validate_audit(value)


def test_contract_identity_can_validate_separate_existing_inputs(tmp_path):
    value = contract(tmp_path)
    input_root = Path(value['input_root'])
    (input_root / 'data').mkdir(parents=True)
    identity = {'train_max_sample_index': 1}
    for split, count in value['expected_records'].items():
        path = input_root / 'data' / f'{split}.jsonl'
        path.write_text('{}\n' * count)
        identity[split] = {
            'records': count,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    base = input_root / 'base'
    base.mkdir()
    for name, content in (('config.json', '{}'), ('model.safetensors', 'weights')):
        path = base / name
        path.write_text(content)
    identity['base_files'] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in base.iterdir()
    }
    dino = input_root / 'dino_cache'
    dino.mkdir()
    manifest = dino / 'manifest.json'
    manifest.write_text(json.dumps({'fingerprint': 'cache-id'}))
    identity['dino_cache_manifest_sha256'] = hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    identity['dino_cache_fingerprint'] = 'cache-id'
    value['input_identity'] = identity
    assert validate_audit(value) == identity
    (input_root / 'data/train.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='changed or invalid train identity'):
        validate_audit(value)


def test_checkpoint_selection_uses_only_initial_and_proven_converged_final(tmp_path):
    initial = tmp_path / 'epoch_000'
    initial.mkdir()
    (initial / 'COMMITTED').write_text(json.dumps({'epoch': 0, 'step': 0}))
    final = tmp_path / 'final'
    final.mkdir()
    (final / 'training_state.pt').write_text('state')
    (tmp_path / 'CONVERGED.json').write_text(json.dumps({'state': {'converged': True}}))
    assert [p.name for p in checkpoints_to_evaluate(tmp_path)] == ['epoch_000', 'final']
    (tmp_path / 'CONVERGED.json').unlink()
    with pytest.raises(ValueError, match='proven converged'):
        checkpoints_to_evaluate(tmp_path)


def test_failed_phase_is_not_retried(tmp_path):
    events = []
    with pytest.raises(RuntimeError, match='failed with code 3'):
        run_phase([sys.executable, '-c', 'print("failure evidence"); raise SystemExit(3)'],
                  phase='failure', checkout=tmp_path, env={}, logs=tmp_path,
                  deadline=time.monotonic() + 60, event=lambda *a, **k: events.append((a, k)))
    assert sum(a[0] == 'phase_started' for a, _k in events) == 1
    assert 'failure evidence' in (tmp_path / 'failure.log').read_text()
