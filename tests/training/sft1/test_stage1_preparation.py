import json

import pytest

from nimloth.agent.action_prompt import format_action_prompt
from nimloth.training.sft.stage1.preparation import prepare_records
from nimloth.training.sft.stage1.workflow import validate_prepared_splits


def source_identity(seed, *, split="train", eval_set="base", source_index=None):
    return {
        "source_index": seed if source_index is None else source_index,
        "source_key": f"{eval_set}:{seed}",
        "eval_set": eval_set,
        "seed": seed,
        "batch": 1,
        "split": split,
    }


def test_prompt_protocol_is_idempotent_and_semantic():
    source = 'The answer must be exactly one of these lowercase action names: moveahead, moveback, moveright, moveleft, rotateright, rotateleft, lookup, lookdown. <answer>moveahead</answer>'
    actual = format_action_prompt(source)
    assert '<|action_start|><|action_(0)|><|action_end|>' in actual
    assert 'lowercase action names' not in actual
    assert '<|action_(0)|> = moveahead (move forward)' in actual
    assert format_action_prompt(actual) == actual


def test_preparation_preserves_targets_and_images(tmp_path):
    target = '<think>real recorded thought</think><|latent_state_0|><|action_start|><|action_(0)|><|action_end|>'
    row = {'id': 'r', 'source_identity': source_identity(1), 'messages': [{'role': 'user', 'content': [
        {'type': 'text', 'text': '<answer>moveahead</answer>'},
        {'type': 'image', 'image': 'original.png'}]}, {'role': 'assistant', 'content': target}]}
    source, out = tmp_path / 'in.jsonl', tmp_path / 'out.jsonl'
    source.write_text(json.dumps(row) + '\n')
    manifest = prepare_records(source, out)
    saved = json.loads(out.read_text())
    assert saved['messages'][1] == row['messages'][1]
    assert saved['messages'][0]['content'][1] == row['messages'][0]['content'][1]
    assert json.loads(source.read_text()) == row
    assert manifest['records'] == 1


def test_success_only_preparation_filters_and_audits_records(tmp_path):
    rows = [
        {
            'id': 'kept',
            'source_identity': source_identity(1),
            'success': True,
            'action_indices': list(range(8)),
            'messages': [{'role': 'user', 'content': '<answer>moveahead</answer>'}],
        },
        {
            'id': 'excluded',
            'source_identity': source_identity(2),
            'success': False,
            'action_indices': [0],
            'messages': [{'role': 'user', 'content': '<answer>moveahead</answer>'}],
        },
    ]
    source, output = tmp_path / 'source.jsonl', tmp_path / 'selected.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    manifest = prepare_records(source, output, success_only=True)
    assert [json.loads(line)['id'] for line in output.read_text().splitlines()] == [
        'kept'
    ]
    assert manifest['selection'] == 'success_is_true'
    assert manifest['source_records'] == 2
    assert manifest['records'] == 1
    assert manifest['trajectory_records'] == 1
    assert manifest['assistant_turns'] == 0
    assert manifest['excluded_records'] == 1
    assert manifest['action_counts'] == {str(index): 1 for index in range(8)}


def test_success_only_preparation_rejects_unknown_success(tmp_path):
    for row, match in [
        ({'id': 'missing', 'source_identity': source_identity(3), 'messages': []}, 'boolean success'),
        ({'id': 'string', 'source_identity': source_identity(4), 'success': 'true', 'messages': []}, 'boolean success'),
    ]:
        source = tmp_path / f"{row['id']}.jsonl"
        output = tmp_path / f"{row['id']}-out.jsonl"
        source.write_text(json.dumps(row) + '\n')
        with pytest.raises((TypeError, ValueError), match=match):
            prepare_records(source, output, success_only=True)


def test_success_only_preparation_records_missing_action_classes(tmp_path):
    row = {
        'id': 'partial',
        'source_identity': source_identity(5),
        'success': True,
        'action_indices': [0],
        'messages': [],
    }
    source = tmp_path / 'partial.jsonl'
    output = tmp_path / 'partial-out.jsonl'
    source.write_text(json.dumps(row) + '\n')
    manifest = prepare_records(source, output, success_only=True)
    assert manifest['records'] == 1
    assert manifest['action_counts'] == {
        '0': 1,
        '1': 0,
        '2': 0,
        '3': 0,
        '4': 0,
        '5': 0,
        '6': 0,
        '7': 0,
    }


def test_prepared_split_relationship_requires_full_heldout_superset(tmp_path):
    def write(name, rows):
        path = tmp_path / name
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return path

    train = write('train.jsonl', [{'id': 'train', 'source_identity': source_identity(1), 'messages': []}])
    val_identity = source_identity(2, split='val')
    val = write('val.jsonl', [{'id': 'val', 'source_identity': val_identity, 'messages': []}])
    complete = write(
        'format.jsonl',
        [
            {'id': 'val', 'source_identity': val_identity, 'messages': []},
            {'id': 'failed-val', 'source_identity': source_identity(3, split='val'), 'messages': []},
        ],
    )
    validate_prepared_splits(train, val, complete)
    overlap = write('overlap.jsonl', [{'id': 'train', 'source_identity': source_identity(1), 'messages': []}])
    with pytest.raises(ValueError, match='Training and format-eval'):
        validate_prepared_splits(train, val, overlap)
    missing = write('missing.jsonl', [{'id': 'other', 'source_identity': source_identity(4, split='val'), 'messages': []}])
    with pytest.raises(ValueError, match='absent from format-eval'):
        validate_prepared_splits(train, val, missing)


def test_prepared_split_rejects_semantic_overlap_even_when_ids_differ(tmp_path):
    def write(name, row):
        path = tmp_path / name
        path.write_text(json.dumps(row) + '\n')
        return path

    train = write(
        'train.jsonl',
        {'id': 'train-id', 'source_identity': source_identity(9), 'messages': []},
    )
    heldout_identity = source_identity(9, split='val', source_index=999)
    heldout = write(
        'heldout.jsonl',
        {'id': 'different-id', 'source_identity': heldout_identity, 'messages': []},
    )
    with pytest.raises(ValueError, match='semantic sources overlap'):
        validate_prepared_splits(train, heldout, heldout)


def test_preparation_refuses_to_invent_source_identity(tmp_path):
    source = tmp_path / 'source.jsonl'
    output = tmp_path / 'output.jsonl'
    source.write_text(json.dumps({'id': 'unknown', 'messages': []}) + '\n')
    with pytest.raises(TypeError, match='has no source_identity'):
        prepare_records(source, output)


def test_stage1_prompt_matches_historical_selected_experiment():
    # Historical implementations are comparison evidence only, never runtime dependencies.
    import runpy
    from pathlib import Path

    from nimloth.training.sft.stage1.data import render_stage_text
    root = Path(__file__).resolve().parents[3]
    convert = runpy.run_path(str(root / 'experiments/training/sft1/vagen_step60_data.py'))['convert_source_prompt']
    rewrite = runpy.run_path(str(root / '.trellis/tasks/09-10-sft1-rollout2000/research/prepare_b_prompt_data.py'))['rewrite_text']
    for action in ('moveahead', 'moveback', 'rotateright', 'action'):
        text = f'Current observation <image>. Example: <answer>{action}</answer>. Write inside <answer> and stop after </answer>.'
        legacy = render_stage_text(rewrite(convert(text, latent_token_count=16)), None)
        assert format_action_prompt(text) == legacy


def test_pruning_rejects_unrelated_checkpoint(tmp_path):
    import pytest
    import torch

    from nimloth.training.sft.stage1.checkpoint import prune_intermediate_checkpoints
    def save(name, state):
        path = tmp_path / name
        path.mkdir()
        (path / 'COMMITTED').touch()
        torch.save(state, path / 'training_state.pt')
        return path
    identity = {'training_stage': 'format', 'model': 'owned'}
    save('epoch_001', {'epoch': 1, 'step': 20, 'training_stage': 'format', 'identity': identity})
    old = save('resume_step_00000010', {'step': 10, 'identity': identity})
    future = save('resume_step_00000030', {'step': 30, 'identity': identity})
    prune_intermediate_checkpoints(tmp_path, 1, 20)
    assert not old.exists() and future.exists()
    other = save('resume_step_00000015', {'step': 15, 'identity': {'model': 'other'}})
    with pytest.raises(ValueError, match='unrelated'):
        prune_intermediate_checkpoints(tmp_path, 1, 20)
    assert other.exists()


def test_workflow_refuses_resume_configuration_drift(tmp_path):
    import pytest

    from nimloth.training.sft.stage1.workflow import main
    root = tmp_path / 'run'
    root.mkdir()
    (root / 'workflow.json').write_text('{}')
    for model in (tmp_path / 'model', root / 'base'):
        model.mkdir()
        (model / 'config.json').write_text('{}')
    train, val, config = [tmp_path / name for name in ('train', 'val', 'config')]
    for path in (train, val, config):
        path.write_text('data')
    with pytest.raises(ValueError, match='differ'):
        main(['--source-model', str(tmp_path / 'model'), '--train-jsonl', str(train),
              '--val-jsonl', str(val), '--format-eval-jsonl', str(val),
              '--output-dir', str(root), '--config', str(config),
              '--success-only', '--resume'])


def test_workflow_resume_allows_cap_change_but_rejects_model_change(tmp_path, monkeypatch):
    import pytest

    from nimloth.training.sft.stage1 import initialization, workflow
    def initialize(source, output):
        output.mkdir()
        (output / 'config.json').write_text('{}')
    monkeypatch.setattr(initialization, 'initialize_model', initialize)
    commands = []
    monkeypatch.setattr(workflow, 'run_command', lambda argv: commands.append(argv) or 0)
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    train, val = tmp_path / 'train', tmp_path / 'val'
    for seed, (name, path) in enumerate([('train', train), ('val', val)], start=1):
        path.write_text(json.dumps({'id': name, 'success': True,
            'source_identity': source_identity(seed, split=name), 'split': name,
            'action_indices': list(range(8)),
            'messages': [{'role': 'user', 'content': '<answer>moveahead</answer>'}]}) + '\n')
    config = tmp_path / 'config'
    config.write_text('train: {}')
    root = tmp_path / 'run'
    args = ['--source-model', str(model), '--train-jsonl', str(train), '--val-jsonl', str(val),
            '--format-eval-jsonl', str(val),
            '--output-dir', str(root), '--config', str(config), '--success-only']
    assert workflow.main(args + ['--max-optimizer-steps', '20']) == 0
    assert workflow.main(args + ['--resume', '--max-optimizer-steps=40']) == 0
    assert workflow.main(args + ['--resume']) == 0
    assert len(commands) == 4  # one cache build, then three training launches
    (root / 'base/config.json').write_text('{"changed": true}')
    with pytest.raises(ValueError, match='differ'):
        workflow.main(args + ['--resume'])
    assert len(commands) == 4
    (root / 'base/config.json').write_text('{}')
    (model / 'config.json').write_text('{"changed": true}')
    with pytest.raises(ValueError, match='differ'):
        workflow.main(args + ['--resume'])
    assert len(commands) == 4


def test_workflow_interruption_terminates_only_owned_group(monkeypatch):
    import signal

    import pytest

    from nimloth.training.sft.stage1 import workflow
    class Child:
        pid = 991
        waits = 0
        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise KeyboardInterrupt()
            return -15
    child = Child()
    calls = []
    monkeypatch.setattr(workflow.subprocess, 'Popen', lambda argv, **kw: child)
    monkeypatch.setattr(workflow.os, 'killpg', lambda pid, sig: calls.append((pid, sig)))
    with pytest.raises(KeyboardInterrupt):
        workflow.run_command(['training'])
    assert calls == [(child.pid, signal.SIGTERM)]
    assert child.waits == 2
