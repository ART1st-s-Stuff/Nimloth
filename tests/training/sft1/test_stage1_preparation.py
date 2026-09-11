import json

from nimloth.agent.action_prompt import format_action_prompt
from nimloth.training.sft.stage1.preparation import prepare_records


def test_prompt_protocol_is_idempotent_and_semantic():
    source = 'The answer must be exactly one of these lowercase action names: moveahead, moveback, moveright, moveleft, rotateright, rotateleft, lookup, lookdown. <answer>moveahead</answer>'
    actual = format_action_prompt(source)
    assert '<|action_start|><|action_(0)|><|action_end|>' in actual
    assert 'lowercase action names' not in actual
    assert '<|action_(0)|> = moveahead (move forward)' in actual
    assert format_action_prompt(actual) == actual


def test_preparation_preserves_targets_and_images(tmp_path):
    target = '<think>real recorded thought</think><|latent_state_0|><|action_start|><|action_(0)|><|action_end|>'
    row = {'id': 'r', 'messages': [{'role': 'user', 'content': [
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
              '--val-jsonl', str(val), '--output-dir', str(root), '--config', str(config), '--resume'])


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
    for name, path in [('train', train), ('val', val)]:
        path.write_text(json.dumps({'id': name, 'messages': [{'role': 'user', 'content': '<answer>moveahead</answer>'}]}) + '\n')
    config = tmp_path / 'config'
    config.write_text('train: {}')
    root = tmp_path / 'run'
    args = ['--source-model', str(model), '--train-jsonl', str(train), '--val-jsonl', str(val),
            '--output-dir', str(root), '--config', str(config)]
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
