import json
from datetime import datetime, timezone

import pytest
from resume_test import remaining_budget, resume_argv, evaluation_complete


def contract(tmp_path):
    shared = ['--model', str(tmp_path / 'base'), '--train-jsonl', str(tmp_path / 'data/train.jsonl'),
              '--val-jsonl', str(tmp_path / 'data/val.jsonl'), '--dino-cache-root',
              str(tmp_path / 'dino_cache'), '--grid-size', '8', '--nproc-per-node=8']
    original = {'root': str(tmp_path), 'train_argv': shared + [
        '-m', 'nimloth.training.sft.stage2', '--output-dir', str(tmp_path / 'train'),
        '--epochs', '2', '--distributed-strategy', 'fsdp', '--save-initial-checkpoint'],
        'eval_argv_template': shared + ['--checkpoint', '{checkpoint}', '--output-dir', '{output}']}
    return original


def test_resume_preserves_training_arguments(tmp_path):
    original = contract(tmp_path)
    result = resume_argv(original, tmp_path / 'train/resume_step_30')
    assert result[:-1] == [v for v in original['train_argv'] if v != '--save-initial-checkpoint']
    assert result[-1:] == ['--resume']


def test_deadline_is_original_start_not_restart(tmp_path):
    original = contract(tmp_path)
    logs = tmp_path / 'test_controller'
    logs.mkdir()
    (logs / 'events.jsonl').write_text(json.dumps({'event': 'test_started',
        'time': '2026-09-12T08:35:54+00:00', 'total_seconds': 21600}) + '\n')
    start = datetime(2026, 9, 12, 8, 35, 54, tzinfo=timezone.utc).timestamp()
    assert remaining_budget(original, start + 7200) == (14400, 3600)
    with pytest.raises(TimeoutError):
        remaining_budget(original, start + 21600)


def test_partial_evaluation_never_overwritten(tmp_path):
    output = tmp_path / 'evaluation'
    assert not evaluation_complete(output, tmp_path / 'checkpoint', tmp_path)
    output.mkdir()
    with pytest.raises(ValueError, match='incomplete'):
        evaluation_complete(output, tmp_path / 'checkpoint', tmp_path)


def test_resume_marker_has_schema_and_step_not_epoch():
    from resume_test import validate_marker
    marker = {'schema': 'nimloth_early_stage_resume_v1', 'step': 32}
    state = {'resume_schema': marker['schema'], 'step': 32, 'epoch': 2}
    validate_marker(marker, state)
    with pytest.raises(ValueError):
        validate_marker({**marker, 'step': 31}, state)
