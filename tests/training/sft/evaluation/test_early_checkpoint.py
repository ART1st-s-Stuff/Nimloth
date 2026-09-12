import json

import pytest

from nimloth.training.sft.evaluation.early_checkpoint import load_early_checkpoint


def checkpoint(tmp_path, **config):
    (tmp_path / 'config.json').write_text(json.dumps(config))
    (tmp_path / 'model.safetensors').write_bytes(b'unit fixture only')
    return tmp_path


def test_format_no_k(tmp_path):
    path = checkpoint(
        tmp_path,
        nimloth_training_stage='format',
        nimloth_format_objective='format_answer_ce_v2',
        nimloth_action_token_loss_scope='action_number_tokens_v1',
        nimloth_action_token_loss_weight=8.0,
    )
    assert load_early_checkpoint(path, 'stage1').query_count is None
    with pytest.raises(ValueError, match='stage mismatch'):
        load_early_checkpoint(path, 'stage2')
    with pytest.raises(ValueError, match='VAGEN'):
        load_early_checkpoint(path, 'vagen')


@pytest.mark.parametrize(
    "override",
    [
        {"nimloth_format_objective": "old"},
        {"nimloth_action_token_loss_scope": "action_boundaries_and_numbers"},
    ],
)
def test_format_rejects_old_loss_identity(tmp_path, override):
    config = {
        "nimloth_training_stage": "format",
        "nimloth_format_objective": "format_answer_ce_v2",
        "nimloth_action_token_loss_scope": "action_number_tokens_v1",
        "nimloth_action_token_loss_weight": 8.0,
        **override,
    }
    path = checkpoint(tmp_path, **config)
    with pytest.raises(ValueError, match="mismatch"):
        load_early_checkpoint(path, "stage1")


@pytest.mark.parametrize('mode', ['inject', 'generate'])
def test_query_requires_artifacts(tmp_path, mode):
    path = checkpoint(tmp_path, nimloth_training_stage='query', nimloth_latent_token_count=1, nimloth_latent_query_mode=mode)
    with pytest.raises(ValueError, match='slot_projector'):
        load_early_checkpoint(path, 'stage2')
    (path / 'slot_projector.pt').write_bytes(b'unit fixture only')
    from dataclasses import asdict

    from nimloth.backbone.dino_grid import DINOV2_LARGE_IDENTITY
    metadata = {
        'training_stage': 'query',
        'objective': {'grid_size': 1, 'projector_hidden_dim': 4},
        'grid_tokens': 1,
        'ordering': 'row_major',
        'shared_slot_projector': True,
        'dino_identity': asdict(DINOV2_LARGE_IDENTITY),
        'projector_hidden_dim': 4,
        'qwen_hidden_dim': 4,
        'state_dim': 1024,
        'query_token_ids': [7],
    }
    (path / 'grid_state_config.json').write_text(json.dumps(metadata))
    (path / 'tokenizer.json').write_text(json.dumps({'added_tokens': [{'content': '<|latent_state|>', 'id': 7}]}))
    assert load_early_checkpoint(path, 'stage2').query_mode == mode


@pytest.mark.parametrize('weight', [1.0, 8.0, 16.0])
def test_declared_protocol_weights_accepted(tmp_path, weight):
    path = checkpoint(tmp_path, nimloth_training_stage='format',
        nimloth_format_objective='format_answer_ce_v2',
        nimloth_action_token_loss_scope='action_number_tokens_v1',
        nimloth_action_token_loss_weight=weight,
        nimloth_boundary_token_loss_scope='action_boundaries_and_eos_v1',
        nimloth_boundary_token_loss_weight=weight)
    assert load_early_checkpoint(path, 'stage1').stage == 'stage1'
