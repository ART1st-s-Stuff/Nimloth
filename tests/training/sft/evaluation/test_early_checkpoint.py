import json

import pytest

from nimloth.training.sft.evaluation.early_checkpoint import load_early_checkpoint


def checkpoint(tmp_path, **config):
    (tmp_path / 'config.json').write_text(json.dumps(config))
    (tmp_path / 'model.safetensors').write_bytes(b'unit fixture only')
    return tmp_path


def test_format_no_k(tmp_path):
    path = checkpoint(tmp_path, nimloth_training_stage='format')
    assert load_early_checkpoint(path, 'stage1').query_count is None
    with pytest.raises(ValueError, match='stage mismatch'):
        load_early_checkpoint(path, 'stage2')
    with pytest.raises(ValueError, match='VAGEN'):
        load_early_checkpoint(path, 'vagen')


@pytest.mark.parametrize('mode', ['inject', 'generate'])
def test_query_requires_artifacts(tmp_path, mode):
    path = checkpoint(tmp_path, nimloth_training_stage='query', nimloth_latent_token_count=1, nimloth_latent_query_mode=mode)
    with pytest.raises(ValueError, match='slot_projector'):
        load_early_checkpoint(path, 'stage2')
    (path / 'slot_projector.pt').write_bytes(b'unit fixture only')
    from dataclasses import asdict

    from nimloth.backbone.dino_grid import DINOV2_LARGE_IDENTITY
    metadata = dict(training_stage='query', objective=dict(grid_size=1, projector_hidden_dim=4),
                    grid_tokens=1, ordering='row_major', shared_slot_projector=True,
                    dino_identity=asdict(DINOV2_LARGE_IDENTITY), projector_hidden_dim=4,
                    qwen_hidden_dim=4, state_dim=1024, query_token_ids=[7])
    (path / 'grid_state_config.json').write_text(json.dumps(metadata))
    (path / 'tokenizer.json').write_text(json.dumps({'added_tokens': [{'content': '<|latent_state|>', 'id': 7}]}))
    assert load_early_checkpoint(path, 'stage2').query_mode == mode
