"""Stage-specific full-HF acceptance before allocating inference resources."""
from __future__ import annotations

import json
from pathlib import Path

from nimloth.agent.evaluation_protocol import EarlyProtocol


def load_early_checkpoint(path: Path, stage: str) -> EarlyProtocol:
    config = json.loads((path / 'config.json').read_text())
    if (path / 'adapter_config.json').exists():
        raise ValueError('evaluation requires exported full HF; use stage1.checkpoint_export')
    weights = [name for name in ('model.safetensors', 'pytorch_model.bin') if (path / name).is_file()]
    for index_name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
        if (path / index_name).is_file():
            index = json.loads((path / index_name).read_text())
            shards = set(index.get('weight_map', {}).values())
            if not shards or any(Path(name).name != name or not (path / name).is_file() for name in shards):
                raise ValueError('HF shard index is empty or references missing/unsafe shards')
            weights.extend(shards)
    if not weights:
        raise ValueError('checkpoint has no full HF weights')
    saved = config.get('nimloth_training_stage')
    count = config.get('nimloth_latent_token_count')
    mode = config.get('nimloth_latent_query_mode')
    if stage == 'vagen':
        if saved is not None or count is not None or mode is not None:
            raise ValueError('VAGEN evaluation cannot consume a Nimloth checkpoint')
        return EarlyProtocol(stage)
    if saved != {'stage1': 'format', 'stage2': 'query'}[stage]:
        raise ValueError(f'checkpoint stage mismatch: requested {stage}, saved {saved}')
    if stage == 'stage1':
        if count is not None or mode is not None:
            raise ValueError('format checkpoint must not enable query slots')
        from nimloth.training.sft.stage1.data import FORMAT_OBJECTIVE
        from nimloth.training.sft.stage1.loss import ACTION_TOKEN_LOSS_SCOPE, BOUNDARY_TOKEN_LOSS_SCOPE, validate_action_weight

        expected = {
            'nimloth_format_objective': FORMAT_OBJECTIVE,
            'nimloth_action_token_loss_scope': ACTION_TOKEN_LOSS_SCOPE,
        }
        for name, value in expected.items():
            if config.get(name) != value:
                raise ValueError(
                    f'format checkpoint {name} mismatch: expected {value!r}, '
                    f'found {config.get(name)!r}'
                )
        if 'nimloth_action_token_loss_weight' not in config:
            raise ValueError('format checkpoint lacks declared action loss weight')
        validate_action_weight(config['nimloth_action_token_loss_weight'])
        boundary_weight = validate_action_weight(config.get('nimloth_boundary_token_loss_weight', 1.0))
        if config.get('nimloth_boundary_token_loss_scope', BOUNDARY_TOKEN_LOSS_SCOPE if boundary_weight == 1 else None) != BOUNDARY_TOKEN_LOSS_SCOPE:
            raise ValueError('format checkpoint boundary loss scope mismatch')
        return EarlyProtocol(stage)
    if type(count) is not int or count < 1 or mode not in {'inject', 'generate'}:
        raise ValueError('query checkpoint requires positive K and explicit inject/generate mode')
    for name in ('slot_projector.pt', 'grid_state_config.json'):
        if not (path / name).is_file():
            raise ValueError(f'query checkpoint is missing {name}')
    # Metadata identity is retained, although direct evaluation never runs this projector.
    from dataclasses import asdict

    from nimloth.backbone.dino_grid import DINOV2_LARGE_IDENTITY
    from nimloth.training.sft.stage2.config import QueryAlignmentConfig
    metadata = json.loads((path / 'grid_state_config.json').read_text())
    objective = QueryAlignmentConfig(**metadata.get('objective', {}))
    if (metadata.get('training_stage') != 'query' or metadata.get('grid_tokens') != count
            or objective.grid_tokens != count or metadata.get('ordering') != 'row_major'
            or metadata.get('shared_slot_projector') is not True
            or metadata.get('dino_identity') != asdict(DINOV2_LARGE_IDENTITY)
            or metadata.get('projector_hidden_dim') != objective.projector_hidden_dim):
        raise ValueError('query grid metadata does not match checkpoint contract')
    for name in ('qwen_hidden_dim', 'state_dim'):
        if type(metadata.get(name)) is not int or metadata[name] < 1:
            raise ValueError(f'invalid query projector dimension: {name}')
    tokenizer = json.loads((path / 'tokenizer.json').read_text())
    token_map = {row['content']: row['id'] for row in tokenizer.get('added_tokens', [])}
    protocol = EarlyProtocol(stage, count, mode)
    query_ids = [token_map.get(token) for token in protocol.query_tokens]
    if any(type(value) is not int for value in query_ids) or metadata.get('query_token_ids') != query_ids:
        raise ValueError('query token IDs do not match tokenizer metadata')
    return EarlyProtocol(stage, count, mode)
