import json
import shutil
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from experiments.training.sft.stage3.cfm_decoder_probe import (
    CLS_SCHEMA,
    PROVENANCE,
    PairedObservationDataset,
    decoder_family_from_payload,
    load_decoder,
    save_decoder,
    train,
    validate_pair,
)
from experiments.training.sft.stage3.frozen_wm_diagnostic import file_sha256, seal_cache
from nimloth.recon.cfm.flow import conditional_flow_matching_loss
from nimloth.recon.cfm.model import (
    CFMConfig,
    SpatialCLSCFMConfig,
    SpatialCLSConditionedFlowUNet,
    SpatialConditionedFlowUNet,
)
from nimloth.training.sft.stage3.diagnostics import FrozenWMTrajectoryWriter


def export(tmp_path, split, *, tokens=64):
    paths = []
    split_offset = 0 if split == 'train' else 100
    for step in range(5):
        path = tmp_path / f'{split}{step}.png'
        Image.new('RGB', (12, 10), (split_offset + step, 10, 20)).save(path)
        paths.append(str(path))
    jsonl = tmp_path / f'{split}.jsonl'
    jsonl.write_text(json.dumps({'id': split, 'image_paths': paths,
                                'action_indices': [0, 1, 2, 3]}) + '\n')
    identity = {key: key for key in PROVENANCE} | {'split': split,
        'split_sha256': file_sha256(jsonl), 'prediction_horizon': 4,
        'grid_tokens': tokens, 'state_dim': 1024, 'action_dim': 8,
        'trajectory_count': 1, 'window_count': 1}
    if tokens == 65:
        identity['state_layout'] = {
            'schema': 'nimloth_grid_state_layout_v2',
            'spatial_grid_size': 8,
            'global_tokens': 1,
            'global_role': 'dino_cls',
            'spatial_tokens': 64,
            'state_tokens': 65,
            'ordering': 'row_major_spatial_then_global',
        }
    writer = FrozenWMTrajectoryWriter(tmp_path / split, rank=0, identity=identity)
    states = torch.randn(5, tokens, 1024)
    batch = SimpleNamespace(observed_dino_target=states + 1, prediction_horizon=4,
        state_keys=tuple((split, i) for i in range(5)), trajectory_ids=(split,),
        state_offsets=(0, 5), window_offsets=(0, 1),
        action_sequences=torch.tensor([[0, 1, 2, 3]]), sample_weights=torch.ones(1))
    writer(batch, SimpleNamespace(online_states=states))
    writer.finalize()
    seal_cache(tmp_path / split, expected_ranks=1)
    return tmp_path / split, jsonl


def test_paired_cache_keys_features_and_split_guards(tmp_path):
    cache, jsonl = export(tmp_path, 'train')
    state = PairedObservationDataset(cache, jsonl, 'train', 'state')
    dino = PairedObservationDataset(cache, jsonl, 'train', 'dino')
    assert state.keys == dino.keys == [('train', i) for i in range(5)]
    torch.testing.assert_close(dino.conditions, state.conditions + 1)
    assert state.images.dtype == torch.uint8 and state.images.shape == (5, 3, 128, 128)
    assert not state.conditions.requires_grad
    with pytest.raises(ValueError, match='split identity'):
        PairedObservationDataset(cache, jsonl, 'eval')
    with pytest.raises(ValueError, match='overlap'):
        validate_pair(state, dino)
    shard = cache / 'rank_000_batch_000000.pt'
    with shard.open('ab') as stream:
        stream.write(b'bad')
    with pytest.raises(ValueError, match='shard identity'):
        PairedObservationDataset(cache, jsonl, 'train')


def test_spatial_cls_dataset_requires_exact_k65_cache(tmp_path):
    cache, jsonl = export(tmp_path, 'train', tokens=65)
    dataset = PairedObservationDataset(
        cache,
        jsonl,
        'train',
        'state',
        decoder_family='spatial_cls_grid_v1',
    )
    assert dataset.conditions.shape == (5, 65, 1024)
    assert dataset.identity['state_layout'] == {
        'schema': 'nimloth_grid_state_layout_v2',
        'spatial_grid_size': 8,
        'global_tokens': 1,
        'global_role': 'dino_cls',
        'spatial_tokens': 64,
        'state_tokens': 65,
        'ordering': 'row_major_spatial_then_global',
    }
    with pytest.raises(ValueError, match='requires exact K64'):
        PairedObservationDataset(cache, jsonl, 'train', 'state')


def test_spatial_cls_dataset_rejects_k65_cache_without_layout(tmp_path):
    cache, jsonl = export(tmp_path, 'train', tokens=65)
    manifest_path = cache / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['identity'].pop('state_layout')
    manifest_path.write_text(json.dumps(manifest))
    (cache / 'COMPLETE').write_text(file_sha256(manifest_path) + '\n')
    with pytest.raises(ValueError, match=r'exact K64\+K1 state layout'):
        PairedObservationDataset(
            cache,
            jsonl,
            'train',
            'state',
            decoder_family='spatial_cls_grid_v1',
        )


def test_spatial_cls_trainer_preflight_binds_frozen_upstream_modules(tmp_path):
    train_cache, train_jsonl = export(tmp_path, 'train', tokens=65)
    eval_cache, eval_jsonl = export(tmp_path, 'eval', tokens=65)
    args = SimpleNamespace(
        train_cache=train_cache,
        train_jsonl=train_jsonl,
        eval_cache=eval_cache,
        eval_jsonl=eval_jsonl,
        condition='state',
        decoder_family='spatial_cls_grid_v1',
        seed=20260921,
        steps=4000,
        batch=32,
        preflight_only=True,
    )
    identity = train(args)
    assert identity['decoder_family'] == 'spatial_cls_grid_v1'
    assert identity['trainable_modules'] == ['SpatialCLSConditionedFlowUNet']
    assert identity['frozen_modules'] == [
        'Qwen', 'StateProjector', 'WorldModel', 'ValueHead', 'OutcomeHead'
    ]
    assert identity['train']['state_layout']['state_tokens'] == 65
    assert identity['eval']['state_layout']['global_role'] == 'dino_cls'


def test_spatial_cls_trainer_excludes_train_rgb_duplicates_from_eval(tmp_path):
    train_cache, train_jsonl = export(tmp_path, 'train', tokens=65)
    eval_cache, eval_jsonl = export(tmp_path, 'eval', tokens=65)
    train_record = json.loads(train_jsonl.read_text())
    eval_record = json.loads(eval_jsonl.read_text())
    shutil.copyfile(train_record['image_paths'][0], eval_record['image_paths'][0])
    args = SimpleNamespace(
        train_cache=train_cache,
        train_jsonl=train_jsonl,
        eval_cache=eval_cache,
        eval_jsonl=eval_jsonl,
        condition='state',
        decoder_family='spatial_cls_grid_v1',
        seed=20260921,
        steps=4000,
        batch=32,
        preflight_only=True,
    )
    identity = train(args)
    assert identity['overlap']['before_filter']['shared_image_hashes'] == 1
    assert identity['overlap']['after_filter']['shared_image_hashes'] == 0
    assert identity['overlap']['train_filter']['excluded_observation_count'] == 1
    assert identity['train']['observation_count'] == 4
    assert identity['eval']['observation_count'] == 5


def test_exact_architecture_size_and_spatial_order():
    model = SpatialConditionedFlowUNet(CFMConfig(token_count=64))
    assert sum(p.numel() for p in model.parameters()) == 17565571
    condition = torch.arange(64 * 1024).float().reshape(1, -1)
    spatial = model.reshape_condition(condition)
    assert spatial[0, 17, 2, 3] == (2 * 8 + 3) * 1024 + 17


def test_decoder_checkpoint_preserves_training_rng_and_optimizer(tmp_path):
    torch.manual_seed(7)
    model = SpatialConditionedFlowUNet(CFMConfig(image_size=16, token_count=16,
        token_dim=4, base_channels=4, condition_dim=8, time_dim=16))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    row_rng, flow_rng = torch.Generator().manual_seed(3), torch.Generator().manual_seed(4)
    images, condition = torch.randn(2, 3, 16, 16), torch.randn(2, 64)
    conditional_flow_matching_loss(model, images, condition, generator=flow_rng).backward()
    optimizer.step()
    path = tmp_path / 'latest.pt'
    identity = {'test': True, 'decoder_family': 'spatial_grid_v1'}
    save_decoder(path, model, optimizer, 1, identity, row_rng, flow_rng)
    restored, payload = load_decoder(path)
    assert payload['step'] == 1 and payload['optimizer']['state']
    assert all(not p.requires_grad for p in restored.parameters())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key])
    replay = torch.Generator().set_state(payload['flow_rng'])
    row_replay = torch.Generator().set_state(payload['row_rng'])
    torch.testing.assert_close(torch.randint(100, (32,), generator=row_rng),
                               torch.randint(100, (32,), generator=row_replay))
    restored.requires_grad_(True).train()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
    restored_optimizer.load_state_dict(payload['optimizer'])
    for decoder, optim, rng in ((model, optimizer, flow_rng), (restored, restored_optimizer, replay)):
        optim.zero_grad(set_to_none=True)
        conditional_flow_matching_loss(decoder, images, condition, generator=rng).backward()
        optim.step()
    for first, second in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_spatial_cls_decoder_checkpoint_round_trip_and_family_binding(tmp_path):
    torch.manual_seed(31)
    model = SpatialCLSConditionedFlowUNet(
        SpatialCLSCFMConfig(
            image_size=16,
            token_dim=4,
            base_channels=4,
            condition_dim=8,
            time_dim=16,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    row_rng = torch.Generator().manual_seed(32)
    flow_rng = torch.Generator().manual_seed(33)
    images = torch.randn(2, 3, 16, 16)
    condition = torch.randn(2, 65 * 4)
    conditional_flow_matching_loss(
        model, images, condition, generator=flow_rng
    ).backward()
    optimizer.step()
    path = tmp_path / 'cls.pt'
    identity = {'decoder_family': 'spatial_cls_grid_v1'}
    save_decoder(
        path,
        model,
        optimizer,
        7,
        identity,
        row_rng,
        flow_rng,
        best_val=0.25,
    )
    restored, payload = load_decoder(path)
    assert payload['schema'] == CLS_SCHEMA
    assert payload['best_val'] == 0.25
    assert restored.decoder_family == 'spatial_cls_grid_v1'
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key])

    payload['identity']['decoder_family'] = 'spatial_grid_v1'
    with pytest.raises(ValueError, match='schema/family mismatch'):
        decoder_family_from_payload(payload)
    with pytest.raises(ValueError, match='identity does not match'):
        save_decoder(
            tmp_path / 'bad.pt',
            model,
            optimizer,
            7,
            {'decoder_family': 'spatial_grid_v1'},
            row_rng,
            flow_rng,
        )


def test_legacy_spatial_decoder_schema_loads_without_family_field(tmp_path):
    model = SpatialConditionedFlowUNet(
        CFMConfig(
            image_size=16,
            token_count=64,
            token_dim=4,
            base_channels=4,
            condition_dim=8,
            time_dim=16,
        )
    )
    path = tmp_path / 'legacy.pt'
    torch.save(
        {
            'schema': 'paired_spatial_cfm_v1',
            'config': model.config.to_metadata(),
            'identity': {'legacy': True},
            'model': model.state_dict(),
        },
        path,
    )
    restored, payload = load_decoder(path)
    assert restored.decoder_family == 'spatial_grid_v1'
    assert decoder_family_from_payload(payload) == 'spatial_grid_v1'
