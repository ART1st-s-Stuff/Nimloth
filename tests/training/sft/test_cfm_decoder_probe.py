import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from experiments.training.sft.stage3.cfm_decoder_probe import (
    PairedObservationDataset, PROVENANCE, load_decoder, save_decoder, validate_pair)
from experiments.training.sft.stage3.frozen_wm_diagnostic import seal_cache, file_sha256
from nimloth.training.sft.stage3.diagnostics import FrozenWMTrajectoryWriter
from nimloth.recon.cfm.model import CFMConfig, SpatialConditionedFlowUNet
from nimloth.recon.cfm.flow import conditional_flow_matching_loss


def export(tmp_path, split):
    paths = []
    for step in range(5):
        path = tmp_path / f'{split}{step}.png'
        Image.new('RGB', (12, 10), (step, 10, 20)).save(path)
        paths.append(str(path))
    jsonl = tmp_path / f'{split}.jsonl'
    jsonl.write_text(json.dumps({'id': split, 'image_paths': paths,
                                'action_indices': [0, 1, 2, 3]}) + '\n')
    identity = {key: key for key in PROVENANCE} | {'split': split,
        'split_sha256': file_sha256(jsonl), 'prediction_horizon': 4,
        'grid_tokens': 64, 'state_dim': 1024, 'action_dim': 8,
        'trajectory_count': 1, 'window_count': 1}
    writer = FrozenWMTrajectoryWriter(tmp_path / split, rank=0, identity=identity)
    states = torch.randn(5, 64, 1024)
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
    save_decoder(path, model, optimizer, 1, {'test': True}, row_rng, flow_rng)
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
