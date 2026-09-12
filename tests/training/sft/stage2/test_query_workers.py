"""CPU-only spawn/order checks; the tokenizer is an isolated interface fixture."""

import copy
import pickle
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, DistributedSampler

from nimloth.backbone.dino_grid import CachedDINOGridTargets, DINOV2_LARGE_IDENTITY
from nimloth.training.sft.stage1.trainer import data_loader_kwargs
from nimloth.training.sft.stage2.data import QueryAlignmentCollator
from test_query_alignment import TextProcessor, records
from test_standalone_dino_cache import fixture_cache


def test_validated_targets_pickle_reopens_shards_without_tensor_payload(tmp_path):
    identity, image, features = fixture_cache(tmp_path)
    targets = CachedDINOGridTargets.from_cache_root(tmp_path, identity=identity)
    state = targets.__getstate__()
    assert all(type(key) is int for key, _ in state['path_to_feature'].values())
    payload = pickle.dumps(targets)
    reopened = pickle.loads(payload)
    torch.testing.assert_close(reopened.load([image], device='cpu'), features)
    # A worker may itself be serialized: its shard IDs must refer to its own tensors.
    torch.testing.assert_close(pickle.loads(pickle.dumps(reopened)).load([image], device='cpu'), features)
    (tmp_path / 'shard_00000.pt').touch()
    with pytest.raises(ValueError, match='changed before worker load'):
        pickle.loads(payload)


def test_query_loader_policy_preserves_stage1():
    args = SimpleNamespace(num_workers=2, prefetch_factor=1)
    plain = data_loader_kwargs(args, None, use_cache=False, query=False)
    assert plain['num_workers'] == 0
    cached = data_loader_kwargs(args, None, use_cache=True, query=False)
    assert cached['num_workers'] == 2
    assert 'multiprocessing_context' not in cached
    query = data_loader_kwargs(args, None, use_cache=False, query=True)
    assert query['num_workers'] == 2
    assert query['multiprocessing_context'] == 'spawn'
    assert query['prefetch_factor'] == 1


def test_spawn_query_collation_matches_serial_order_and_main_rng(tmp_path):
    batch = records(tmp_path)
    samples = [copy.deepcopy(batch[0]) for _ in range(5)]
    for i, sample in enumerate(samples):
        sample['success'] = i % 2 == 0
    features = torch.randn(1, 4, 1024)
    shard = tmp_path / 'features.pt'
    torch.save({'features': features}, shard)
    targets = CachedDINOGridTargets(
        identity=DINOV2_LARGE_IDENTITY, grid_size=2,
        path_to_feature={str(tmp_path / 'image.png'): (features, 0)},
        cache_fingerprint='test',
        shard_references={id(features): CachedDINOGridTargets._shard_reference(shard, features)},
    )
    processor = TextProcessor()
    collator = QueryAlignmentCollator(processor, 1000, 4, targets)
    collator(samples)  # Stabilize the isolated fixture vocabulary before spawning.
    sampler = DistributedSampler(samples, num_replicas=1, rank=0, seed=42)
    sampler.set_epoch(3)
    args = SimpleNamespace(num_workers=2, prefetch_factor=1)
    parallel = DataLoader(samples, sampler=sampler, batch_size=1, timeout=30,
                          **data_loader_kwargs(args, collator, use_cache=False, query=True))
    serial = DataLoader(samples, sampler=sampler, batch_size=1, collate_fn=collator)
    torch.manual_seed(123)
    expected = list(serial)
    serial_rng = torch.get_rng_state()
    torch.manual_seed(123)
    try:
        actual = list(parallel)
        assert torch.equal(torch.get_rng_state(), serial_rng)
        for left, right in zip(expected, actual, strict=True):
            assert left.keys() == right.keys()
            for key in left:
                torch.testing.assert_close(left[key], right[key])
                assert right[key].device.type == 'cpu'
        # Persistent workers keep the sampler order and do not consume new main RNG.
        resumed = list(parallel)
        assert torch.equal(torch.get_rng_state(), serial_rng)
        for left, right in zip(actual[2:], resumed[2:], strict=True):
            for key in left:
                torch.testing.assert_close(left[key], right[key])
    finally:
        if parallel._iterator is not None:
            parallel._iterator._shutdown_workers()
