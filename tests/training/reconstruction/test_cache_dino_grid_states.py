from pathlib import Path

import torch

from nimloth.training.reconstruction.cache_dino_grid_states import (
    GRID_SHAPE,
    _valid_shard,
    contiguous_rank_bounds,
    rank_shard_specs,
)


def test_rank_shards_cover_dataset_exactly_once() -> None:
    total = 103
    specs = [
        spec
        for rank in range(4)
        for spec in rank_shard_specs(
            total,
            rank=rank,
            world_size=4,
            shard_size=9,
        )
    ]
    ranges = [(start, end) for _local, start, end in specs]
    assert ranges[0][0] == 0
    assert ranges[-1][1] == total
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
    assert [contiguous_rank_bounds(total, rank, 4) for rank in range(4)] == [
        (0, 25),
        (25, 51),
        (51, 77),
        (77, 103),
    ]


def test_resume_shard_requires_contract_range_shape_and_finite_states(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shard.pt"
    payload = {
        "contract_fingerprint": "abc",
        "start": 4,
        "end": 7,
        "state_emb": torch.zeros(3, *GRID_SHAPE, dtype=torch.float16),
        "rows": [{}, {}, {}],
    }
    torch.save(payload, path)
    assert _valid_shard(
        path,
        contract_fingerprint="abc",
        start=4,
        end=7,
    )
    assert not _valid_shard(
        path,
        contract_fingerprint="wrong",
        start=4,
        end=7,
    )
    payload["state_emb"][0, 0, 0] = float("nan")
    torch.save(payload, path)
    assert not _valid_shard(
        path,
        contract_fingerprint="abc",
        start=4,
        end=7,
    )
