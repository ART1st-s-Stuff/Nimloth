from pathlib import Path

import torch

from nimloth.rollout.transitions import TransitionSample
from nimloth.training.reconstruction.cache_dino_grid_states import (
    GRID_SHAPE,
    _build_cfm_state_pairs,
    _cfm_pair_rows,
    _interleave_cfm_states,
    _valid_shard,
    contiguous_rank_bounds,
    rank_shard_specs,
)


def _sample(
    record_id: str,
    step: int,
    current_image: str,
    next_image: str,
) -> TransitionSample:
    return TransitionSample(
        record_id=record_id,
        step_index=step,
        prefix_messages=[],
        prefix_image_paths=[],
        action_index=step % 4,
        current_image_path=current_image,
        next_image_path=next_image,
    )


def test_cfm_pairs_use_actual_current_and_wm_predicted_next_targets() -> None:
    samples = [
        _sample("a", 0, "a0.png", "a1.png"),
        _sample("a", 1, "a1.png", "a2.png"),
    ]
    rows = _cfm_pair_rows(samples)
    assert [
        (row["pair_type"], row["current_image_path"], row["action_index"])
        for row in rows
    ] == [
        ("actual_current", "a0.png", 0),
        ("wm_predicted_next", "a1.png", 0),
        ("actual_current", "a1.png", 1),
        ("wm_predicted_next", "a2.png", 1),
    ]


def test_cfm_states_interleave_current_then_predicted_for_each_transition() -> None:
    current = torch.tensor([[[1.0]], [[2.0]]])
    predicted = torch.tensor([[[11.0]], [[12.0]]])
    combined = _interleave_cfm_states(current, predicted)
    assert combined[:, 0, 0].tolist() == [1.0, 11.0, 2.0, 12.0]


def test_cfm_next_condition_is_frozen_wm_prediction() -> None:
    class _Predictor(torch.nn.Module):
        def forward(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
            return state + actions[:, None, None]

    current = torch.tensor([[[1.0]], [[2.0]]])
    actions = torch.tensor([3, 5])
    combined = _build_cfm_state_pairs(_Predictor(), current, actions)
    assert combined[:, 0, 0].tolist() == [1.0, 4.0, 2.0, 7.0]


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
