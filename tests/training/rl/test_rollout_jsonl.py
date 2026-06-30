"""Tests for JSONLRolloutCollector: source loading, iteration cycling, distributed safety."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import pytest

from nimloth.training.rl.rollout import JSONLRolloutCollector, RolloutTrajectory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_traj(record_id: str, num_steps: int = 3) -> RolloutTrajectory:
    return RolloutTrajectory(
        record_id=record_id,
        image_paths=[f"/tmp/{record_id}_step{s}.png" for s in range(num_steps + 1)],
        action_indices=[i % 8 for i in range(num_steps)],
        action_names=["moveahead", "rotateright", "lookup"][:num_steps],
        action_log_probs=[[-2.0] * 8 for _ in range(num_steps)],
        nav_instruction="Go to the couch.",
        success=(num_steps % 2 == 0),
        reward=10.0 if num_steps % 2 == 0 else 0.0,
    )


def _write_jsonl(path: Path, trajectories: list[RolloutTrajectory]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for traj in trajectories:
            f.write(json.dumps(traj.to_record(), ensure_ascii=False) + "\n")


def _make_collector_from_trajs(
    tmp_path: Path, trajectories: list[RolloutTrajectory], filename: str = "trajs.jsonl"
) -> JSONLRolloutCollector:
    jsonl_path = tmp_path / filename
    _write_jsonl(jsonl_path, trajectories)
    return JSONLRolloutCollector(sources=[jsonl_path])


# ---------------------------------------------------------------------------
# Basic loading
# ---------------------------------------------------------------------------


def test_jsonl_collector_loads_trajectories(tmp_path: Path) -> None:
    trajs = [_make_traj(f"ep_{i:03d}", num_steps=4) for i in range(8)]
    collector = _make_collector_from_trajs(tmp_path, trajs)

    result = collector.collect(num_episodes=4)
    assert len(result) == 4
    assert all(isinstance(t, RolloutTrajectory) for t in result)


def test_jsonl_collector_returns_available_when_fewer_than_requested(tmp_path: Path) -> None:
    """If loop=False and fewer trajectories than requested, return what's available."""
    trajs = [_make_traj(f"ep_{i:03d}") for i in range(3)]
    collector = _make_collector_from_trajs(tmp_path, trajs, filename="few.jsonl")
    collector._loop = False

    result1 = collector.collect(num_episodes=10)
    assert len(result1) == 3

    result2 = collector.collect(num_episodes=5)
    assert len(result2) == 0  # cursor exhausted, no loop


def test_jsonl_collector_cycles_when_loop_enabled(tmp_path: Path) -> None:
    """With loop=True, data cycles after exhaustion."""
    trajs = [_make_traj(f"ep_{i:03d}") for i in range(3)]
    collector = _make_collector_from_trajs(tmp_path, trajs)

    # Request more than total
    result1 = collector.collect(num_episodes=5)
    assert len(result1) == 5  # 3 + 2 cycled

    # Check cycling: first 3 unique, then repeats
    ids1 = [t.record_id for t in result1]
    assert len(set(ids1[:3])) == 3
    assert ids1[3] in ids1[:3]  # cycled back

    # Continuous cycling across calls
    result2 = collector.collect(num_episodes=7)
    assert len(result2) == 7


def test_jsonl_collector_from_directory(tmp_path: Path) -> None:
    """Collector discovers .jsonl files inside a directory."""
    subdir = tmp_path / "rollouts"
    subdir.mkdir()
    _write_jsonl(subdir / "batch1.jsonl", [_make_traj(f"a_{i:03d}") for i in range(3)])
    _write_jsonl(subdir / "batch2.jsonl", [_make_traj(f"b_{i:03d}") for i in range(2)])

    collector = JSONLRolloutCollector(sources=[subdir])
    result = collector.collect(num_episodes=10)
    assert len(result) >= 5  # all 5 trajectories loaded, then cycled


def test_jsonl_collector_multiple_source_files(tmp_path: Path) -> None:
    """Multiple explicit file paths work."""
    f1 = tmp_path / "a.jsonl"
    f2 = tmp_path / "b.jsonl"
    _write_jsonl(f1, [_make_traj(f"a_{i:03d}") for i in range(3)])
    _write_jsonl(f2, [_make_traj(f"b_{i:03d}") for i in range(4)])

    collector = JSONLRolloutCollector(sources=[f1, f2])
    assert collector.total_trajectories == 7

    result = collector.collect(num_episodes=10)
    assert len(result) == 10


def test_jsonl_collector_reads_gzip_files(tmp_path: Path) -> None:
    """Directory expansion advertises .jsonl.gz, so loading must support gzip."""
    gz_path = tmp_path / "compressed.jsonl.gz"
    trajs = [_make_traj(f"gz_{i:03d}") for i in range(4)]
    with gzip.open(gz_path, "wt", encoding="utf-8") as f:
        for traj in trajs:
            f.write(json.dumps(traj.to_record(), ensure_ascii=False) + "\n")

    collector = JSONLRolloutCollector(sources=[tmp_path])
    assert collector.total_trajectories == 4


# ---------------------------------------------------------------------------
# Determinism (critical for distributed FSDP safety)
# ---------------------------------------------------------------------------


def test_jsonl_collector_deterministic_across_calls(tmp_path: Path) -> None:
    """Same init + same collect calls → same results (simulates multiple ranks)."""
    trajs = [_make_traj(f"ep_{i:03d}") for i in range(20)]
    jsonl_path = tmp_path / "trajs.jsonl"
    _write_jsonl(jsonl_path, trajs)

    # Simulate two "ranks" with identical collectors
    c1 = JSONLRolloutCollector(sources=[jsonl_path])
    c2 = JSONLRolloutCollector(sources=[jsonl_path])

    for _ in range(5):
        r1 = c1.collect(num_episodes=8)
        r2 = c2.collect(num_episodes=8)
        assert [t.record_id for t in r1] == [t.record_id for t in r2]


def test_jsonl_collector_empty_sources_raises(tmp_path: Path) -> None:
    """Empty sources list raises FileNotFoundError."""
    collector = JSONLRolloutCollector(sources=[])
    with pytest.raises(FileNotFoundError, match="未找到任何 JSONL 文件"):
        collector.collect(num_episodes=1)


def test_jsonl_collector_nonexistent_source(tmp_path: Path) -> None:
    """Non-existent file path is silently skipped (no crash), raises if no data found."""
    collector = JSONLRolloutCollector(sources=[tmp_path / "nonexistent.jsonl"])
    with pytest.raises(FileNotFoundError, match="未找到任何 JSONL 文件"):
        collector.collect(num_episodes=1)


# ---------------------------------------------------------------------------
# Advantage normalization batch-size-1 NaN fix
# ---------------------------------------------------------------------------


def test_advantage_std_single_sample_no_nan() -> None:
    """compute_advantages with single sample must not produce NaN."""
    import torch
    from nimloth.training.rl.loss import compute_advantages

    targets = torch.tensor([5.0])
    values = torch.tensor([4.0])
    result = compute_advantages(value_targets=targets, predicted_values=values)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()
    # Mean-centered → should be 0 for single sample
    assert result.item() == 0.0


def test_advantage_std_multi_sample() -> None:
    """compute_advantages with multiple samples works normally."""
    import torch
    from nimloth.training.rl.loss import compute_advantages

    torch.manual_seed(42)
    targets = torch.tensor([5.0, 3.0, 7.0])
    values = torch.tensor([4.0, 2.0, 8.0])
    result = compute_advantages(value_targets=targets, predicted_values=values)
    assert result.shape == (3,)
    assert torch.isclose(result.mean(), torch.tensor(0.0), atol=1e-7)
    assert torch.isclose(result.std(unbiased=False), torch.tensor(1.0), atol=1e-3)


def test_advantage_std_batch_size_one_explicit() -> None:
    """Verify that torch.std with unbiased=False avoids NaN when N=1."""
    import torch

    # unbiased=True (default): NaN when N=1 (division by 0)
    x = torch.tensor([3.0])
    assert torch.isnan(x.std(unbiased=True))

    # unbiased=False: 0 when N=1 (all values identical)
    assert x.std(unbiased=False).item() == 0.0
    assert not torch.isnan(x.std(unbiased=False))
