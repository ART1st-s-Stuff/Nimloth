from __future__ import annotations

import pytest

from experiments.training.rl.gpu_gate_ppo_value_critic import (
    _assigned_qualifying_candidate,
)


def test_gate_prefers_distinct_global_long_prefixes() -> None:
    token_counts = [11_332, 16_184, 14_500, 9_000]

    assert _assigned_qualifying_candidate(
        token_counts,
        rank=0,
        minimum_state_tokens=14_000,
    ) == (1, 2, False)
    assert _assigned_qualifying_candidate(
        token_counts,
        rank=1,
        minimum_state_tokens=14_000,
    ) == (2, 2, False)


def test_gate_reuses_a_real_long_prefix_when_only_one_qualifies() -> None:
    token_counts = [11_332, 16_184, 9_000]

    assert _assigned_qualifying_candidate(
        token_counts,
        rank=1,
        minimum_state_tokens=14_000,
    ) == (1, 1, True)


def test_gate_fails_when_no_real_prefix_meets_the_contract() -> None:
    with pytest.raises(RuntimeError, match="maximum_tokens=11332, minimum=14000"):
        _assigned_qualifying_candidate(
            [9_000, 11_332],
            rank=0,
            minimum_state_tokens=14_000,
        )
