from __future__ import annotations

from pathlib import Path

import pytest
import torch

from nimloth.training.rl.planner_verl_adapter import (
    PLANNER_VERL_SCHEMA_VERSION,
    PINNED_VERL_COMMIT,
    assert_pinned_verl_source,
    build_planner_update_dataproto,
    planner_update_inputs,
    planner_verl_micro_batches,
)


class _Transition:
    def __init__(self, identity: str) -> None:
        self.identity = identity


def _rows():  # type: ignore[no-untyped-def]
    transitions = (_Transition("a"), _Transition("b"))
    return {
        "transitions": transitions,
        "return_targets": (torch.tensor(1.0), torch.tensor(2.0)),
        "old_action_values": (torch.tensor(0.1), torch.tensor(0.2)),
        "old_policy_log_probs": (torch.tensor(-0.3), torch.tensor(-0.4)),
        "policy_advantages": (torch.tensor(0.9), torch.tensor(1.8)),
        "loss_weights": (4.0, 0.0),
        "token_counts": (100, 200),
        "dino_grid_targets": (
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0, 4.0]),
        ),
    }


def test_planner_dataproto_roundtrips_action_level_update_rows() -> None:
    data = build_planner_update_dataproto(
        **_rows(),
        total_transitions=17,
    )

    assert len(data) == 2
    assert data.meta_info == {
        "schema_version": PLANNER_VERL_SCHEMA_VERSION,
        "objective": "receding_horizon_planner_policy_ppo_v1",
        "total_transitions": 17,
        "has_dino_grid_targets": True,
    }
    assert set(data.batch.keys()) == {
        "return_targets",
        "old_action_values",
        "old_policy_log_probs",
        "policy_advantages",
        "loss_weights",
        "token_counts",
        "dino_grid_targets",
    }
    assert data.non_tensor_batch["transitions"][0].identity == "a"

    restored = planner_update_inputs(data)

    assert tuple(item.identity for item in restored.transitions) == ("a", "b")
    assert restored.total_transitions == 17
    assert restored.token_counts == (100, 200)
    torch.testing.assert_close(
        torch.stack(restored.dino_grid_targets),
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    assert restored.loss_weights == (4.0, 0.0)


def test_planner_dataproto_rejects_partial_or_nonfinite_rows() -> None:
    rows = _rows()
    rows["old_policy_log_probs"] = (torch.tensor(-0.3), None)
    with pytest.raises(ValueError, match="old_policy_log_probs"):
        build_planner_update_dataproto(
            **rows,
            total_transitions=2,
        )

    rows = _rows()
    rows["policy_advantages"] = (torch.tensor(float("nan")), torch.tensor(1.0))
    with pytest.raises(ValueError, match="policy_advantages must be finite"):
        build_planner_update_dataproto(
            **rows,
            total_transitions=2,
        )


def test_planner_dataproto_keeps_dino_targets_all_or_none() -> None:
    rows = _rows()
    rows["dino_grid_targets"] = (torch.tensor([1.0]), None)
    with pytest.raises(ValueError, match="all rows or no rows"):
        build_planner_update_dataproto(
            **rows,
            total_transitions=2,
        )

    rows = _rows()
    rows["dino_grid_targets"] = None
    data = build_planner_update_dataproto(
        **rows,
        total_transitions=2,
    )
    assert "dino_grid_targets" not in data.batch
    assert data.meta_info["has_dino_grid_targets"] is False
    assert planner_update_inputs(data).dino_grid_targets == (None, None)


def test_planner_verl_packing_budgets_actual_padded_tokens() -> None:
    rows = _rows()
    rows["transitions"] = tuple(_Transition(name) for name in "abcd")
    rows["return_targets"] = tuple(torch.tensor(float(i)) for i in range(4))
    rows["old_action_values"] = tuple(torch.tensor(0.0) for _ in range(4))
    rows["old_policy_log_probs"] = tuple(torch.tensor(-0.5) for _ in range(4))
    rows["policy_advantages"] = tuple(torch.tensor(1.0) for _ in range(4))
    rows["loss_weights"] = (1.0,) * 4
    rows["token_counts"] = (100, 200, 50, 180)
    rows["dino_grid_targets"] = None
    data = build_planner_update_dataproto(
        **rows,
        total_transitions=4,
    )

    batches = planner_verl_micro_batches(
        data,
        max_padded_tokens=400,
        max_rows=2,
    )

    assert [len(batch) for batch in batches] == [2, 2]
    restored = [planner_update_inputs(batch) for batch in batches]
    assert [item.token_counts for item in restored] == [(200, 180), (100, 50)]
    assert [
        transition.identity
        for item in restored
        for transition in item.transitions
    ] == ["b", "d", "a", "c"]
    assert all(
        max(item.token_counts) * len(item.token_counts) <= 400
        for item in restored
    )


def test_planner_verl_packing_rejects_one_over_budget_prefix() -> None:
    data = build_planner_update_dataproto(
        **_rows(),
        total_transitions=2,
    )

    with pytest.raises(ValueError, match="exceeds max_padded_tokens"):
        planner_verl_micro_batches(
            data,
            max_padded_tokens=199,
            max_rows=2,
        )


def test_planner_verl_runtime_is_exact_pinned_submodule() -> None:
    repo_root = Path(__file__).resolve().parents[3]

    source = assert_pinned_verl_source(repo_root / "external/VAGEN/verl")

    assert source.commit == PINNED_VERL_COMMIT
    assert source.root == (repo_root / "external/VAGEN/verl").resolve()
