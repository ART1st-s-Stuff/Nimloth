from __future__ import annotations

from pathlib import Path

import torch

from nimloth.wm import PlannerPolicyHead, ValueHead


def test_planner_policy_head_matches_value_head_structure() -> None:
    value_head = ValueHead(emb_dim=5, hidden_dim=7, num_actions=8)
    policy_head = PlannerPolicyHead(emb_dim=5, hidden_dim=7, num_actions=8)

    assert [type(module) for module in policy_head.net] == [
        type(module) for module in value_head.net
    ]
    assert [tuple(parameter.shape) for parameter in policy_head.parameters()] == [
        tuple(parameter.shape) for parameter in value_head.parameters()
    ]
    assert policy_head(torch.ones(3, 5)).shape == (3, 8)


def test_planner_policy_head_checkpoint_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(7)
    policy_head = PlannerPolicyHead(emb_dim=4, hidden_dim=6, num_actions=8)
    policy_head.save_checkpoint(tmp_path)

    restored = PlannerPolicyHead.load_checkpoint(
        tmp_path,
        emb_dim=4,
        hidden_dim=6,
        num_actions=8,
    )

    for expected, actual in zip(
        policy_head.parameters(),
        restored.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)
