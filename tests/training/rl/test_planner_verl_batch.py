from __future__ import annotations

from dataclasses import dataclass

import torch

from nimloth.training.rl.planner_verl_adapter import planner_update_inputs
from nimloth.training.rl.planner_verl_batch import (
    PreparedPlannerRow,
    build_planner_rank_rounds,
    build_replicated_planner_gate_round,
    prepare_planner_behavior_row,
)
from nimloth.wm import PlannerPolicyHead, ValueHead


@dataclass(frozen=True)
class _Transition:
    name: str
    state: torch.Tensor
    action_index: int
    behavior: torch.Tensor

    def rollout_decision_state(self) -> torch.Tensor:
        return self.state

    def behavior_action_log_probs(self) -> torch.Tensor:
        return self.behavior


def _heads() -> tuple[ValueHead, PlannerPolicyHead]:
    value = ValueHead(emb_dim=3, num_actions=2)
    policy = PlannerPolicyHead(emb_dim=3, num_actions=2)
    with torch.no_grad():
        for parameter in value.parameters():
            parameter.fill_(0.1)
        for parameter in policy.parameters():
            parameter.fill_(0.2)
    return value, policy


def _row(name: str = "real") -> PreparedPlannerRow:
    transition = _Transition(
        name=name,
        state=torch.ones(2, 3),
        action_index=1,
        behavior=torch.log(torch.tensor([0.5, 0.5])),
    )
    return PreparedPlannerRow(
        transition=transition,  # type: ignore[arg-type]
        return_target=torch.tensor(2.0),
        old_action_value=torch.tensor(0.25),
        old_policy_log_prob=torch.tensor(-0.5),
        policy_advantage=torch.tensor(1.75),
    )


def test_behavior_row_reconstructs_value_and_policy_from_real_state() -> None:
    value, policy = _heads()
    state = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    with torch.no_grad():
        pooled = state.mean(dim=0, keepdim=True)
        values = value(pooled).float().squeeze(0)
        log_probs = torch.log_softmax(policy(pooled).float(), dim=-1).squeeze(0)
    transition = _Transition(
        name="actual",
        state=state,
        action_index=1,
        behavior=log_probs,
    )

    row = prepare_planner_behavior_row(
        transition,  # type: ignore[arg-type]
        return_target=torch.tensor(4.0),
        value_head=value,
        planner_policy_head=policy,
        temperature=1.0,
    )

    assert torch.equal(row.old_action_value, values[1])
    assert torch.equal(row.old_policy_log_prob, log_probs[1])
    expected_state_value = (log_probs.exp() * values).sum()
    assert torch.equal(row.policy_advantage, torch.tensor(4.0) - expected_state_value)


def test_behavior_row_rejects_mismatched_stored_policy() -> None:
    value, policy = _heads()
    transition = _Transition(
        name="mismatch",
        state=torch.ones(2, 3),
        action_index=0,
        behavior=torch.log(torch.tensor([0.9, 0.1])),
    )

    try:
        prepare_planner_behavior_row(
            transition,  # type: ignore[arg-type]
            return_target=torch.tensor(1.0),
            value_head=value,
            planner_policy_head=policy,
            temperature=1.0,
        )
    except ValueError as error:
        assert "do not match" in str(error)
    else:  # pragma: no cover
        raise AssertionError("mismatched behavior policy was accepted")


def test_rank_rounds_pad_only_missing_ranks_with_zero_weight() -> None:
    rows = (_row("a"), _row("b"), _row("c"))
    rounds = build_planner_rank_rounds(
        rows,
        token_counts=(10, 20, 30),
        dino_grid_targets=(torch.ones(1, 2),) * 3,
        world_size=2,
        provisional_update_id="provisional",
    )

    assert len(rounds) == 2
    assert all(len(rank_batches) == 2 for rank_batches in rounds)
    first = tuple(planner_update_inputs(batch) for batch in rounds[0])
    second = tuple(planner_update_inputs(batch) for batch in rounds[1])
    assert tuple(row.loss_weights for row in first) == ((2.0,), (2.0,))
    assert tuple(row.loss_weights for row in second) == ((2.0,), (0.0,))
    assert first[0].transitions[0].name == "a"
    assert first[1].transitions[0].name == "b"
    assert second[0].transitions[0].name == "c"
    assert second[1].transitions[0].name == "a"
    assert all(row.total_transitions == 3 for row in (*first, *second))


def test_replicated_gate_round_uses_one_real_objective_on_every_rank() -> None:
    rounds = build_replicated_planner_gate_round(
        _row(),
        token_count=6332,
        dino_grid_target=torch.ones(1, 2),
        world_size=3,
        provisional_update_id="gate",
    )
    restored = tuple(planner_update_inputs(batch) for batch in rounds[0])
    assert len(restored) == 3
    assert all(row.loss_weights == (1.0,) for row in restored)
    assert all(row.total_transitions == 1 for row in restored)
    assert all(row.token_counts == (6332,) for row in restored)
