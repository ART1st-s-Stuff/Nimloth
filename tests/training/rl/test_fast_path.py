"""WM+ValueHead fast-path state-machine tests."""

from __future__ import annotations

import torch

from nimloth.training.rl.rollout import WMValueFastPathController


class CountingEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, observation: object) -> torch.Tensor:
        self.calls += 1
        return torch.tensor([[float(observation)]])


class AddActionPredictor:
    def __init__(self) -> None:
        self.calls = 0

    def predict_next_emb(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        self.calls += 1
        return state + action.float().unsqueeze(1) + 1.0


class GreedyValueHead:
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        # State 10 -> action 2, predicted state 13 -> action 5.
        best = state[:, 0].long().remainder(8)
        values = torch.zeros(state.shape[0], 8)
        return values.scatter(1, best.unsqueeze(1), 1.0)


def test_fast_path_uses_predicted_state_until_segment_resync() -> None:
    encoder = CountingEncoder()
    predictor = AddActionPredictor()
    controller = WMValueFastPathController(
        encode_state=encoder,
        predictor=predictor,
        value_head=GreedyValueHead(),
        horizon=2,
    )

    first = controller.select_action(10)
    assert first.action_index == 2
    assert first.state_source == "qwen_gt"
    assert first.fast_path_step == 0
    controller.advance(first.action_index, done=False)

    # Observation 999 must be ignored inside the same fast-path segment.
    second = controller.select_action(999)
    assert second.action_index == 5  # predicted state = 10 + 2 + 1 = 13
    assert second.state_source == "wm_predicted"
    assert second.fast_path_step == 1
    assert encoder.calls == 1
    controller.advance(second.action_index, done=False)

    # Horizon reached: the next step must re-sync from the real observation.
    third = controller.select_action(7)
    assert third.action_index == 7
    assert third.state_source == "qwen_gt"
    assert third.fast_path_step == 0
    assert encoder.calls == 2
    assert predictor.calls == 1


def test_fast_path_done_forces_new_qwen_gt_segment() -> None:
    encoder = CountingEncoder()
    controller = WMValueFastPathController(
        encode_state=encoder,
        predictor=AddActionPredictor(),
        value_head=GreedyValueHead(),
        horizon=4,
    )

    decision = controller.select_action(1)
    controller.advance(decision.action_index, done=True)
    next_decision = controller.select_action(6)

    assert next_decision.state_source == "qwen_gt"
    assert next_decision.action_index == 6
    assert encoder.calls == 2


def test_fast_path_rejects_non_positive_horizon() -> None:
    try:
        WMValueFastPathController(
            encode_state=CountingEncoder(),
            predictor=AddActionPredictor(),
            value_head=GreedyValueHead(),
            horizon=0,
        )
        raised = False
    except ValueError:
        raised = True
    assert raised
