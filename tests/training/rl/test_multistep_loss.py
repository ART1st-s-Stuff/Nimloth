"""Multi-step RL dynamics loss and contiguous-window tests."""

from __future__ import annotations

import torch
from torch import nn

from nimloth.training.rl.loss import compute_multistep_predictor_loss
from nimloth.training.rl.trainer import collate_transition_windows


class IdentityProjector(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim == 3:
            return hidden.flatten(1)
        return hidden


class AdditivePredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return state + self.scale * actions.float().unsqueeze(1)


def test_multistep_loss_recurses_through_predicted_states() -> None:
    predictor = AdditivePredictor()
    current = torch.tensor([[0.0, 0.0]])
    actions = torch.tensor([[1, 2, 3]])
    # Recursive predictions are [1,1], [3,3], [6,6].
    targets = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [6.0, 6.0]]])
    mask = torch.ones(1, 3, dtype=torch.bool)

    loss, metrics = compute_multistep_predictor_loss(
        qwen_hidden_current=current,
        qwen_hidden_targets=targets,
        action_sequences=actions,
        valid_mask=mask,
        state_proj=IdentityProjector(),
        wm_predictor=predictor,
        loss_decay=1.0,
    )

    assert loss.item() == 0.0
    assert metrics["wm_mse_h1"] == 0.0
    assert metrics["wm_mse_h3"] == 0.0


def test_multistep_loss_masks_short_trajectory_and_backpropagates() -> None:
    predictor = AdditivePredictor()
    current = torch.tensor([[0.0], [10.0]])
    actions = torch.tensor([[1, 2], [1, 0]])
    targets = torch.tensor([[[1.0], [3.0]], [[11.0], [999.0]]])
    mask = torch.tensor([[True, True], [True, False]])

    loss, _ = compute_multistep_predictor_loss(
        qwen_hidden_current=current,
        qwen_hidden_targets=targets,
        action_sequences=actions,
        valid_mask=mask,
        state_proj=IdentityProjector(),
        wm_predictor=predictor,
        loss_decay=0.5,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert predictor.scale.grad is not None
    assert torch.isfinite(predictor.scale.grad)


def test_multistep_loss_supports_k_query_hidden_blocks() -> None:
    predictor = AdditivePredictor()
    current = torch.tensor([[[0.0], [10.0]]])  # [B=1,k=2,H=1]
    targets = torch.tensor([[[[1.0], [11.0]], [[3.0], [13.0]]]])
    actions = torch.tensor([[1, 2]])
    mask = torch.ones(1, 2, dtype=torch.bool)

    loss, _ = compute_multistep_predictor_loss(
        qwen_hidden_current=current,
        qwen_hidden_targets=targets,
        action_sequences=actions,
        valid_mask=mask,
        state_proj=IdentityProjector(),
        wm_predictor=predictor,
    )
    assert loss.item() == 0.0


def test_collate_transition_windows_pads_only_with_masked_rows() -> None:
    batch = [
        {
            "qwen_hidden_current": torch.tensor([0.0, 1.0]),
            "qwen_hidden_targets": torch.tensor([[1.0, 2.0], [2.0, 3.0]]),
            "action_sequence": torch.tensor([1, 2]),
        },
        {
            "qwen_hidden_current": torch.tensor([4.0, 5.0]),
            "qwen_hidden_targets": torch.tensor([[5.0, 6.0]]),
            "action_sequence": torch.tensor([3]),
        },
    ]

    current, targets, actions, mask = collate_transition_windows(batch)

    assert current.shape == (2, 2)
    assert targets.shape == (2, 2, 2)
    assert actions.tolist() == [[1, 2], [3, 0]]
    assert mask.tolist() == [[True, True], [True, False]]
