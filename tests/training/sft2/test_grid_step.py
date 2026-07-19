from __future__ import annotations

import torch

from nimloth.training.sft2.grid_step import compute_grid_sft2_loss
from nimloth.wm.grid import GridLatentWMPredictor, SharedSlotProjector
from nimloth.wm.value_head import ValueHead


def test_grid_sft2_trains_one_joint_wm_and_value_only() -> None:
    torch.manual_seed(0)
    projector = SharedSlotProjector(6, 8, 10, grid_tokens=4)
    projector.requires_grad_(False)
    wm = GridLatentWMPredictor(
        grid_tokens=4,
        emb_dim=8,
        action_dim=6,
        depth=1,
        heads=2,
        mlp_dim=16,
    )
    value = ValueHead(emb_dim=8, hidden_dim=8)
    current = torch.randn(3, 4, 6)
    next_hidden = torch.randn(3, 4, 6)

    loss, metrics = compute_grid_sft2_loss(
        current_query_hidden=current,
        next_query_hidden=next_hidden,
        action_indices=torch.tensor([0, 2, 5]),
        value_targets=torch.tensor([1.0, 0.5, 0.0]),
        slot_projector=projector,
        grid_wm=wm,
        value_head=value,
    )
    loss.backward()

    assert loss.ndim == 0
    assert set(metrics) == {"grid_wm_mse", "value_mse", "total"}
    assert all(parameter.grad is None for parameter in projector.parameters())
    assert any(parameter.grad is not None for parameter in wm.parameters())
    assert any(parameter.grad is not None for parameter in value.parameters())
