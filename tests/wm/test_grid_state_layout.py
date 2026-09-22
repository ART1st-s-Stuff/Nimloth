from __future__ import annotations

import pytest
import torch
from torch import nn

from nimloth.wm.grid import (
    GridPredictorConfig,
    GridWorldModel,
    ResidualTemporalSpatialGridPredictor,
    fixed_2d_sincos_position,
)
from nimloth.wm.layout import GridStateLayout


class _RecordingHead(nn.Module):
    def __init__(self, *, outcome: bool = False) -> None:
        super().__init__()
        self.outcome = outcome
        self.last_shape = None
        self.last_value = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.last_shape = tuple(value.shape)
        self.last_value = value.detach().clone()
        if self.outcome:
            return value.mean((-2, -1))
        return value.new_zeros(*value.shape[:-1], 8)


def _config() -> GridPredictorConfig:
    return GridPredictorConfig(
        grid_tokens=5,
        spatial_grid_size=2,
        global_tokens=1,
        position_encoding="fixed_2d_sincos_v1",
        emb_dim=8,
        action_dim=3,
        history_size=1,
        depth=1,
        heads=1,
        dim_head=8,
        mlp_dim=16,
        dropout=0.0,
    )


def test_fixed_2d_position_is_deterministic_persistent_and_global_zero() -> None:
    layout = GridStateLayout(
        spatial_grid_size=2, global_tokens=1, global_role="dino_cls"
    )
    first = fixed_2d_sincos_position(layout, 8)
    second = fixed_2d_sincos_position(layout, 8)
    assert torch.equal(first, second)
    assert first.shape == (1, 5, 8)
    assert torch.count_nonzero(first[:, -1]) == 0
    assert torch.unique(first[:, :4], dim=1).shape[1] == 4

    model = ResidualTemporalSpatialGridPredictor(_config())
    assert "body.spatial_position" in model.state_dict()
    assert "body.spatial_position" not in dict(model.named_parameters())


def test_spatial_identity_gate_accepts_k64_reference_and_rejects_drift() -> None:
    layout = GridStateLayout(
        spatial_grid_size=2, global_tokens=1, global_role="dino_cls"
    )
    reference = torch.randn(3, 4, 8)
    candidate = torch.cat((reference.clone(), torch.randn(3, 1, 8)), dim=1)
    assert layout.assert_spatial_identity(reference, candidate) == 0.0

    candidate[1, 2, 3] += 1e-3
    with pytest.raises(ValueError, match="identity gate failed"):
        layout.assert_spatial_identity(reference, candidate)


def test_k65_residual_copy_and_heads_receive_only_spatial_slots(tmp_path) -> None:
    predictor = ResidualTemporalSpatialGridPredictor(_config()).eval()
    state = torch.randn(2, 5, 8)
    actions = torch.tensor([0, 1])
    assert torch.equal(predictor(state, actions), state)

    value = _RecordingHead()
    outcome = _RecordingHead(outcome=True)
    wm = GridWorldModel(
        state_proj=nn.Identity(),
        wm_predictor=predictor,
        value_head=value,
        outcome_head=outcome,
    )
    wm.predict_action_values(state)
    wm.predict_outcome_logits(state[:, None])
    assert value.last_shape == (2, 8)
    torch.testing.assert_close(value.last_value, state[:, :4].mean(1))
    assert outcome.last_shape == (2, 1, 4, 8)

    predictor.save_checkpoint(tmp_path)
    restored = ResidualTemporalSpatialGridPredictor.load_checkpoint(tmp_path)
    assert restored.config.state_layout == predictor.config.state_layout
    assert torch.equal(restored.body.spatial_position, predictor.body.spatial_position)
