from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from nimloth.wm.grid import (  # noqa: E402
    GridPredictorConfig,
    ResidualTemporalSpatialGridPredictor,
)


def _model() -> ResidualTemporalSpatialGridPredictor:
    return ResidualTemporalSpatialGridPredictor(
        GridPredictorConfig(
            grid_tokens=4,
            emb_dim=8,
            history_size=1,
            depth=1,
            heads=2,
            dim_head=4,
            mlp_dim=16,
            dropout=0.0,
        )
    )


def test_residual_grid_predictor_is_exact_copy_at_initialization() -> None:
    model = _model().eval()
    state = torch.randn(3, 4, 8)
    action = torch.tensor([0, 2, 4])
    prediction = model(state, action)
    assert torch.equal(prediction, state)
    assert model.is_zero_initialized()


def test_residual_grid_predictor_learns_nonzero_delta() -> None:
    model = _model().train()
    state = torch.randn(3, 4, 8)
    target = state + 0.25
    loss = torch.nn.functional.mse_loss(model(state, torch.tensor([0, 2, 4])), target)
    loss.backward()
    assert model.delta_head.weight.grad is not None
    assert torch.isfinite(model.delta_head.weight.grad).all()
    assert model.delta_head.weight.grad.abs().sum() > 0


def test_residual_grid_predictor_sequence_preserves_shape() -> None:
    model = _model().eval()
    state = torch.randn(2, 1, 4, 8)
    action = torch.tensor([[0], [3]])
    prediction = model.predict_sequence(state, action)
    assert prediction.shape == state.shape
    assert torch.equal(prediction, state)
