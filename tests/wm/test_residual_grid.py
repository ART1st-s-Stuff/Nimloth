from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from nimloth.wm.grid import (  # noqa: E402
    GridPredictorConfig,
    ResidualTemporalSpatialGridPredictor,
    TemporalSpatialGridPredictor,
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


def test_residual_copy_gradient_and_checkpoint_roundtrip(tmp_path) -> None:
    model = _model()
    state = torch.randn(2, 1, 4, 8, requires_grad=True)
    actions = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    prediction = model.rollout_from_history(state, actions[:, :0], actions)
    torch.testing.assert_close(prediction, state.expand(-1, 4, -1, -1), rtol=0, atol=0)
    prediction.sum().backward()
    torch.testing.assert_close(state.grad, torch.full_like(state, 4))
    assert model.delta_head.weight.grad.abs().sum() > 0
    for parameter in model.body.parameters():
        if parameter.grad is not None:
            assert torch.count_nonzero(parameter.grad) == 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    optimizer.step()
    model.save_checkpoint(tmp_path / "residual")
    restored = ResidualTemporalSpatialGridPredictor.load_checkpoint(tmp_path / "residual")
    torch.testing.assert_close(restored(state, actions[:, :1]), model(state, actions[:, :1]))
    with pytest.raises((TypeError, ValueError)):
        TemporalSpatialGridPredictor.load_checkpoint(tmp_path / "residual")
    TemporalSpatialGridPredictor(model.config).save_checkpoint(tmp_path / "direct")
    with pytest.raises(ValueError, match="schema"):
        ResidualTemporalSpatialGridPredictor.load_checkpoint(tmp_path / "direct")
