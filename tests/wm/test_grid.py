from __future__ import annotations

import torch

from nimloth.training.sft2.history_cache import OnlineHistoryStateCache
from nimloth.wm.grid import (
    GridPredictorConfig,
    GridStateProjector,
    LeWMGridEncoder,
    SharedSlotProjector,
    TemporalSpatialGridPredictor,
)


def _open_attention_gate(predictor: TemporalSpatialGridPredictor) -> None:
    with torch.no_grad():
        bias = predictor.layers[0].adaLN_modulation[-1].bias
        start = 2 * predictor.emb_dim
        bias[start : start + predictor.emb_dim].fill_(1.0)


def test_temporal_spatial_predictor_is_causal_in_time_and_noncausal_in_space() -> None:
    torch.manual_seed(0)
    predictor = TemporalSpatialGridPredictor(
        GridPredictorConfig(
            grid_tokens=4,
            emb_dim=8,
            action_dim=3,
            history_size=3,
            depth=1,
            heads=2,
            dim_head=4,
            mlp_dim=16,
            dropout=0.0,
        )
    ).eval()
    _open_attention_gate(predictor)
    states = torch.randn(1, 3, 4, 8)
    actions = torch.tensor([[0, 1, 2]])
    baseline = predictor(states, actions)

    future_changed = states.clone()
    future_changed[:, 2, 3, 0] += 20.0
    future_output = predictor(future_changed, actions)
    torch.testing.assert_close(future_output[:, 0], baseline[:, 0])

    same_time_changed = states.clone()
    same_time_changed[:, 0, 3, 0] += 20.0
    same_time_output = predictor(same_time_changed, actions)
    assert not torch.allclose(same_time_output[:, 0, 0], baseline[:, 0, 0])


def test_grid_state_projector_freezes_sft1_weights_but_backpropagates_to_qwen() -> None:
    slot_projector = SharedSlotProjector(
        input_dim=6,
        output_dim=8,
        hidden_dim=12,
        grid_tokens=4,
    ).requires_grad_(False)
    online_encoder = LeWMGridEncoder(emb_dim=8, hidden_dim=16)
    projector = GridStateProjector(slot_projector, online_encoder)
    hidden = torch.randn(2, 4, 6, requires_grad=True)

    projector(hidden).square().mean().backward()

    assert hidden.grad is not None
    assert torch.count_nonzero(hidden.grad) > 0
    assert all(parameter.grad is None for parameter in slot_projector.parameters())
    assert any(parameter.grad is not None for parameter in online_encoder.parameters())


def test_online_history_cache_round_trips_grid_states(tmp_path) -> None:
    cache = OnlineHistoryStateCache()
    cache.start(epoch=1, phase="train")
    keys = ((("record", 0),),)
    grid = torch.randn(1, 4, 8)
    cache.store((("record", 0),), grid)

    history = cache.history(keys, reference=torch.randn(1, 4, 8))
    assert history.shape == (1, 1, 4, 8)
    torch.testing.assert_close(history[:, 0], grid)

    path = tmp_path / "history.pt"
    cache.save(path)
    restored = OnlineHistoryStateCache()
    restored.load(path)
    torch.testing.assert_close(
        restored.history(keys, reference=grid),
        history,
    )
