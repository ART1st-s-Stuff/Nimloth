from __future__ import annotations

import torch

from nimloth.wm.lewm import LeWMConfig
from nimloth.wm.predictor import LatentWMPredictor


def _make_predictor(history_size: int = 4, emb_dim: int = 64) -> LatentWMPredictor:
    cfg = LeWMConfig(history_size=history_size, emb_dim=emb_dim)
    return LatentWMPredictor.create(cfg)


def test_predict_next_emb_shape() -> None:
    """Single-step prediction returns correct shape."""
    emb_dim = 64
    B = 2
    predictor = _make_predictor(history_size=4, emb_dim=emb_dim)
    state = torch.randn(B, emb_dim)
    actions = torch.randint(0, 8, (B,))
    out = predictor.predict_next_emb(state, actions)
    assert out.shape == (B, emb_dim)


def test_predict_next_emb_equals_full_context_single_step() -> None:
    """For history_size=1, predict_next_emb and _predict_from_context are equivalent."""
    predictor = _make_predictor(history_size=1)
    state = torch.randn(4, 64)
    actions = torch.randint(0, 8, (4,))
    out1 = predictor.predict_next_emb(state, actions)
    out2 = predictor._predict_from_context(state.unsqueeze(1), actions.unsqueeze(1))
    assert torch.allclose(out1, out2, atol=1e-6)


def test_rollout_states_shape() -> None:
    """rollout_states returns (B, num_steps, emb_dim)."""
    B, num_steps, emb_dim = 2, 5, 64
    predictor = _make_predictor(history_size=4, emb_dim=emb_dim)
    state = torch.randn(B, emb_dim)
    action_seq = torch.randint(0, 8, (B, num_steps))
    out = predictor.rollout_states(state, action_seq)
    assert out.shape == (B, num_steps, emb_dim)


def test_rollout_states_single_step_eq_predict_next_emb() -> None:
    """With num_steps=1 and history_size=1, rollout_states matches predict_next_emb."""
    predictor = _make_predictor(history_size=1)
    state = torch.randn(4, 64)
    actions = torch.randint(0, 8, (4, 1))
    out_rollout = predictor.rollout_states(state, actions).squeeze(1)  # (B, emb_dim)
    out_single = predictor.predict_next_emb(state, actions.squeeze(1))
    assert torch.allclose(out_rollout, out_single, atol=1e-6)


def test_rollout_states_different_history_sizes() -> None:
    """rollout_states works for history_size 1, 2, 4."""
    B, num_steps = 3, 4
    for H in [1, 2, 4]:
        predictor = _make_predictor(history_size=H)
        state = torch.randn(B, 64)
        action_seq = torch.randint(0, 8, (B, num_steps))
        out = predictor.rollout_states(state, action_seq)
        assert out.shape == (B, num_steps, 64)


def test_rollout_states_deterministic() -> None:
    """Same inputs produce same outputs (no randomness in eval mode)."""
    predictor = _make_predictor(history_size=4).eval()
    state = torch.randn(1, 64)
    action_seq = torch.randint(0, 8, (1, 6))
    out1 = predictor.rollout_states(state, action_seq)
    out2 = predictor.rollout_states(state, action_seq)
    assert torch.allclose(out1, out2, atol=1e-6)
