from __future__ import annotations

import pytest
import torch

from nimloth.wm.lewm import LeWMConfig
from nimloth.wm.predictor import LatentWMPredictor


def _make_predictor(history_size: int = 4, emb_dim: int = 64) -> LatentWMPredictor:
    cfg = LeWMConfig(history_size=history_size, emb_dim=emb_dim)
    return LatentWMPredictor.create(cfg)


def _make_tokenized_predictor() -> LatentWMPredictor:
    return LatentWMPredictor.create(
        LeWMConfig(
            emb_dim=64,
            history_size=4,
            state_token_count=4,
            residual_prediction=True,
            predictor_hidden_dim=16,
            predictor_mlp_dim=32,
            predictor_depth=2,
            predictor_heads=2,
            predictor_dim_head=8,
        )
    )


def test_factorized_dynamics_preserve_external_state_shape_and_checkpoint(tmp_path) -> None:
    common = dict(
        history_size=1,
        emb_dim=64,
        predictor_hidden_dim=32,
        predictor_mlp_dim=64,
        predictor_heads=2,
    )
    factorized = LatentWMPredictor.create(LeWMConfig(**common, dynamics_dim=16))
    unfactorized = LatentWMPredictor.create(LeWMConfig(**common))
    state = torch.randn(2, 64)
    actions = torch.randint(0, 8, (2, 3))

    output = factorized.rollout_states(state, actions)
    mixed_dtype_output = factorized.predict_next_emb(
        state.to(torch.bfloat16), actions[:, 0]
    )

    assert output.shape == (2, 3, 64)
    assert mixed_dtype_output.shape == (2, 64)
    assert torch.isfinite(mixed_dtype_output).all()
    assert factorized.dynamics_dim == 16
    assert sum(p.numel() for p in factorized.parameters()) < sum(
        p.numel() for p in unfactorized.parameters()
    )
    factorized.save_checkpoint(tmp_path)
    reloaded = LatentWMPredictor.load_checkpoint(tmp_path)
    assert reloaded.config.dynamics_dim == 16
    reloaded.load_state_dict(factorized.state_dict(), strict=True)


def test_predict_next_emb_shape() -> None:
    """Single-step prediction returns correct shape for a T=1 architecture."""
    emb_dim = 64
    B = 2
    predictor = _make_predictor(history_size=1, emb_dim=emb_dim)
    state = torch.randn(B, emb_dim)
    actions = torch.randint(0, 8, (B,))
    out = predictor.predict_next_emb(state, actions)
    assert out.shape == (B, emb_dim)


def test_predict_next_emb_rejects_history_size_four() -> None:
    predictor = _make_predictor(history_size=4)
    with pytest.raises(ValueError, match="configured history_size=4"):
        predictor.predict_next_emb(torch.randn(2, 64), torch.zeros(2, dtype=torch.long))


def test_context_length_must_equal_configured_history_size() -> None:
    predictor = _make_predictor(history_size=4)
    with pytest.raises(ValueError, match="expected exactly T=4"):
        predictor._predict_from_context(
            torch.randn(2, 1, 64), torch.zeros(2, 1, dtype=torch.long)
        )


def test_predict_next_emb_equals_full_context_single_step() -> None:
    """For history_size=1, predict_next_emb and _predict_from_context are equivalent."""
    predictor = _make_predictor(history_size=1)
    state = torch.randn(4, 64)
    actions = torch.randint(0, 8, (4,))
    out1 = predictor.predict_next_emb(state, actions)
    out2 = predictor._predict_from_context(state.unsqueeze(1), actions.unsqueeze(1))
    assert torch.allclose(out1, out2, atol=1e-6)


def test_rollout_states_shape() -> None:
    """A T=1 architecture recursively returns all requested states."""
    B, num_steps, emb_dim = 2, 5, 64
    predictor = _make_predictor(history_size=1, emb_dim=emb_dim)
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


def test_rollout_states_rejects_unavailable_multi_step_history() -> None:
    predictor = _make_predictor(history_size=4)
    with pytest.raises(ValueError, match="requires a tokenized predictor"):
        predictor.rollout_states(
            torch.randn(3, 64), torch.randint(0, 8, (3, 4))
        )


def test_history_override_migrates_checkpoint_to_one_slot(tmp_path) -> None:
    predictor = _make_predictor(history_size=4)
    predictor.save_checkpoint(tmp_path)
    migrated = LatentWMPredictor.load_checkpoint(
        tmp_path, history_size_override=1
    )
    assert migrated.config.history_size == 1
    assert migrated.predictor.pos_embedding.shape == (1, 1, 64)
    torch.testing.assert_close(
        migrated.predictor.pos_embedding,
        predictor.predictor.pos_embedding[:, :1],
    )


def test_tokenized_t4_zero_delta_starts_as_identity() -> None:
    predictor = _make_tokenized_predictor()
    states = torch.randn(3, 4, 64)
    actions = torch.randint(0, 8, (3, 4))
    valid = torch.tensor([[False, False, False, True], [False, True, True, True], [True] * 4])
    output = predictor.predict_next_from_history(states, actions, valid)
    torch.testing.assert_close(output, states[:, -1])


def test_tokenized_t4_padding_values_are_masked() -> None:
    predictor = _make_tokenized_predictor().eval()
    torch.nn.init.normal_(predictor.tokenized_predictor.delta_head[-1].weight, std=0.01)
    states = torch.randn(2, 4, 64)
    actions = torch.randint(0, 8, (2, 4))
    valid = torch.tensor([[False, False, False, True], [False, False, True, True]])
    expected = predictor.predict_next_from_history(states, actions, valid)
    changed_states = states.clone()
    changed_actions = actions.clone()
    changed_states[~valid] = torch.randn_like(changed_states[~valid]) * 100
    changed_actions[~valid] = 7
    actual = predictor.predict_next_from_history(changed_states, changed_actions, valid)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_tokenized_t4_recursive_rollout_uses_masked_warmup() -> None:
    predictor = _make_tokenized_predictor().eval()
    initial = torch.randn(2, 64)
    actions = torch.randint(0, 8, (2, 5))
    output = predictor.rollout_states(initial, actions)
    assert output.shape == (2, 5, 64)
    torch.testing.assert_close(output, initial[:, None].expand_as(output))


def test_tokenized_t4_requires_last_slot_valid() -> None:
    predictor = _make_tokenized_predictor()
    with pytest.raises(ValueError, match="final history slot"):
        predictor.predict_next_from_history(
            torch.randn(1, 4, 64),
            torch.zeros(1, 4, dtype=torch.long),
            torch.tensor([[True, True, True, False]]),
        )


def test_rollout_states_deterministic() -> None:
    """Same inputs produce same outputs (no randomness in eval mode)."""
    predictor = _make_predictor(history_size=1).eval()
    state = torch.randn(1, 64)
    action_seq = torch.randint(0, 8, (1, 6))
    out1 = predictor.rollout_states(state, action_seq)
    out2 = predictor.rollout_states(state, action_seq)
    assert torch.allclose(out1, out2, atol=1e-6)
