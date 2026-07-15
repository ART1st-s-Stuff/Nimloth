from __future__ import annotations

from pathlib import Path

import torch

from nimloth.training.reconstruction.query_bottleneck_probe import (
    QueryBottleneckAdapter,
)
from nimloth.wm.frozen_query_state import FrozenQueryStateEncoder, StateViews
from nimloth.wm.matched_heads import MatchedHeadSpec, MatchedWMHeads


def probe_checkpoint(path: Path) -> None:
    model = QueryBottleneckAdapter(
        input_tokens=8,
        input_dim=32,
        bottleneck_dim=16,
    )
    torch.save(
        {
            "model": model.state_dict(),
            "step": 7,
            "invariants": {
                "input_shape": [8, 32],
                "bottleneck_shape": [8, 16],
            },
        },
        path,
    )


def test_frozen_encoder_emits_exact_token_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "probe.pt"
    probe_checkpoint(checkpoint)
    encoder = FrozenQueryStateEncoder.from_probe_checkpoint(checkpoint)

    state = encoder(torch.randn(3, 8, 32))

    assert state.shape == (3, 8, 16)
    assert torch.isfinite(state).all()
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert encoder.source_step == 7


def test_state_views_share_exact_scalar_content() -> None:
    tokens = torch.randn(2, 8, 16)

    views = StateViews.from_tokens(tokens)

    assert views.tokens.shape == (2, 8, 16)
    assert views.vector.shape == (2, 1, 128)
    assert torch.equal(views.vector.reshape_as(tokens), tokens)
    assert views.vector.untyped_storage().data_ptr() == tokens.untyped_storage().data_ptr()


def test_matched_heads_predict_rollout_and_reload(tmp_path: Path) -> None:
    spec = MatchedHeadSpec(
        state_tokens=8,
        token_dim=16,
        vector_hidden_dim=12,
        token_hidden_dim=16,
        depth=1,
        heads=4,
        mlp_ratio=2,
    )
    heads = MatchedWMHeads.create(spec)
    state = StateViews.from_tokens(torch.randn(2, 8, 16))
    actions = torch.tensor([4, 5])
    sequence = torch.tensor([[4, 0, 5], [5, 0, 4]])

    vector_next, token_next = heads.predict_next(state, actions)
    vector_rollout, token_rollout = heads.rollout(state, sequence)
    heads.save_checkpoint(tmp_path)
    reloaded = MatchedWMHeads.load_checkpoint(tmp_path)

    assert vector_next.shape == (2, 1, 128)
    assert token_next.shape == (2, 8, 16)
    assert vector_rollout.shape == (2, 3, 1, 128)
    assert token_rollout.shape == (2, 3, 8, 16)
    assert all(torch.isfinite(item).all() for item in (vector_next, token_next))
    assert reloaded.spec == spec
