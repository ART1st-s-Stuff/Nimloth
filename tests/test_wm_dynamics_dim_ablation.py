from __future__ import annotations

from pathlib import Path

import torch

from nimloth.wm.dynamics_dim_heads import (
    DynamicsDimHeadSpec,
    DynamicsDimWMHeads,
    parameter_counts_meta,
)


def tiny_spec() -> DynamicsDimHeadSpec:
    return DynamicsDimHeadSpec(
        external_dim=32,
        full_dynamics_dim=32,
        factorized_dynamics_dim=8,
        predictor_hidden_dim=8,
        predictor_depth=1,
        predictor_heads=2,
        predictor_mlp_dim=16,
        history_size=1,
    )


def test_dynamics_dim_heads_predict_rollout_and_reload(tmp_path: Path) -> None:
    heads = DynamicsDimWMHeads.create(tiny_spec())
    state = torch.randn(3, 32)
    actions = torch.tensor([4, 5, 0])
    sequences = torch.tensor([[4, 0, 5], [5, 0, 4], [0, 4, 5]])

    one = heads.predict_next(state, actions)
    rollout = heads.rollout(state, sequences)
    heads.save_checkpoint(tmp_path)
    loaded = DynamicsDimWMHeads.load_checkpoint(tmp_path)

    assert all(value.shape == (3, 32) for value in one)
    assert all(value.shape == (3, 3, 32) for value in rollout)
    assert all(torch.isfinite(value).all() for value in (*one, *rollout))
    assert loaded.spec == heads.spec


def test_production_parameter_counts_are_explicitly_unmatched() -> None:
    counts = parameter_counts_meta(DynamicsDimHeadSpec())

    assert counts == {"full": 408_321_096, "factorized": 160_642_120}
    assert counts["full"] / counts["factorized"] > 2.5
