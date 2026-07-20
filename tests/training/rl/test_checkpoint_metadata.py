"""RL checkpoint protocol invariant tests."""

from __future__ import annotations

import pytest

from nimloth.training.rl.checkpoint import validate_rl_checkpoint_metadata
from nimloth.wm.state_proj import StateProjector


def _projector() -> StateProjector:
    return StateProjector(
        qwen_hidden_dim=16,
        lewm_emb_dim=8,
        projector_hidden_dim=12,
        latent_token_count=8,
    )


def test_resume_metadata_accepts_matching_k8_wm_value_protocol() -> None:
    validate_rl_checkpoint_metadata(
        {
            "latent_token_count": 8,
            "latent_query_mode": "inject",
            "qwen_hidden_dim": 16,
            "state_proj_input_dim": 128,
            "rollout_policy": "wm_value",
            "fast_path_horizon": 2,
            "predictor_rollout_steps": 2,
            "predictor_rollout_loss_decay": 1.0,
            "latent_query_token_ids": list(range(100, 108)),
        },
        state_proj=_projector(),
        latent_query_mode="inject",
        rollout_policy="wm_value",
        fast_path_horizon=2,
        predictor_rollout_steps=2,
        predictor_rollout_loss_decay=1.0,
        latent_query_token_ids=list(range(100, 108)),
    )


def test_resume_metadata_rejects_horizon_or_k_change() -> None:
    with pytest.raises(ValueError, match="latent_token_count mismatch"):
        validate_rl_checkpoint_metadata(
            {"latent_token_count": 1},
            state_proj=_projector(),
            latent_query_mode="inject",
            rollout_policy="wm_value",
            fast_path_horizon=2,
            predictor_rollout_steps=2,
            predictor_rollout_loss_decay=1.0,
        )
    with pytest.raises(ValueError, match="fast_path_horizon mismatch"):
        validate_rl_checkpoint_metadata(
            {"fast_path_horizon": 4},
            state_proj=_projector(),
            latent_query_mode="inject",
            rollout_policy="wm_value",
            fast_path_horizon=2,
            predictor_rollout_steps=2,
            predictor_rollout_loss_decay=1.0,
        )
