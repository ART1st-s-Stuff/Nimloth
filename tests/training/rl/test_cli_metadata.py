"""RL model/checkpoint metadata resolution tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from nimloth.training.rl.cli import load_rl_config, resolve_rl_init_metadata


ROOT = Path(__file__).resolve().parents[3]


def test_fresh_k8_inject_metadata_comes_from_model_config() -> None:
    protocol, hidden_dim = resolve_rl_init_metadata(SimpleNamespace(
        hidden_size=2048,
        nimloth_latent_token_count=8,
        nimloth_latent_query_mode="inject",
    ))
    assert protocol.latent_token_count == 8
    assert protocol.latent_query_mode == "inject"
    assert hidden_dim == 2048


def test_lora_resume_can_take_protocol_from_rl_state() -> None:
    protocol, hidden_dim = resolve_rl_init_metadata(
        SimpleNamespace(hidden_size=2048),
        {
            "latent_token_count": 8,
            "latent_query_mode": "inject",
            "qwen_hidden_dim": 2048,
        },
    )
    assert protocol.latent_token_count == 8
    assert hidden_dim == 2048


def test_k8_wm_fastpath_config_trains_qwen_actor_and_wm_heads() -> None:
    config = load_rl_config(ROOT / "configs/training/rl/k8_wm_fastpath.yaml")
    assert config["freeze"] == {"qwen": False, "state_proj": True}
    assert config["actor"]["clip_ratio"] == 0.2
    assert config["rollout"]["policy"] == "qwen_wm"
    assert config["rollout"]["fast_path_horizon"] == 2
    assert config["predictor"]["rollout_steps"] == 2


def test_k8_wm_fastpath_smoke_config_covers_two_step_windows() -> None:
    config = load_rl_config(ROOT / "configs/training/rl/k8_wm_fastpath_smoke.yaml")
    assert config["freeze"] == {"qwen": False, "state_proj": True}
    assert config["actor"]["clip_ratio"] == 0.2
    assert config["rollout"] == {
        "policy": "qwen_wm",
        "fast_path_horizon": 2,
        "eval_sets": ["base_train"],
    }
    assert config["predictor"]["rollout_steps"] == 2
    assert config["rl"]["batch_size"] == 8
    assert config["training"]["save_interval"] == 1


def test_resume_rejects_model_state_protocol_mismatch() -> None:
    with pytest.raises(ValueError, match="model/state protocol mismatch"):
        resolve_rl_init_metadata(
            SimpleNamespace(
                hidden_size=2048,
                nimloth_latent_token_count=1,
                nimloth_latent_query_mode="inject",
            ),
            {
                "latent_token_count": 8,
                "latent_query_mode": "inject",
                "qwen_hidden_dim": 2048,
            },
        )
