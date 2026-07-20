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
    runner = (
        ROOT / "experiments/training/rl/run_k8_wm_fastpath_smoke.sh"
    ).read_text()
    assert '"model.layers.0" in key' in runner
    assert 'key.startswith("visual.")' in runner


def test_encode_trajectory_hiddens_passes_k_to_qwen_batch(monkeypatch) -> None:
    import torch

    from nimloth.latent import extraction
    from nimloth.training.common import qwen_batch
    from nimloth.training.rl import trainer
    from nimloth.training.rl.rollout import RolloutTrajectory

    observed: list[int | None] = []

    def fake_build(items, processor, max_length, *, latent_token_count=None, **kwargs):
        observed.append(latent_token_count)
        return {"input_ids": torch.arange(8).unsqueeze(0)}

    monkeypatch.setattr(qwen_batch, "build_qwen_batch", fake_build)
    monkeypatch.setattr(
        extraction,
        "find_last_latent_state_block",
        lambda *args, **kwargs: tuple(range(8)),
    )
    def model(**kwargs):
        return SimpleNamespace(hidden_states=(torch.zeros(1, 8, 4),))
    trajectory = RolloutTrajectory(
        record_id="k8",
        image_paths=["0.png", "1.png"],
        action_indices=[0],
        action_names=["moveahead"],
        nav_instruction="Find it.",
        latent_token_count=8,
        latent_query_mode="inject",
    )

    states = trainer.encode_trajectory_hiddens(
        trajectory,
        qwen_model=model,
        processor=object(),
        token_id_map={},
        device=torch.device("cpu"),
        latent_token_count=8,
    )

    assert len(states) == 2
    assert observed == [8, 8]


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
