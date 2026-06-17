from __future__ import annotations

from pathlib import Path

from nimloth.training.common.config import flatten_yaml_config, load_yaml_config


def test_flatten_yaml_config_maps_tuning_and_data() -> None:
    cfg = {
        "tuning": {"llm_tune": "freeze", "vision_tune": "full", "vision_ema": True},
        "data": {"include_failed_rollouts": False},
        "monitor": {"wandb": False},
    }
    flat = flatten_yaml_config(cfg)
    assert flat["llm_tune"] == "freeze"
    assert flat["vision_tune"] == "full"
    assert flat["vision_ema"] is True
    assert flat["success_only"] is True
    assert flat["no_wandb"] is True


def test_load_default_sft2_config() -> None:
    path = Path(__file__).resolve().parents[3] / "configs" / "training" / "sft2" / "latent_wm_value.yaml"
    cfg = load_yaml_config(path)
    flat = flatten_yaml_config(cfg)
    assert flat["llm_tune"] == "freeze"
    assert flat["vision_tune"] == "full"
    assert flat["success_only"] is False
    assert flat["no_wandb"] is False
