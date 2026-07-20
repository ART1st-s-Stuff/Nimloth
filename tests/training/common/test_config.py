from __future__ import annotations

from pathlib import Path

from nimloth.training.common.config import load_yaml_config


def test_load_yaml_config_returns_nested_mapping() -> None:
    path = Path(__file__).resolve().parents[3] / "configs" / "training" / "sft2" / "latent_wm_value.yaml"
    cfg = load_yaml_config(path)
    assert cfg["tuning"]["llm_tune"] == "freeze"
    assert cfg["tuning"]["vision_tune"] == "full"
    assert cfg["data"]["include_failed_rollouts"] is True
    assert cfg["monitor"]["wandb"] is True
