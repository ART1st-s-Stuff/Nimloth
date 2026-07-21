"""阶段无关 Agent 与 rollout 配置的边界测试。"""

from __future__ import annotations

import pytest

from nimloth.config.agent import parse_agent_config
from nimloth.config.rollout import parse_rollout_config


def test_agent_config_builds_persistable_prompt_spec() -> None:
    config = parse_agent_config(
        {
            "prompt_template": "nimloth-latent-action",
            "thought": "Choose the safest next action.",
        }
    )

    spec = config.prompt_spec(latent_token_count=8)
    assert spec.identifier == "nimloth-latent-action"
    assert spec.config == {
        "latent_token_count": 8,
        "thought": "Choose the safest next action.",
    }


def test_common_configs_reject_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown agent config field"):
        parse_agent_config({"environment_instruction": "moveahead"})
    with pytest.raises(ValueError, match="unknown rollout config field"):
        parse_rollout_config({"trainer_batch_size": 4})


def test_rollout_config_is_independent_of_training_phase() -> None:
    config = parse_rollout_config(
        {
            "train_datasets": ["base_train"],
            "temperature": 0.7,
            "top_p": 0.95,
        }
    )

    assert config.train_datasets == ("base_train",)
    assert config.temperature == 0.7
    assert config.top_p == 0.95
