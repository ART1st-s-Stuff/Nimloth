from __future__ import annotations

from copy import deepcopy

import pytest

from nimloth.training.sft1.config import (
    SFT1_V2_CONFIG_SCHEMA,
    STATE_INTERFACE_OBJECTIVE_VERSION,
    parse_sft1_v2_config,
)


def _raw_config() -> dict[str, object]:
    return {
        "schema": SFT1_V2_CONFIG_SCHEMA,
        "state": {
            "objective_version": STATE_INTERFACE_OBJECTIVE_VERSION,
            "latent_query_mode": "inject",
            "query_tune": "adapter",
            "grid_tokens": 16,
            "qwen_hidden_dim": 2048,
            "state_dim": 1024,
            "projector_hidden_dim": 2048,
            "instruction_teacher_dim": 2048,
            "action_dim": 8,
            "movement_action_indices": [0, 2, 3],
        },
        "objective": {
            "visual_weight": 1.0,
            "visual_relation_coefficient": 0.5,
            "instruction_weight": 1.0,
            "instruction_contrastive_coefficient": 0.5,
            "observed_feasibility_weight": 1.0,
            "actor_preservation_weight": 1.0,
            "state_policy_weight": 1.0,
            "policy_temperature": 1.0,
            "contrastive_temperature": 0.1,
        },
        "freeze": {
            "freeze_qwen_language_body": True,
            "freeze_vision_tower": True,
            "freeze_dino_teacher": True,
            "freeze_instruction_action_teacher": True,
            "train_query_adapter": True,
            "train_fresh_projector": True,
            "train_readouts": True,
        },
        "optimizer": {"learning_rate": 1e-4, "weight_decay": 0.0},
    }


def test_v2_config_is_strict_and_identity_bound() -> None:
    config = parse_sft1_v2_config(_raw_config())
    assert config.state.movement_action_indices == (0, 2, 3)
    assert config.state.qwen_hidden_dim == 2048

    invalid_cases = []
    unknown = deepcopy(_raw_config())
    unknown["state"]["unknown"] = 1  # type: ignore[index]
    invalid_cases.append((unknown, "unknown SFT1-v2 config field"))
    wrong_actions = deepcopy(_raw_config())
    wrong_actions["state"]["movement_action_indices"] = [4, 5, 6]  # type: ignore[index]
    invalid_cases.append((wrong_actions, "move_forward/move_right/move_left"))
    wrong_hidden = deepcopy(_raw_config())
    wrong_hidden["state"]["qwen_hidden_dim"] = 1024  # type: ignore[index]
    invalid_cases.append((wrong_hidden, "2048-dimensional"))
    legacy = deepcopy(_raw_config())
    legacy["schema"] = "legacy_sft1"
    invalid_cases.append((legacy, "legacy state config cannot resume"))

    for raw, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            parse_sft1_v2_config(raw)
