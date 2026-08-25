"""Tests for the upstream VAGEN K16 Nimloth navigation protocol."""

from __future__ import annotations

import sys
from pathlib import Path

_VAGEN_ROOT = Path(__file__).resolve().parents[1] / "external" / "VAGEN"
if _VAGEN_ROOT.is_dir() and str(_VAGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_VAGEN_ROOT))

from vagen.envs.navigation.utils.nimloth_format import latent_state_tokens
from vagen.envs.navigation.utils.parse import parse_response
from vagen.envs.navigation.utils.prompt import system_prompt


def _response(action_index: int) -> str:
    return (
        "<think>Use one real action and reassess.</think>"
        + "".join(latent_state_tokens(16))
        + f"<|action_start|><|action_({action_index})|><|action_end|>"
    )


def test_nimloth_prompt_has_exact_k16_block_before_action_start() -> None:
    text = system_prompt(
        format_name="nimloth",
        max_actions_per_step=1,
        example_count=0,
        latent_token_count=16,
    )
    block = "".join(latent_state_tokens(16))
    assert block + "<|action_start|>" in text
    assert "Choose exactly one valid action" in text
    assert "take multiple actions" not in text


def test_parse_nimloth_k16_success() -> None:
    parsed = parse_response(
        _response(3),
        prompt_format="nimloth",
        max_actions=1,
        latent_token_count=16,
    )
    assert parsed["format_correct"] is True
    assert parsed["actions"] == ["move_left"]


def test_parse_nimloth_rejects_k1_or_old_wm_envelope() -> None:
    k1 = (
        "<think>Turn left.</think>"
        "<|latent_state|><|action_start|><|action_(3)|><|action_end|>"
    )
    old_wm = _response(3) + "<prediction>future</prediction>"
    for response in (k1, old_wm):
        parsed = parse_response(
            response,
            prompt_format="nimloth",
            max_actions=1,
            latent_token_count=16,
        )
        assert parsed["format_correct"] is False
        assert parsed["actions"] == []


def test_parse_nimloth_rejects_latent_after_action_start() -> None:
    response = (
        "<think>Turn left.</think><|action_start|>"
        + "".join(latent_state_tokens(16))
        + "<|action_(3)|><|action_end|>"
    )
    parsed = parse_response(
        response,
        prompt_format="nimloth",
        max_actions=1,
        latent_token_count=16,
    )
    assert parsed["format_correct"] is False
    assert parsed["actions"] == []
