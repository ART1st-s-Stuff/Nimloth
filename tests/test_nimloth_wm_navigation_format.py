"""Tests for VAGEN navigation prompt_format=nimloth_wm."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_VAGEN_ROOT = Path(__file__).resolve().parents[1] / "external" / "VAGEN"
if _VAGEN_ROOT.is_dir() and str(_VAGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_VAGEN_ROOT))


def _load_module(relpath: str, name: str):
    path = _VAGEN_ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


nimloth_format = _load_module(
    "vagen/envs/navigation/utils/nimloth_format.py",
    "vagen.envs.navigation.utils.nimloth_format",
)
prompt = _load_module(
    "vagen/envs/navigation/utils/prompt.py",
    "vagen.envs.navigation.utils.prompt",
)
parse = _load_module(
    "vagen/envs/navigation/utils/parse.py",
    "vagen.envs.navigation.utils.parse",
)


def test_nimloth_wm_prompt_has_latent_before_action_start() -> None:
    text = prompt.system_prompt(format_name="nimloth_wm", max_actions_per_step=1, example_count=0)
    latent_pos = text.index("<|latent_state|>")
    action_start_pos = text.index("<|action_start|>")
    assert latent_pos < action_start_pos
    assert "<observation>" in text
    assert "<prediction>" in text
    assert "<action>" not in text


def test_nimloth_prompt_keeps_latent_before_action_start() -> None:
    text = prompt.system_prompt(format_name="nimloth", max_actions_per_step=1, example_count=0)
    body = nimloth_format.NIMLOTH_FORMAT_BODY
    assert body.index("<|latent_state|>") < body.index("<|action_start|>")


def test_parse_nimloth_wm_success() -> None:
    response = (
        "<observation>Garbage can on the left.</observation>"
        "<think>Turn left first.</think>"
        "<|latent_state|><|action_start|><|action_(3)|><|action_end|>"
        "<prediction>The can will appear closer on the left.</prediction>"
    )
    parsed = parse.parse_response(response, prompt_format="nimloth_wm", max_actions=1)
    assert parsed["format_correct"] is True
    assert parsed["actions"] == ["move_left"]
    assert parsed["observation"] == "Garbage can on the left."
    assert parsed["prediction"].startswith("The can")


def test_parse_nimloth_wm_rejects_latent_after_action_start() -> None:
    response = (
        "<observation>Garbage can on the left.</observation>"
        "<think>Turn left first.</think>"
        "<|action_start|><|latent_state|><|action_(3)|><|action_end|>"
        "<prediction>The can will appear closer on the left.</prediction>"
    )
    parsed = parse.parse_response(response, prompt_format="nimloth_wm", max_actions=1)
    assert parsed["format_correct"] is False
    assert parsed["actions"] == []


def test_parse_nimloth_requires_latent_before_action_start() -> None:
    good = (
        "<think>Turn left first.</think>"
        "<|latent_state|><|action_start|><|action_(3)|><|action_end|>"
    )
    bad = (
        "<think>Turn left first.</think>"
        "<|action_start|><|action_(3)|><|action_end|>"
    )
    assert parse.parse_response(good, prompt_format="nimloth", max_actions=1)["format_correct"] is True
    assert parse.parse_response(bad, prompt_format="nimloth", max_actions=1)["format_correct"] is False
