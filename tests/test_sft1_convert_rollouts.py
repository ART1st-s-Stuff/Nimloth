"""Tests for source-format SFT1 rollout conversion."""

import importlib.util
from pathlib import Path
import sys


_SCRIPT = Path(__file__).resolve().parents[1] / "experiments/training/sft1/convert_rollouts.py"
_SPEC = importlib.util.spec_from_file_location("sft1_convert_rollouts", _SCRIPT)
assert _SPEC and _SPEC.loader
convert = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = convert
_SPEC.loader.exec_module(convert)


def _source_record() -> object:
    transcript = (
        "<|im_start|>system\nThe response format is "
        "<think><observation>...</observation><reasoning>...</reasoning>"
        "<prediction>...</prediction></think><answer>...</answer>. "
        "Example: <answer>moveleft</answer><|im_end|>\n"
        "<|im_start|>user\n<image>\nChoose one action: <answer>...</answer><|im_end|>\n"
        "<|im_start|>assistant\n<think><observation>room</observation>"
        "<reasoning>advance</reasoning><prediction>closer</prediction></think>"
        "<answer>moveahead</answer><|im_end|>\n"
        "<|im_start|>user\n<image>\nContinue.<|im_end|>\n"
        "<|im_start|>assistant\n<think><observation>wall</observation>"
        "<reasoning>turn</reasoning><prediction>new view</prediction></think>"
        "<answer>rotateright</answer><|im_end|>\n"
        "<|im_start|>user\n<image>\nContinue.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return convert.SourceRecord(
        split="train",
        shard="shard_smoke",
        jsonl_path=Path("/tmp/shard_smoke/0.jsonl"),
        line_index=0,
        rollout_step=0,
        payload={
            "output_str": transcript,
            "image_paths": ["/tmp/0.png", "/tmp/1.png", "/tmp/2.png"],
            "metrics": {"success": 0.0, "score": 1.0},
        },
    )


def test_answer_source_profile_converts_to_nimloth_actions() -> None:
    record = convert.convert_one(_source_record(), source_action_tag="answer")

    assert record["actions"] == ["move_forward", "turn_right"]
    assert record["action_indices"] == [0, 4]
    assert record["validation_issues"] == []
    assert len(record["messages"]) == 6
    for message in record["messages"]:
        assert "<answer>" not in message["content"]
        assert "</answer>" not in message["content"]
    assistants = [m for m in record["messages"] if m["role"] == "assistant"]
    assert all("<|latent_state|>" in m["content"] for m in assistants)


def test_default_action_profile_does_not_silently_accept_answer_xml() -> None:
    record = convert.convert_one(_source_record(), source_action_tag="action")

    assert record["actions"] == []
    assert "no_parsed_actions" in record["validation_issues"]
