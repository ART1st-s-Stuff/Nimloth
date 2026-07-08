from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONVERTER_PATH = _REPO_ROOT / "experiments" / "training" / "sft1" / "convert_rollouts.py"

spec = importlib.util.spec_from_file_location("convert_rollouts", _CONVERTER_PATH)
convert_rollouts = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = convert_rollouts
spec.loader.exec_module(convert_rollouts)


def test_convert_assistant_preserves_legacy_answer_multi_action() -> None:
    converted, actions, think = convert_rollouts.convert_assistant(
        "<think>go twice</think><answer>moveahead, moveleft</answer>"
    )

    assert think == "go twice"
    assert actions == ["move_forward", "move_left"]
    assert "<|action_(0)|><|action_(3)|>" in converted


def test_convert_assistant_preserves_multi_nimloth_tokens() -> None:
    converted, actions, _ = convert_rollouts.convert_assistant(
        "<think>go twice</think>"
        "<|latent_state|><|action_start|><|action_(0)|><|action_(0)|><|action_end|>"
    )

    assert actions == ["move_forward", "move_forward"]
    assert converted.count("<|action_(0)|>") == 2


def test_convert_one_records_action_groups(tmp_path: Path) -> None:
    output_str = (
        "<|im_start|>system\n"
        "Respond in this format:\n<think>...</think><answer>some_action</answer>"
        "<|im_end|>"
        "<|im_start|>user\nDecide your next action.<|im_end|>"
        "<|im_start|>assistant\n"
        "<think>go twice</think><answer>moveahead, moveleft</answer>"
        "<|im_end|>"
    )
    src = convert_rollouts.SourceRecord(
        split="train",
        shard="shard_000",
        jsonl_path=tmp_path / "50.jsonl",
        line_index=0,
        payload={"output_str": output_str, "step": 50, "traj_success": 1.0},
    )

    rec = convert_rollouts.convert_one(src, target_max_actions_per_step=5)

    assert rec["actions"] == ["move_forward", "move_left"]
    assert rec["action_groups"] == [["move_forward", "move_left"]]
    assert rec["action_indices_by_turn"] == [[0, 3]]
    assert rec["validation_issues"] == []
    assert "<answer>" not in rec["messages"][0]["content"]
    assert "[<|action_(idx)|>...]" in rec["messages"][0]["content"]
