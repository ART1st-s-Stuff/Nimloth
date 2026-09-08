"""结构诊断不泄露文本，且区分完整输出与条件动作输出。"""

import importlib.util
from pathlib import Path

FILE = (
    Path(__file__).resolve().parents[4]
    / "experiments/training/sft/evaluation/diagnose_stage1_format.py"
)
spec = importlib.util.spec_from_file_location("format_diagnostic", FILE)
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def test_format_structure_and_failure():
    from nimloth.latent import latent_state_block

    valid = (
        "<think>private source</think>"
        + latent_state_block(1)
        + "<|action_start|><|action_(0)|><|action_end|>"
    )
    result = diag.classify(valid)
    assert result["full_format_pass"] and result["all_markers_ordered"]
    assert "private" not in str(result)
    action_only = diag.classify("<|action_start|><|action_(0)|><|action_end|>")
    assert action_only["action_format_pass"] and not action_only["full_format_pass"]
    truncated = diag.classify(valid.replace("<|action_end|>", ""))
    assert not truncated["full_format_pass"]
    assert truncated["marker_counts"][-1] == 0


def test_termination_eos_and_limit_are_independent():
    assert diag.termination([1, 2], [2, 3], 2) == {
        "generated_tokens": 2,
        "ended_with_eos": True,
        "reached_token_limit": True,
    }
    assert not diag.termination([], 2, 128)["ended_with_eos"]
    assert not diag.termination([1], 2, 128)["reached_token_limit"]
