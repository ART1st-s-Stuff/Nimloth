from __future__ import annotations

from experiments.training.sft1.convert_rollouts import (
    NIMLOTH_ACTION_BLOCK,
    rewrite_prompt_instruction,
)


def test_rewrite_preserves_canonical_prefix_without_duplicate_requirement() -> None:
    source = (
        "You must take exactly one action in each response. "
        "You can optionally think first, then give your action. Respond in this format:\n"
        "<think>...</think><action>some_action</action>"
    )

    rewritten = rewrite_prompt_instruction(source)

    assert rewritten.count("You must take exactly one action in each response.") == 1
    assert rewritten.startswith(
        "You must take exactly one action in each response. "
        "You can optionally think first, then give your action. Respond in this format:\n"
    )
    assert NIMLOTH_ACTION_BLOCK in rewritten
    assert "<action>" not in rewritten
