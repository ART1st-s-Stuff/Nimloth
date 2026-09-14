"""Approved B prompt action-token rewrite, promoted from the immutable task script.

Source: .trellis/tasks/archive/2026-09/09-10-sft1-rollout2000/research/
prepare_b_prompt_data.py. Rules below are unchanged; query count is handled by
vagen_step60_data.convert_source_prompt before this rewrite.
"""

import re

NAMES = (
    "moveahead",
    "moveback",
    "moveright",
    "moveleft",
    "rotateright",
    "rotateleft",
    "lookup",
    "lookdown",
)
MEANINGS = (
    "move forward",
    "move backward",
    "move right",
    "move left",
    "rotate right",
    "rotate left",
    "tilt camera up",
    "tilt camera down",
)
TOKENS = tuple(f"<|action_({i})|>" for i in range(8))
OLD_LEGEND = (
    "Nimloth action indices: "
    + ", ".join(f"{i}={n}" for i, n in enumerate(NAMES))
    + "."
)
NEW_LEGEND = (
    "Nimloth action tokens: "
    + ", ".join(f"{t} = {n} ({m})" for t, n, m in zip(TOKENS, NAMES, MEANINGS))
    + "."
)
VERSION = "sft1_b_prompt_action_tokens_v1"
COUNTS = {"sft1_train_all.jsonl": 1709, "sft1_heldout_all.jsonl": 193}
PROVENANCE = "b_prompt_provenance"


def rewrite_text(text):
    for subject in ("answer", "action"):
        old = (
            f"The {subject} must be exactly one of these lowercase action names: "
            + ", ".join(NAMES)
            + "."
        )
        new = (
            f"The {subject} must be exactly one of these action tokens: "
            + ", ".join(TOKENS)
            + "."
        )
        text = text.replace(old, new)
    text = text.replace(
        "Format correct only when the answer is exactly one valid action name.",
        "Format correct only when the action block contains exactly one valid action token between <|action_start|> and <|action_end|>.",
    )
    text = text.replace(
        "Invalid action names receive negative feedback.",
        "Invalid action tokens receive negative feedback.",
    )
    text = text.replace(OLD_LEGEND, NEW_LEGEND)
    if re.search(
        r"lowercase action names|exactly one valid action name|Invalid action names|Nimloth action indices:",
        text,
    ):
        raise ValueError("Unconverted action-name output instruction")
    return text

