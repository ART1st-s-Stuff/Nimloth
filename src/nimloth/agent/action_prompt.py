"""Stage 1 action protocol shared by training and environment evaluation."""
from __future__ import annotations

import copy
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
VERSION = "sft1_action_tokens_v1"
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


def rewrite_content(content):
    """Only textual content changes; image paths, URLs and metadata are opaque."""
    if isinstance(content, str):
        return rewrite_text(content)
    if isinstance(content, list):
        return [rewrite_content(item) for item in content]
    if isinstance(content, dict):
        result = copy.deepcopy(content)
        if result.get("type") in ("text", "input_text") and isinstance(
            result.get("text"), str
        ):
            result["text"] = rewrite_text(result["text"])
        return result
    return copy.deepcopy(content)



def format_action_prompt(text: str, *, latent_token_count: int | None = None) -> str:
    """Convert source navigation instructions without changing observed content."""
    if not isinstance(text, str):
        raise TypeError("Prompt must be text")
    from nimloth.latent.extraction import latent_state_tokens
    query_block = "".join(latent_state_tokens(latent_token_count)) if latent_token_count is not None else ""
    text = re.sub(r"<\|latent_state(?:_\d+)?\|>", "", text)
    had_answer = "<answer>" in text or "</answer>" in text
    def envelope(match):
        action = match.group(1).strip().lower()
        token = TOKENS[NAMES.index(action)] if action in NAMES else "<|action_(idx)|>"
        return query_block + "<|action_start|>" + token + "<|action_end|>"
    text = re.sub(r"<answer>(.*?)</answer>", envelope, text, flags=re.DOTALL)
    text = text.replace("inside <answer>", "between <|action_start|> and <|action_end|>")
    text = text.replace("after </answer>", "after <|action_end|>")
    text = text.replace("<answer>", "<|action_start|>").replace("</answer>", "<|action_end|>")
    if had_answer and OLD_LEGEND not in text and NEW_LEGEND not in text:
        text += "\n" + OLD_LEGEND
    return rewrite_text(text)
