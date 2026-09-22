"""Qwen chat-template transcript parsing shared by offline source converters."""

from __future__ import annotations

import re

IM_END = "<|im_end|>"

def parse_im_messages(text: str) -> list[dict[str, str]]:
    """Parse a Qwen chat-template string into role/content messages."""
    messages: list[dict[str, str]] = []
    pos = 0
    token_re = re.compile(r"<\|im_start\|>(system|user|assistant)\n", re.S)
    while True:
        m = token_re.search(text, pos)
        if not m:
            break
        role = m.group(1)
        content_start = m.end()
        end = text.find(IM_END, content_start)
        if end < 0:
            content = text[content_start:]
            pos = len(text)
        else:
            content = text[content_start:end]
            pos = end + len(IM_END)
        messages.append({"role": role, "content": content})
    return messages

