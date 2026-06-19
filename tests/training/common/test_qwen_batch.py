from __future__ import annotations

from nimloth.training.common.qwen_batch import assistant_char_spans


class FakeProcessor:
    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
        assert tokenize is False
        rendered = ""
        for msg in messages:
            if msg["role"] == "assistant":
                rendered += "<assistant>" + msg["content"]
            else:
                rendered += f"<{msg['role']}>" + msg["content"]
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


def test_assistant_char_spans_only_returns_final_assistant_span() -> None:
    messages = [
        {"role": "user", "content": "obs0"},
        {"role": "assistant", "content": "act0"},
        {"role": "user", "content": "obs1"},
        {"role": "assistant", "content": "act1"},
    ]

    spans = assistant_char_spans(messages, FakeProcessor())

    assert len(spans) == 1
    start, end = spans[0]
    rendered = FakeProcessor().apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    assert rendered[start:end] == "act1"


def test_assistant_char_spans_empty_without_assistant() -> None:
    assert assistant_char_spans([{"role": "user", "content": "obs"}], FakeProcessor()) == []
