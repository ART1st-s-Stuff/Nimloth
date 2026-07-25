from __future__ import annotations

import torch

from nimloth.training.sft2.data.terminal_cot import (
    _CONTINUATION_PREVIEW_CHARS,
    _StopAfterText,
    TerminalCoTFormatError,
    _missing_close_error,
)


class _BoundaryMergingTokenizer:
    _pieces = {
        1: "Move",
        2: " left",
        3: ".</",
        4: "think",
        5: ">",
    }

    def decode(self, token_ids, **_kwargs) -> str:
        return "".join(self._pieces[token_id] for token_id in token_ids)


def test_terminal_stop_detects_close_text_across_merged_token_boundary() -> None:
    stopping = _StopAfterText(
        _BoundaryMergingTokenizer(),
        start_length=1,
        text="</think>",
    )

    assert not stopping(torch.tensor([[0, 1, 2, 3, 4]]), torch.empty(0))
    assert stopping(torch.tensor([[0, 1, 2, 3, 4, 5]]), torch.empty(0))


def test_missing_terminal_close_reports_bounded_continuation_preview() -> None:
    continuation = "x" * (_CONTINUATION_PREVIEW_CHARS + 10)

    error = _missing_close_error(
        record_id="train/example",
        max_reasoning_tokens=128,
        continuation_ids=tuple(range(131)),
        decoded_continuation=continuation,
    )

    message = str(error)
    assert isinstance(error, TerminalCoTFormatError)
    assert "record 'train/example'" in message
    assert "generated_tokens=131" in message
    assert "x" * _CONTINUATION_PREVIEW_CHARS in message
    assert "x" * (_CONTINUATION_PREVIEW_CHARS + 1) not in message
