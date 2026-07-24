from __future__ import annotations

import torch

from nimloth.training.sft2.data.terminal_cot import _StopAfterText


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
