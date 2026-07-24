"""通用 Qwen input builder 与 SFT2 batch assembler 的边界测试。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from nimloth.backbone import BackboneBatch
from nimloth.backbone.qwen25vl.input import Qwen25VLInputBuilder
from nimloth.training.sft2.batch import CachedNextBatch, SFT2BatchAssembler


def _assembler() -> SFT2BatchAssembler:
    processor = MagicMock()
    processor.tokenizer.pad_token_id = 0
    return SFT2BatchAssembler(
        input_builder=Qwen25VLInputBuilder(
            processor=processor,
            max_length=32,
        ),
        device=torch.device("cpu"),
        history_size=1,
    )


def _item(identifier: str, next_messages):
    return {
        "id": identifier,
        "record_id": "record",
        "step_index": 0,
        "messages": [{"role": "user", "content": identifier}],
        "next_messages": next_messages,
        "action_index": 0,
        "action_value_target": 1.0,
        "success": True,
    }


def test_prepare_deduplicates_identical_next_prompts() -> None:
    shared_next = [{"role": "user", "content": "shared"}]
    batch_sizes: list[int] = []

    def fake_build(items, _processor, _max_length, **_kwargs):
        batch_sizes.append(len(items))
        return {"input_ids": torch.zeros((len(items), 4), dtype=torch.long)}

    with patch(
        "nimloth.backbone.qwen25vl.input.build_qwen_batch",
        side_effect=fake_build,
    ):
        batch = _assembler().prepare(
            [_item("a", shared_next), _item("b", shared_next)]
        )

    assert batch_sizes == [2, 2, 1]
    assert batch.online_tail.tensors["input_ids"].shape[0] == 2
    assert batch.next.tensors["input_ids"].shape[0] == 1
    assert batch.next_indices.tolist() == [[0], [0]]
    assert batch.transitions.non_terminal_mask.tolist() == [True, True]


def test_prepare_uses_worker_prebatched_next_cache() -> None:
    next_messages = [{"role": "user", "content": "next"}]
    assembler = _assembler()
    cached = CachedNextBatch(
        keys=(assembler.input_builder.cache_key(next_messages, ()),),
        batch=BackboneBatch(
            {"input_ids": torch.ones((1, 3), dtype=torch.long)}
        ),
    )
    raw = {
        "items": [_item("a", next_messages)],
        "current": BackboneBatch(
            {"input_ids": torch.ones((1, 2), dtype=torch.long)}
        ),
        "online_tail": BackboneBatch(
            {"input_ids": torch.ones((1, 3), dtype=torch.long)}
        ),
        "next": cached,
    }

    with patch("nimloth.backbone.qwen25vl.input.build_qwen_batch") as build:
        batch = assembler.prepare(raw)

    build.assert_not_called()
    assert torch.equal(
        batch.next.tensors["input_ids"],
        torch.ones((1, 3), dtype=torch.long),
    )


def test_window_without_final_real_state_is_rejected() -> None:
    seen: list[list[dict]] = []

    def fake_build(items, _processor, _max_length, **_kwargs):
        seen.append(items)
        return {"input_ids": torch.ones((len(items), 2), dtype=torch.long)}

    with patch(
        "nimloth.backbone.qwen25vl.input.build_qwen_batch",
        side_effect=fake_build,
    ):
        with pytest.raises(ValueError, match="current step requires a real next state"):
            _assembler().prepare([_item("terminal", None)])
