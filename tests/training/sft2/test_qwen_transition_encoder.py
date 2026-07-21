"""Qwen batch builder 的下一状态去重与 cache 契约测试。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from nimloth.backbone.qwen25vl.batch import message_cache_key
from nimloth.backbone.qwen25vl.transition import (
    CachedQwenNextBatch,
    Qwen25VLBatchBuilder,
)


def _builder() -> Qwen25VLBatchBuilder:
    processor = MagicMock()
    processor.tokenizer.pad_token_id = 0
    return Qwen25VLBatchBuilder(
        processor=processor,
        device=torch.device("cpu"),
        max_length=32,
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
        "nimloth.backbone.qwen25vl.transition.build_qwen_batch",
        side_effect=fake_build,
    ):
        batch = _builder().prepare(
            [_item("a", shared_next), _item("b", shared_next)]
        )

    assert batch_sizes == [2, 1]
    assert batch.next.tensors["input_ids"].shape[0] == 1
    assert batch.next_indices.tolist() == [0, 0]
    assert batch.non_terminal_mask.tolist() == [True, True]


def test_prepare_uses_worker_prebatched_next_cache() -> None:
    next_messages = [{"role": "user", "content": "next"}]
    cached = CachedQwenNextBatch(
        keys=(message_cache_key(next_messages),),
        encoding={"input_ids": torch.ones((1, 3), dtype=torch.long)},
    )
    raw = {
        "items": [_item("a", next_messages)],
        "current_enc": {"input_ids": torch.ones((1, 2), dtype=torch.long)},
        "next_enc_bundle": cached,
    }

    with patch("nimloth.backbone.qwen25vl.transition.build_qwen_batch") as build:
        batch = _builder().prepare(raw)

    build.assert_not_called()
    assert torch.equal(
        batch.next.tensors["input_ids"],
        torch.ones((1, 3), dtype=torch.long),
    )


def test_terminal_batch_builds_one_dummy_next_input() -> None:
    seen: list[list[dict]] = []

    def fake_build(items, _processor, _max_length, **_kwargs):
        seen.append(items)
        return {"input_ids": torch.ones((len(items), 2), dtype=torch.long)}

    with patch(
        "nimloth.backbone.qwen25vl.transition.build_qwen_batch",
        side_effect=fake_build,
    ):
        batch = _builder().prepare([_item("terminal", None)])

    assert len(seen) == 2
    assert seen[1] == [
        {"messages": [{"role": "user", "content": "terminal"}]}
    ]
    assert batch.non_terminal_mask.tolist() == [False]
    assert batch.next_indices.tolist() == [0]
