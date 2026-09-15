"""通用 Qwen input builder 与 SFT2 batch assembler 的边界测试。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from nimloth.backbone.qwen25vl.input import Qwen25VLInputBuilder
from nimloth.training.sft.stage3.batch import SFT2BatchAssembler, SFT2RolloutBatch


def _assembler() -> SFT2BatchAssembler:
    processor = MagicMock()
    processor.tokenizer.pad_token_id = 0
    processor.image_token = None
    return SFT2BatchAssembler(
        input_builder=Qwen25VLInputBuilder(
            processor=processor,
            max_length=32,
        ),
        device=torch.device("cpu"),
        history_size=1,
    )


def _rollout_assembler() -> SFT2BatchAssembler:
    processor = MagicMock()
    processor.tokenizer.pad_token_id = 0
    processor.image_token = None
    return SFT2BatchAssembler(
        input_builder=Qwen25VLInputBuilder(
            processor=processor,
            max_length=32,
        ),
        device=torch.device("cpu"),
        history_size=1,
        prediction_horizon=4,
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


def test_prepare_uses_worker_materialized_compact_rows() -> None:
    next_messages = [{"role": "user", "content": "next"}]
    assembler = _assembler()
    raw = {
        "items": [_item("a", next_messages)],
        "current_enc_rows": [
            {
                "input_ids": torch.ones(2, dtype=torch.long),
                "attention_mask": torch.ones(2, dtype=torch.long),
                "labels": torch.ones(2, dtype=torch.long),
            }
        ],
        "next_enc_rows": [
            {
                "input_ids": torch.ones(3, dtype=torch.long),
                "attention_mask": torch.ones(3, dtype=torch.long),
            }
        ],
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


def test_prepare_t4_rollout_keeps_recorded_actions_and_all_next_targets() -> None:
    items = []
    for step, action in enumerate((2, 0, 1, 2)):
        item = _item(
            f"record:{step}",
            [{"role": "user", "content": f"next-{step}"}],
        )
        item.update(
            step_index=step,
            action_index=action,
            action_value_target=float(10 + step),
            prediction_horizon=4,
            rollout_position=step,
            is_current_step=step == 0,
            needs_next_state=True,
            next_image_path=f"image-{step}.png",
        )
        items.append(item)
    batch_sizes: list[int] = []

    def fake_build(rows, _processor, _max_length, **_kwargs):
        batch_sizes.append(len(rows))
        return {"input_ids": torch.zeros((len(rows), 4), dtype=torch.long)}

    with patch(
        "nimloth.backbone.qwen25vl.input.build_qwen_batch",
        side_effect=fake_build,
    ):
        batch = _rollout_assembler().prepare(items)

    assert isinstance(batch, SFT2RolloutBatch)
    assert batch_sizes == [1, 1, 4]
    assert batch.action_sequences.tolist() == [[2, 0, 1, 2]]
    assert batch.value_target_sequences.tolist() == [[10.0, 11.0, 12.0, 13.0]]
    assert batch.next_indices.tolist() == [[0, 1, 2, 3]]
    assert batch.next_image_paths == tuple(f"image-{step}.png" for step in range(4))


def _encoded_row(value: int, *, labels: bool) -> dict[str, torch.Tensor]:
    row = {
        "input_ids": torch.tensor([value, value + 1]),
        "attention_mask": torch.ones(2, dtype=torch.long),
    }
    if labels:
        row["labels"] = torch.tensor([value, value + 1])
    return row


def test_prepare_cached_t4_accepts_unlabelled_terminal_next_state() -> None:
    items = []
    next_rows = []
    for step, action in enumerate((2, 0, 1, 2)):
        item = _item(
            f"record:{step}",
            [{"role": "user", "content": f"next-{step}"}],
        )
        item.update(
            step_index=step,
            action_index=action,
            action_value_target=float(10 + step),
            prediction_horizon=4,
            rollout_position=step,
            is_current_step=step == 0,
            needs_next_state=True,
            next_image_path=f"image-{step}.png",
        )
        items.append(item)
        next_rows.append(_encoded_row(10 + step, labels=step < 3))
    raw = {
        "items": items,
        "current_enc_rows": [_encoded_row(1, labels=True)],
        "next_enc_rows": next_rows,
    }

    batch = _rollout_assembler().prepare(raw)

    assert isinstance(batch, SFT2RolloutBatch)
    assert batch.next.tensors["input_ids"].shape == (4, 2)
    assert "labels" not in batch.next.tensors
    assert "labels" not in batch.online_tail.tensors
    assert batch.action_sequences.tolist() == [[2, 0, 1, 2]]


def test_supervised_cached_rows_require_labels_on_every_row() -> None:
    processor = MagicMock()
    processor.tokenizer.pad_token_id = 0
    processor.image_token = None
    builder = Qwen25VLInputBuilder(processor=processor, max_length=32)

    with pytest.raises(ValueError, match="must all contain labels"):
        builder.collate_encoded(
            [_encoded_row(1, labels=True), _encoded_row(2, labels=False)],
            include_labels=True,
        )


@pytest.mark.parametrize("success", [True, False])
def test_success_mask_and_counts_follow_full_trajectory_metadata(success):
    assembler = _assembler()
    item = _item("a", [{"role": "user", "content": "next"}])
    item["success"] = success
    assert assembler.supervision_counts([item]) == (1, int(success))
    with patch("nimloth.backbone.qwen25vl.input.build_qwen_batch",
               return_value={"input_ids": torch.zeros((1, 4), dtype=torch.long)}):
        batch = assembler.prepare([item])
    assert batch.current.tensors["lm_row_weights"].tolist() == [float(success)]
    assert batch.sample_weights.tolist() == [1.]
    assert batch.current_value_targets.tolist() == [1.]


@pytest.mark.parametrize("success", [None, "false", 0])
def test_missing_or_non_boolean_success_is_rejected(success):
    item = _item("a", [])
    item["success"] = success
    with pytest.raises(ValueError, match="success boolean"):
        _assembler().supervision_counts([item])


@pytest.mark.parametrize("padding", [False, True])
def test_outcome_labels_align_rollout_and_exclude_padding(padding):
    assembler = _rollout_assembler()
    items = []
    for index, label in enumerate([True, False, None, True]):
        item = _item(str(index), [{"role": "user", "content": f"next {index}"}])
        item.update(step_index=index, prediction_horizon=4, rollout_position=index,
                    is_current_step=index == 0, needs_next_state=True,
                    action_success=label, loss_weight=float(not padding))
        items.append(item)
    assert assembler.outcome_count(items) == (0 if padding else 3)
    with patch("nimloth.backbone.qwen25vl.input.build_qwen_batch",
               return_value={"input_ids": torch.zeros((1, 4), dtype=torch.long)}):
        batch = assembler.prepare(items)
    assert batch.outcome_targets.tolist() == [[1.0, 0.0, 0.0, 1.0]]
    assert batch.outcome_mask.tolist() == [[not padding, not padding, False, not padding]]
    assert assembler.outcome_count(batch) == (0 if padding else 3)


def test_missing_outcome_remains_unsupervised():
    assembler = _assembler()
    item = _item("a", [{"role": "user", "content": "next"}])
    with patch("nimloth.backbone.qwen25vl.input.build_qwen_batch",
               return_value={"input_ids": torch.zeros((1, 4), dtype=torch.long)}):
        batch = assembler.prepare([item])
    assert batch.outcome_mask.tolist() == [False]
    assert assembler.outcome_count([item]) == 0
