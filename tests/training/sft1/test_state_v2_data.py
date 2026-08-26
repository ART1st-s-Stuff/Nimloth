from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from nimloth.training.sft1.data import prepare_sft1_v2_row
from tests.training.sft1._state_v2_fixtures import (
    manifest,
    teacher_row,
    trajectory_record,
)


def test_current_trajectory_builds_real_cot_and_actual_outcome_row(tmp_path) -> None:
    bound = manifest()
    record, image = trajectory_record(tmp_path)
    row = prepare_sft1_v2_row(
        record,
        step_index=0,
        encoded_tensors={
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.ones(3, dtype=torch.long),
        },
        teacher=teacher_row(image, manifest_value=bound),
        manifest=bound,
    )

    assert row.archived_assistant_response.startswith("<think>")
    assert row.executed_action_index == 0
    assert row.movement_success is True
    assert row.original_image_sha256 == teacher_row(
        image, manifest_value=bound
    ).original_image_sha256
    assert row.encoded_tensors["input_ids"].shape == (3,)
    assert not row.dino_regions.requires_grad


def test_nonmovement_and_unavailable_outcomes_are_masked_not_invented(tmp_path) -> None:
    bound = manifest()
    rotation, rotation_image = trajectory_record(
        tmp_path,
        record_id="rotation",
        action_index=4,
    )
    rotation_row = prepare_sft1_v2_row(
        rotation,
        step_index=0,
        encoded_tensors={"input_ids": torch.tensor([1])},
        teacher=teacher_row(
            rotation_image,
            record_id="rotation",
            manifest_value=bound,
        ),
        manifest=bound,
    )
    assert rotation_row.movement_success is None

    unavailable, unavailable_image = trajectory_record(
        tmp_path,
        record_id="unavailable",
        feedback="feedback unavailable",
    )
    unavailable_row = prepare_sft1_v2_row(
        unavailable,
        step_index=0,
        encoded_tensors={"input_ids": torch.tensor([1])},
        teacher=teacher_row(
            unavailable_image,
            record_id="unavailable",
            manifest_value=bound,
        ),
        manifest=bound,
    )
    assert unavailable_row.movement_success is None


def test_data_contract_rejects_aliases_student_cache_and_mixed_identity(tmp_path) -> None:
    bound = manifest()
    record, image = trajectory_record(tmp_path)
    teacher = teacher_row(image, manifest_value=bound)

    legacy_alias = dict(record)
    legacy_alias["response"] = legacy_alias["assistant_responses"]
    with pytest.raises(ValueError, match="unknown field"):
        prepare_sft1_v2_row(
            legacy_alias,
            step_index=0,
            encoded_tensors={"input_ids": torch.tensor([1])},
            teacher=teacher,
            manifest=bound,
        )

    with pytest.raises(ValueError, match="precomputed student"):
        prepare_sft1_v2_row(
            record,
            step_index=0,
            encoded_tensors={
                "input_ids": torch.tensor([1]),
                "query_hidden": torch.zeros(16, 2048),
            },
            teacher=teacher,
            manifest=bound,
        )

    with pytest.raises(ValueError, match="manifest identity mismatch"):
        prepare_sft1_v2_row(
            record,
            step_index=0,
            encoded_tensors={"input_ids": torch.tensor([1])},
            teacher=replace(teacher, manifest_identity="0" * 64),
            manifest=bound,
        )

    image.write_bytes(b"changed-after-cache")
    with pytest.raises(ValueError, match="image digest"):
        prepare_sft1_v2_row(
            record,
            step_index=0,
            encoded_tensors={"input_ids": torch.tensor([1])},
            teacher=teacher,
            manifest=bound,
        )
