from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from nimloth.training.sft1.query_state_training_manifest import (
    build_connected_validation_split,
    build_coverage_manifest,
    build_full_training_manifest,
    build_generation_format_manifest,
    deserialize_generation_format_manifest,
    deserialize_query_state_training_manifest,
    deserialize_query_state_validation_split,
    normalize_instruction_identity,
    rows_for_training_mode,
    rows_for_validation_mode,
    validate_query_state_row_audit,
)
from nimloth.training.sft1.real_rows import SFT1V2Early4Row, SFT1V2RowAudit


_SHA = "a" * 64


def _row(
    ordinal: int,
    *,
    step: int,
    action: int,
    success: bool | None,
    record: str | None = None,
    image: str | None = None,
    instruction: str | None = None,
    external: bool = True,
) -> SFT1V2Early4Row:
    record_id = record or f"record-{ordinal}"
    image_id = image or f"{ordinal + 1:064x}"
    text = instruction or f"Find target {ordinal}"
    return SFT1V2Early4Row(
        schema="nimloth_sft1_state_v2_early4_row_v2",
        ordinal=ordinal,
        source_path="/data/source.jsonl",
        source_sha256=_SHA,
        split="val" if external else "train",
        record_id=record_id,
        step_index=step,
        original_image_path=f"/images/{image_id}.png",
        original_image_sha256=image_id,
        image_content_group=image_id,
        instruction=text,
        instruction_char_span=(0, len(text)),
        instruction_equivalence_group=normalize_instruction_identity(text),
        archived_assistant_response=(
            "<think>real archived thought</think><|latent_state|>"
            "<|action_start|><|action_(0)|><|action_end|>"
        ),
        executed_action_index=action,
        movement_success=success,
        external_eligible=external,
        record={},
    )


def _with_production_record(row: SFT1V2Early4Row) -> SFT1V2Early4Row:
    count = row.step_index + 1
    images = [f"/images/prior-{index}.png" for index in range(count)]
    images[-1] = row.original_image_path
    actions = [0] * count
    actions[-1] = row.executed_action_index
    responses = [
        "<think>real prior thought</think><|action_start|><|action_(0)|><|action_end|>"
        for _index in range(count)
    ]
    return replace(
        row,
        record={
            "system_prompt": "Navigate safely.",
            "observation_texts": [f"<image> observation {index}" for index in range(count + 1)],
            "image_paths": [*images, "/images/next.png"],
            "action_indices": actions,
            "assistant_responses": responses,
        },
    )


def _audit() -> SFT1V2RowAudit:
    return SFT1V2RowAudit(
        train_source_sha256=_SHA,
        validation_source_sha256="b" * 64,
        train_records=3211,
        validation_records=355,
        train_rows=12836,
        excluded_train_empty_cot_rows=5,
        raw_validation_rows=1420,
        excluded_validation_empty_cot_rows=0,
        external_validation_rows=1413,
        train_unique_images=10292,
        validation_unique_images=1060,
        cross_split_image_hashes=5,
        action_counts={},
        movement_outcome_counts={},
        same_image_multi_instruction_groups=42,
        same_instruction_multi_image_groups=101,
    )


def test_row_audit_reestablishes_all_reviewed_counts() -> None:
    validate_query_state_row_audit(_audit())
    for field in (
        "train_rows",
        "excluded_train_empty_cot_rows",
        "raw_validation_rows",
        "external_validation_rows",
        "same_image_multi_instruction_groups",
        "same_instruction_multi_image_groups",
    ):
        with pytest.raises(ValueError, match=field):
            validate_query_state_row_audit(replace(_audit(), **{field: getattr(_audit(), field) + 1}))


def test_coverage_selector_is_deterministic_stratified_and_identity_bound() -> None:
    rows = (
        _row(0, step=0, action=0, success=True),
        _row(1, step=0, action=0, success=True, record="shared"),
        _row(2, step=0, action=0, success=False),
        _row(3, step=1, action=4, success=None),
        _row(4, step=1, action=4, success=True),  # non-movement never uses outcome
        _row(5, step=2, action=2, success=None),
    )
    rendered = {row.identity: (100 + row.ordinal, 10 + row.ordinal) for row in rows}
    requested = {
        "step=0/action=0/outcome=success": 1,
        "step=0/action=0/outcome=failure": 1,
        "step=1/action=4/outcome=non_movement": 1,
        "step=2/action=2/outcome=unknown": 2,
    }
    one = build_coverage_manifest(rows, requested_per_stratum=requested, rendered_counts=rendered, seed=9)
    two = build_coverage_manifest(tuple(reversed(rows)), requested_per_stratum=requested, rendered_counts=rendered, seed=9)
    assert one.identity == two.identity
    assert one.row_identities == two.row_identities
    assert one.shortages == {"step=2/action=2/outcome=unknown": 1}
    assert {entry.rendered_token_count for entry in one.entries}
    assert {entry.valid_lm_token_count for entry in one.entries}
    assert len(one.row_identities) == len(set(one.row_identities))


def test_coverage_does_not_invent_outcomes_or_use_trajectory_success() -> None:
    movement_unknown = _row(0, step=0, action=0, success=None)
    rotation_with_record_success = replace(
        _row(1, step=0, action=4, success=None),
        record={"traj_success": True, "success": [True]},
    )
    rendered = {row.identity: (100, 10) for row in (movement_unknown, rotation_with_record_success)}
    manifest = build_coverage_manifest(
        (movement_unknown, rotation_with_record_success),
        requested_per_stratum={
            "step=0/action=0/outcome=unknown": 1,
            "step=0/action=4/outcome=non_movement": 1,
        },
        rendered_counts=rendered,
        seed=1,
    )
    assert {entry.outcome_bucket for entry in manifest.entries} == {"unknown", "non_movement"}


def test_full_manifest_covers_each_valid_row_exactly_once_per_epoch() -> None:
    rows = tuple(_row(index, step=index % 4, action=index % 8, success=None, external=False) for index in range(8))
    rendered = {row.identity: (100, 10) for row in rows}
    manifest = build_full_training_manifest(rows, rendered_counts=rendered, seed=3)
    assert len(manifest.entries) == 8
    assert set(manifest.row_identities) == {row.identity for row in rows}
    assert rows_for_training_mode(manifest, mode="formal") == manifest.row_identities
    with pytest.raises(ValueError, match="full.*formal"):
        rows_for_training_mode(manifest, mode="pilot")


def test_connected_split_keeps_image_instruction_components_wholly_disjoint() -> None:
    rows = (
        _row(0, step=0, action=0, success=True, image="1" * 64, instruction="Find Chair"),
        _row(1, step=1, action=2, success=False, image="1" * 64, instruction="Find lamp"),
        _row(2, step=2, action=3, success=True, image="2" * 64, instruction="  find   LAMP  "),
        _row(3, step=3, action=0, success=False, image="3" * 64, instruction="Find sofa"),
        _row(4, step=0, action=2, success=True, image="4" * 64, instruction="Find desk"),
        _row(5, step=1, action=3, success=False, image="5" * 64, instruction="Find plant"),
    )
    split = build_connected_validation_split(
        rows,
        calibration_numerator=1,
        calibration_denominator=2,
        seed=17,
    )
    assert len(split.calibration_row_identities) + len(split.holdout_row_identities) == len(rows)
    assert not set(split.calibration_component_ids) & set(split.holdout_component_ids)
    side = {
        identity: "calibration" for identity in split.calibration_row_identities
    } | {identity: "holdout" for identity in split.holdout_row_identities}
    assert side[rows[0].identity] == side[rows[1].identity] == side[rows[2].identity]
    assert split.identity == build_connected_validation_split(
        tuple(reversed(rows)), calibration_numerator=1, calibration_denominator=2, seed=17
    ).identity


def test_strict_training_deserialization_recomputes_selector_and_identity(tmp_path: Path) -> None:
    rows = tuple(
        _row(index, step=index % 2, action=4, success=None, external=False)
        for index in range(4)
    )
    rendered = {row.identity: (100 + row.ordinal, 10 + row.ordinal) for row in rows}
    requested = {
        "step=0/action=4/outcome=non_movement": 1,
        "step=1/action=4/outcome=non_movement": 1,
    }
    manifest = build_coverage_manifest(
        rows,
        requested_per_stratum=requested,
        rendered_counts=rendered,
        seed=7,
    )
    raw = {
        "schema": manifest.schema,
        "kind": manifest.kind,
        "seed": manifest.seed,
        "entries": [asdict(entry) for entry in manifest.entries],
        "requested_per_stratum": requested,
        "shortages": dict(manifest.shortages),
        "selection_priority": "record_then_image_then_normalized_instruction_then_sha256",
        "movement_action_table": [
            "moveahead", "moveback", "moveright", "moveleft",
            "rotateright", "rotateleft", "lookup", "lookdown",
        ],
        "identity": manifest.identity,
    }
    path = tmp_path / "train.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    parsed = deserialize_query_state_training_manifest(
        path,
        rows=rows,
        expected_identity=manifest.identity,
        expected_mode="pilot",
        expected_rows=2,
        expected_seed=7,
    )
    assert parsed == manifest

    raw["entries"][0]["record_id"] = "tampered"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="audited row|identity"):
        deserialize_query_state_training_manifest(
            path,
            rows=rows,
            expected_identity=manifest.identity,
            expected_mode="pilot",
            expected_rows=2,
            expected_seed=7,
        )


def test_strict_validation_deserialization_recomputes_all_1413_connected_rows(tmp_path: Path) -> None:
    rows = tuple(
        _row(index, step=index % 4, action=index % 8, success=None)
        for index in range(1413)
    )
    split = build_connected_validation_split(
        rows, calibration_numerator=1, calibration_denominator=2, seed=19
    )
    raw = json.loads(json.dumps(asdict(split)))
    path = tmp_path / "validation.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert deserialize_query_state_validation_split(
        path, rows=rows, expected_identity=split.identity
    ) == split

    raw["holdout_row_identities"].append(raw["calibration_row_identities"][0])
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="connected split|identity"):
        deserialize_query_state_validation_split(
            path, rows=rows, expected_identity=split.identity
        )


def test_generation_format_manifest_binds_real_split_rows_budgets_and_protocols(
    tmp_path: Path,
) -> None:
    rows = tuple(
        _with_production_record(
            _row(index, step=index % 4, action=index % 8, success=None)
        )
        for index in range(8)
    )
    split = build_connected_validation_split(
        rows, calibration_numerator=1, calibration_denominator=2, seed=5
    )
    by_identity = {row.identity: row for row in rows}
    calibration_rows = tuple(by_identity[value] for value in split.calibration_row_identities[:2])
    manifest = build_generation_format_manifest(
        calibration_rows,
        validation_split=split,
        mode="pilot",
        max_reasoning_tokens=12,
        max_output_tokens=36,
        turn_generation_spec_identity="c" * 64,
    )
    assert manifest.split == "calibration"
    assert len(manifest.entries) == len(calibration_rows)
    assert all(len(entry.prompt_identity) == 64 for entry in manifest.entries)
    path = tmp_path / "generation-format.json"
    path.write_text(json.dumps(asdict(manifest)), encoding="utf-8")
    assert deserialize_generation_format_manifest(
        path,
        rows=rows,
        validation_split=split,
        expected_identity=manifest.identity,
        expected_mode="pilot",
    ) == manifest

    duplicate = json.loads(path.read_text(encoding="utf-8"))
    duplicate["entries"].append(duplicate["entries"][0])
    path.write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        deserialize_generation_format_manifest(
            path,
            rows=rows,
            validation_split=split,
            expected_identity=manifest.identity,
            expected_mode="pilot",
        )

    unregistered = json.loads(json.dumps(asdict(manifest)))
    unregistered["entries"][0]["row_identity"] = "f" * 64
    path.write_text(json.dumps(unregistered), encoding="utf-8")
    with pytest.raises(ValueError, match="unregistered"):
        deserialize_generation_format_manifest(
            path,
            rows=rows,
            validation_split=split,
            expected_identity=manifest.identity,
            expected_mode="pilot",
        )

    holdout_row = by_identity[split.holdout_row_identities[0]]
    with pytest.raises(ValueError, match="outside.*calibration"):
        build_generation_format_manifest(
            (holdout_row,),
            validation_split=split,
            mode="pilot",
            max_reasoning_tokens=12,
            max_output_tokens=36,
            turn_generation_spec_identity="c" * 64,
        )


def test_pilot_cannot_open_holdout_and_formal_cannot_open_calibration() -> None:
    rows = tuple(_row(index, step=index % 4, action=index % 8, success=None) for index in range(6))
    split = build_connected_validation_split(rows, calibration_numerator=1, calibration_denominator=2, seed=2)
    assert rows_for_validation_mode(split, mode="pilot") == split.calibration_row_identities
    assert rows_for_validation_mode(split, mode="formal") == split.holdout_row_identities
    with pytest.raises(ValueError, match="pilot.*holdout"):
        rows_for_validation_mode(split, mode="pilot", requested_split="holdout")
    with pytest.raises(ValueError, match="formal.*holdout"):
        rows_for_validation_mode(split, mode="formal", requested_split="calibration")
