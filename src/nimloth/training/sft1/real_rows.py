"""Strict early-4 trajectory indexing and K8-record to K16 prompt rendering."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from nimloth.backbone.qwen25vl.batch import collect_message_images, render_messages
from nimloth.backbone.qwen25vl.state_training import (
    exact_instruction_token_span,
    require_archived_assistant_response,
)
from nimloth.latent import (
    LatentActionTokens,
    find_extraction_positions,
    latent_state_block,
    special_token_ids,
)
from nimloth.rollout import RolloutTrajectory
from nimloth.training.sft1.experiment_config import EARLY4_STEPS, SFT1V2Config
from nimloth.training.sft1.data import sha256_file


EARLY4_ROW_SCHEMA = "nimloth_sft1_state_v2_early4_row_v1"


@dataclass(frozen=True)
class SFT1V2Early4Row:
    schema: str
    ordinal: int
    source_path: str
    source_sha256: str
    split: str
    record_id: str
    step_index: int
    original_image_path: str
    original_image_sha256: str
    image_content_group: str
    instruction: str
    instruction_equivalence_group: str
    archived_assistant_response: str
    executed_action_index: int
    movement_success: bool | None
    external_eligible: bool
    record: Mapping[str, Any]

    @property
    def identity(self) -> str:
        payload = {
            key: value
            for key, value in asdict(self).items()
            if key not in {"ordinal", "record"}
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True)
class SFT1V2RowAudit:
    train_source_sha256: str
    validation_source_sha256: str
    train_records: int
    validation_records: int
    train_rows: int
    raw_validation_rows: int
    external_validation_rows: int
    train_unique_images: int
    validation_unique_images: int
    cross_split_image_hashes: int
    action_counts: Mapping[str, Mapping[int, int]]
    movement_outcome_counts: Mapping[str, Mapping[int, Mapping[str, int]]]
    same_image_multi_instruction_groups: int
    same_instruction_multi_image_groups: int
    overlap_key: str = "original_image_sha256"


@dataclass(frozen=True)
class SFT1V2RenderedRow:
    row: SFT1V2Early4Row
    rendered_text: str
    input_ids: torch.Tensor
    instruction_token_span: tuple[int, int]
    action_boundary_index: int
    encoded_tensors: Mapping[str, torch.Tensor]


def _movement_success(trajectory: RolloutTrajectory, step_index: int) -> bool | None:
    action = int(trajectory.action_indices[step_index])
    if action not in (0, 2, 3):
        return None
    if len(trajectory.observation_texts) != trajectory.num_steps + 1:
        raise ValueError("movement feedback requires action-aligned observations")
    feedback = str(trajectory.observation_texts[step_index + 1])
    success = "Last action is executed successfully." in feedback
    failure = "Last action is not executed successfully." in feedback
    if success == failure:
        return None if not success else (_ for _ in ()).throw(
            ValueError("movement feedback is contradictory")
        )
    return success


def _read_records(path: Path, expected_sha256: str, expected_split: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"trajectory source is missing: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"trajectory source hash mismatch: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid trajectory JSON at {path}:{line_number}") from error
            if not isinstance(raw, dict):
                raise ValueError(f"trajectory row is not an object at {path}:{line_number}")
            trajectory = RolloutTrajectory.from_record(raw)
            if trajectory.split != expected_split:
                raise ValueError(f"trajectory split mismatch at {path}:{line_number}")
            if set(raw) != set(trajectory.to_record()):
                unknown = sorted(set(raw) - set(trajectory.to_record()))
                raise ValueError(f"unknown trajectory field at {path}:{line_number}: {unknown[0]}")
            records.append(raw)
    return records


def _selected_rows(
    records: Sequence[dict[str, Any]],
    *,
    source_path: Path,
    source_sha256: str,
    split: str,
    train_image_hashes: frozenset[str],
    ordinal_start: int,
) -> list[SFT1V2Early4Row]:
    rows: list[SFT1V2Early4Row] = []
    for record in records:
        trajectory = RolloutTrajectory.from_record(record)
        for step_index in EARLY4_STEPS:
            if step_index >= trajectory.num_steps:
                continue
            response = require_archived_assistant_response(
                trajectory.assistant_responses[step_index], source="archived"
            )
            # Historical archives in this experiment are structurally K8. Reject
            # missing/malformed/generated rows rather than fabricating a prefix.
            expected_k8 = latent_state_block(8)
            if response.count(expected_k8 + LatentActionTokens().action_start) != 1:
                raise ValueError(
                    f"record {trajectory.record_id} step {step_index} has no single structural K8 block"
                )
            action_start = response.find(LatentActionTokens().action_start)
            if action_start < 0 or response.find(LatentActionTokens().action_start, action_start + 1) >= 0:
                raise ValueError("archived response must contain one exact action boundary")
            image_path = Path(trajectory.image_paths[step_index])
            if not image_path.is_file():
                raise FileNotFoundError(f"original observation image is missing: {image_path}")
            image_sha = sha256_file(image_path)
            instruction = trajectory.instruction
            if not isinstance(instruction, str) or not instruction:
                raise ValueError("trajectory instruction must be non-empty")
            instruction_group = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
            rows.append(SFT1V2Early4Row(
                schema=EARLY4_ROW_SCHEMA,
                ordinal=ordinal_start + len(rows),
                source_path=str(source_path),
                source_sha256=source_sha256,
                split=split,
                record_id=trajectory.record_id,
                step_index=step_index,
                original_image_path=str(image_path),
                original_image_sha256=image_sha,
                image_content_group=image_sha,
                instruction=instruction,
                instruction_equivalence_group=instruction_group,
                archived_assistant_response=response,
                executed_action_index=int(trajectory.action_indices[step_index]),
                movement_success=_movement_success(trajectory, step_index),
                external_eligible=split != "val" or image_sha not in train_image_hashes,
                record=record,
            ))
    return rows


def _group_count(rows: Sequence[SFT1V2Early4Row], group: str, varying: str) -> int:
    groups: dict[str, set[str]] = {}
    for row in rows:
        if not row.external_eligible:
            continue
        groups.setdefault(str(getattr(row, group)), set()).add(str(getattr(row, varying)))
    return sum(len(values) > 1 for values in groups.values())


def index_early4_rows(
    config: SFT1V2Config,
    *,
    enforce_approved_counts: bool = True,
) -> tuple[tuple[SFT1V2Early4Row, ...], SFT1V2RowAudit]:
    """Read immutable sources, hash original images, and build deterministic rows."""

    train_path = Path(config.data.train_jsonl)
    validation_path = Path(config.data.validation_jsonl)
    train_records = _read_records(train_path, config.data.train_sha256, config.data.train_split)
    validation_records = _read_records(validation_path, config.data.validation_sha256, config.data.validation_split)

    # First pass obtains train image identities before external eligibility is assigned.
    train_rows = _selected_rows(
        train_records, source_path=train_path, source_sha256=config.data.train_sha256,
        split="train", train_image_hashes=frozenset(), ordinal_start=0,
    )
    train_images = frozenset(row.original_image_sha256 for row in train_rows)
    validation_rows = _selected_rows(
        validation_records, source_path=validation_path,
        source_sha256=config.data.validation_sha256, split="val",
        train_image_hashes=train_images, ordinal_start=len(train_rows),
    )
    all_rows = (*train_rows, *validation_rows)
    validation_images = frozenset(row.original_image_sha256 for row in validation_rows)

    action_counts: dict[str, dict[int, int]] = {"train": {}, "val": {}}
    outcomes: dict[str, dict[int, dict[str, int]]] = {"train": {}, "val": {}}
    for row in all_rows:
        action_counts[row.split][row.executed_action_index] = action_counts[row.split].get(row.executed_action_index, 0) + 1
        if row.movement_success is not None:
            bucket = outcomes[row.split].setdefault(row.executed_action_index, {"success": 0, "failure": 0})
            bucket["success" if row.movement_success else "failure"] += 1
    audit = SFT1V2RowAudit(
        train_source_sha256=config.data.train_sha256,
        validation_source_sha256=config.data.validation_sha256,
        train_records=len(train_records), validation_records=len(validation_records),
        train_rows=len(train_rows), raw_validation_rows=len(validation_rows),
        external_validation_rows=sum(row.external_eligible for row in validation_rows),
        train_unique_images=len(train_images), validation_unique_images=len(validation_images),
        cross_split_image_hashes=len(train_images & validation_images),
        action_counts=action_counts, movement_outcome_counts=outcomes,
        same_image_multi_instruction_groups=_group_count(validation_rows, "image_content_group", "instruction_equivalence_group"),
        same_instruction_multi_image_groups=_group_count(validation_rows, "instruction_equivalence_group", "image_content_group"),
    )
    if enforce_approved_counts:
        checks = {
            "train_records": config.selection.train_records,
            "validation_records": config.selection.validation_records,
            "train_rows": config.selection.train_rows,
            "raw_validation_rows": config.selection.raw_validation_rows,
            "external_validation_rows": config.selection.external_validation_rows,
            "cross_split_image_hashes": config.selection.cross_split_image_hashes,
            "same_image_multi_instruction_groups": config.selection.same_image_multi_instruction_groups,
            "same_instruction_multi_image_groups": config.selection.same_instruction_multi_image_groups,
        }
        for field, expected in checks.items():
            if getattr(audit, field) != expected:
                raise ValueError(f"early-4 audit {field} mismatch: {getattr(audit, field)} != {expected}")
    return tuple(all_rows), audit


def _single_action_boundary(input_ids: torch.Tensor, tokenizer: Any) -> int:
    token = LatentActionTokens().action_start
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError("action boundary token is unavailable")
    positions = torch.nonzero(input_ids == int(token_id), as_tuple=False).flatten()
    if positions.numel() < 1:
        raise ValueError("rendered row has no action boundary")
    # State replay may contain prior action boundaries; the current boundary is last.
    return int(positions[-1].item())


def audit_rendered_token_upper_bound(
    rows: Sequence[SFT1V2Early4Row],
    *,
    processor: Any,
    max_sequence_length: int,
    max_pixels: int,
) -> int:
    """Prove a conservative no-truncation bound without image tensor creation."""

    image_processor = getattr(processor, "image_processor", None)
    patch_size = int(getattr(image_processor, "patch_size", 0))
    merge_size = int(getattr(image_processor, "merge_size", 0))
    if patch_size < 1 or merge_size < 1 or max_pixels < 1:
        raise ValueError("Qwen image processor patch/merge/max-pixel contract is invalid")
    patch_area = patch_size * patch_size
    # Bound pre-merge patch count and add one rounded grid row/column. This is
    # intentionally looser than the actual merge-size reduction, so resize
    # rounding cannot make the no-truncation proof optimistic.
    max_image_tokens = (
        (max_pixels + patch_area - 1) // patch_area
        + 2 * ((math.isqrt(max_pixels) + patch_size - 1) // patch_size)
        + 4
    )
    maximum = 0
    for row in rows:
        trajectory = RolloutTrajectory.from_record(dict(row.record))
        messages = trajectory.build_state_prompt(row.step_index).bound_messages()
        rendered = render_messages(
            messages,
            processor,
            add_generation_prompt=False,
            latent_token_count=16,
        )
        encoded = processor.tokenizer(rendered, add_special_tokens=False)
        ids = encoded["input_ids"]
        image_count = sum(
            1
            for message in messages
            for item in (
                message.get("content", [])
                if isinstance(message.get("content"), list)
                else []
            )
            if item.get("type") == "image"
        )
        # The rendered text already contains one placeholder token per image;
        # adding the full upper bound again is conservative by construction.
        bound = len(ids) + image_count * (max_image_tokens + 2)
        maximum = max(maximum, bound)
        if bound > max_sequence_length:
            raise ValueError(
                f"row {row.record_id}:{row.step_index} may exceed the no-truncation "
                f"limit: upper_bound={bound} > {max_sequence_length}"
            )
    return maximum


def render_early4_row(
    row: SFT1V2Early4Row,
    *,
    processor: Any,
    max_length: int,
) -> SFT1V2RenderedRow:
    """Render the real trajectory prefix in inject mode without truncation."""

    if max_length < 1:
        raise ValueError("max_length must be positive")
    trajectory = RolloutTrajectory.from_record(dict(row.record))
    if trajectory.record_id != row.record_id or row.step_index >= trajectory.num_steps:
        raise ValueError("early-4 row record identity mismatch")
    response = trajectory.assistant_responses[row.step_index]
    if response != row.archived_assistant_response:
        raise ValueError("archived response changed after indexing")
    k8 = latent_state_block(8)
    k16 = latent_state_block(16)
    if response.count(k8 + LatentActionTokens().action_start) != 1:
        raise ValueError("historical response must contain exactly one K8 structural block")
    expected_prefix = response[: response.index(LatentActionTokens().action_start) + len(LatentActionTokens().action_start)]
    prompt = trajectory.build_state_prompt(row.step_index)
    messages = prompt.bound_messages()
    rendered = render_messages(messages, processor, add_generation_prompt=False, latent_token_count=16)
    if (k8 + LatentActionTokens().action_start) in rendered or rendered.count(k16) < 1:
        raise ValueError("inject renderer did not normalize K8 structural queries to K16")
    # The current assistant payload in the rendered chat must differ only in the
    # structural block. This locks actual CoT bytes and the action boundary.
    normalized_prefix = expected_prefix.replace(k8, k16)
    if normalized_prefix not in rendered:
        raise ValueError("K8 to K16 rendering changed archived CoT/action boundaries")

    images = collect_message_images(messages)
    encoded = processor(
        text=[rendered], images=[images] if images else None, padding=False,
        truncation=False, return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("processor returned invalid rendered input_ids")
    if input_ids.shape[1] > max_length:
        raise ValueError("early-4 row would truncate; truncation is forbidden")
    one_row = input_ids[0].detach().cpu()
    span = exact_instruction_token_span(one_row, tokenizer=processor.tokenizer, instruction=row.instruction)
    boundary = _single_action_boundary(one_row, processor.tokenizer)
    token_map = special_token_ids(processor.tokenizer, latent_token_count=16)
    positions = find_extraction_positions(
        one_row,
        token_map,
        LatentActionTokens(),
        latent_token_count=16,
    )
    if (
        positions.latent_state_indices is None
        or len(positions.latent_state_indices) != 16
        or positions.action_start_index != boundary
        or positions.latent_state_indices[-1] + 1 != boundary
    ):
        raise ValueError("rendered row lacks one exact current K16/action boundary")
    if not 0 <= span[0] < span[1] <= boundary:
        raise ValueError("instruction span is not before the exact action boundary")
    tensors: dict[str, torch.Tensor] = {}
    for name, value in encoded.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError("processor outputs must be tensors")
        squeezed = value.squeeze(0)
        if name == "image_grid_thw" and squeezed.ndim == 1:
            squeezed = squeezed.unsqueeze(0)
        tensors[name] = squeezed.contiguous().detach()
    return SFT1V2RenderedRow(
        row=row, rendered_text=rendered, input_ids=one_row,
        instruction_token_span=span, action_boundary_index=boundary,
        encoded_tensors=tensors,
    )


__all__ = [
    "EARLY4_ROW_SCHEMA", "SFT1V2Early4Row", "SFT1V2RenderedRow",
    "SFT1V2RowAudit", "audit_rendered_token_upper_bound",
    "index_early4_rows", "render_early4_row",
]
