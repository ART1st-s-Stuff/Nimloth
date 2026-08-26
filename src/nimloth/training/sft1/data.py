"""Strict current-trajectory rows and detached teacher targets for SFT1-v2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping, Protocol

import torch

from nimloth.backbone.qwen25vl.state_training import require_archived_assistant_response
from nimloth.rollout import RolloutTrajectory
from nimloth.training.sft1.manifest import SFT1V2Manifest
from nimloth.training.sft1.objective import OBSERVED_MOVEMENT_ACTION_INDICES


_SUCCESS_FEEDBACK = "Last action is executed successfully."
_FAILURE_FEEDBACK = "Last action is not executed successfully."
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_STUDENT_KEYS = frozenset(
    {"hidden", "query_hidden", "student_hidden", "state", "projected_state"}
)


@dataclass(frozen=True)
class SFT1V2TeacherRow:
    """One immutable cache row; every tensor is a detached target, never student state."""

    manifest_identity: str
    record_id: str
    step_index: int
    original_image_sha256: str
    image_content_group: str
    instruction_equivalence_group: str
    dino_regions: torch.Tensor
    instruction_teacher: torch.Tensor
    actor_teacher_log_probs: torch.Tensor


@dataclass(frozen=True)
class SFT1V2PreparedRow:
    """One encoded student prompt paired with strict detached teacher targets."""

    record_id: str
    step_index: int
    split: str
    original_image_path: str
    original_image_sha256: str
    image_content_group: str
    instruction_equivalence_group: str
    archived_assistant_response: str
    executed_action_index: int
    movement_success: bool | None
    encoded_tensors: Mapping[str, torch.Tensor]
    dino_regions: torch.Tensor
    instruction_teacher: torch.Tensor
    actor_teacher_log_probs: torch.Tensor
    manifest_identity: str

    @property
    def token_count(self) -> int:
        return int(self.encoded_tensors["input_ids"].numel())


class SFT1V2TeacherTargetBuilder(Protocol):
    """Preparation-only boundary; implementations must use original images and exact spans."""

    def build(
        self,
        *,
        record_id: str,
        step_index: int,
        original_image_path: Path,
        original_image_sha256: str,
        rendered_input_ids: torch.Tensor,
        instruction_token_span: tuple[int, int],
        archived_assistant_response: str,
        manifest: SFT1V2Manifest,
    ) -> SFT1V2TeacherRow: ...


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_movement_success(
    trajectory: RolloutTrajectory,
    step_index: int,
    action_index: int,
) -> bool | None:
    if action_index not in OBSERVED_MOVEMENT_ACTION_INDICES:
        return None
    if len(trajectory.observation_texts) != trajectory.num_steps + 1:
        raise ValueError("movement feedback requires action-aligned observation_texts")
    feedback = str(trajectory.observation_texts[step_index + 1])
    succeeded = _SUCCESS_FEEDBACK in feedback
    failed = _FAILURE_FEEDBACK in feedback
    if succeeded and failed:
        raise ValueError("movement feedback contains contradictory outcomes")
    if not succeeded and not failed:
        return None
    return succeeded


def _validate_teacher_row(
    teacher: SFT1V2TeacherRow,
    manifest: SFT1V2Manifest,
) -> None:
    if teacher.manifest_identity != manifest.identity:
        raise ValueError("teacher row manifest identity mismatch")
    if _SHA256_RE.fullmatch(teacher.original_image_sha256) is None:
        raise ValueError("teacher original image identity must be a SHA256 digest")
    if not teacher.image_content_group or not teacher.instruction_equivalence_group:
        raise ValueError("teacher grouping identities must be non-empty")
    expected = {
        "dino_regions": (16, 1024),
        "instruction_teacher": (2048,),
        "actor_teacher_log_probs": (8,),
    }
    for name, shape in expected.items():
        value = getattr(teacher, name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"teacher {name} must have shape {shape}")
        if value.requires_grad or not torch.isfinite(value).all():
            raise ValueError(f"teacher {name} must be detached and finite")
    normalized = torch.logsumexp(teacher.actor_teacher_log_probs.float(), dim=-1)
    if not torch.allclose(normalized, torch.zeros_like(normalized), atol=1e-5, rtol=0):
        raise ValueError("teacher actor log-probabilities must be normalized")


def _validate_encoded_tensors(encoded: Mapping[str, torch.Tensor]) -> None:
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise ValueError("encoded student row requires input_ids")
    unknown_student = sorted(_FORBIDDEN_STUDENT_KEYS & set(encoded))
    if unknown_student:
        raise ValueError(
            "encoded student row may not contain precomputed student tensors: "
            + ", ".join(unknown_student)
        )
    if "labels" in encoded:
        raise ValueError("state-interface-v2 student rows do not carry LM labels")
    input_ids = encoded["input_ids"]
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 1:
        raise ValueError("encoded student input_ids must be one unpadded token row")
    if input_ids.numel() < 1:
        raise ValueError("encoded student input_ids may not be empty")
    for name, value in encoded.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("encoded student inputs must be named tensors")
        if value.requires_grad:
            raise ValueError("encoded input tensors may not own an autograd graph")


def prepare_sft1_v2_row(
    record: Mapping[str, Any],
    *,
    step_index: int,
    encoded_tensors: Mapping[str, torch.Tensor],
    teacher: SFT1V2TeacherRow,
    manifest: SFT1V2Manifest,
) -> SFT1V2PreparedRow:
    """Build one fail-closed row from the current trajectory schema only."""

    if not isinstance(record, dict):
        raise ValueError("trajectory row must be a mapping")
    trajectory = RolloutTrajectory.from_record(record)
    expected_fields = set(trajectory.to_record())
    unknown = sorted(set(record) - expected_fields)
    if unknown:
        raise ValueError(f"current trajectory contains unknown field: {unknown[0]}")
    if not 0 <= step_index < trajectory.num_steps:
        raise ValueError("SFT1-v2 row must correspond to an executed nonterminal action")
    if trajectory.split not in {manifest.train_split, manifest.external_validation_split}:
        raise ValueError("trajectory split is outside the manifest train/validation boundary")
    if teacher.record_id != trajectory.record_id or teacher.step_index != step_index:
        raise ValueError("teacher row does not match trajectory record/step identity")
    _validate_teacher_row(teacher, manifest)
    _validate_encoded_tensors(encoded_tensors)

    response = require_archived_assistant_response(
        trajectory.assistant_responses[step_index],
        source="archived",
    )
    action_index = int(trajectory.action_indices[step_index])
    if not 0 <= action_index < manifest.action_count:
        raise ValueError("executed action is outside the manifest action support")
    image_path = str(trajectory.image_paths[step_index])
    original_image = Path(image_path)
    if not original_image.is_file():
        raise FileNotFoundError(f"original observation image is missing: {image_path}")
    if sha256_file(original_image) != teacher.original_image_sha256:
        raise ValueError("original observation image digest differs from teacher identity")
    movement_success = _optional_movement_success(
        trajectory,
        step_index,
        action_index,
    )
    return SFT1V2PreparedRow(
        record_id=trajectory.record_id,
        step_index=step_index,
        split=trajectory.split,
        original_image_path=image_path,
        original_image_sha256=teacher.original_image_sha256,
        image_content_group=teacher.image_content_group,
        instruction_equivalence_group=teacher.instruction_equivalence_group,
        archived_assistant_response=response,
        executed_action_index=action_index,
        movement_success=movement_success,
        encoded_tensors={name: value.detach() for name, value in encoded_tensors.items()},
        dino_regions=teacher.dino_regions.detach(),
        instruction_teacher=teacher.instruction_teacher.detach(),
        actor_teacher_log_probs=teacher.actor_teacher_log_probs.detach(),
        manifest_identity=manifest.identity,
    )


__all__ = [
    "SFT1V2PreparedRow",
    "SFT1V2TeacherRow",
    "SFT1V2TeacherTargetBuilder",
    "prepare_sft1_v2_row",
    "sha256_file",
]
