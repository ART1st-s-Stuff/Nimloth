"""SFT1-v2 DataProto transport and deterministic padded-token packing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from nimloth.backbone.base import BackboneInputBuilder
from nimloth.backbone.qwen25vl.state_training import QwenStateTrainingBatch
from nimloth.training.sft1.data import SFT1V2PreparedRow
from nimloth.training.sft1.config import STATE_INTERFACE_OBJECTIVE_VERSION
from nimloth.training.sft1.manifest import (
    PINNED_VAGEN_COMMIT,
    PINNED_VERL_COMMIT,
    SFT1V2Manifest,
    SFT1_V2_MANIFEST_SCHEMA,
    SFT1_V2_SUPERVISION_SCHEMA,
    verify_pinned_vagen_verl_source,
)
from nimloth.training.sft1.objective import SFT1V2Targets
from nimloth.training.verl.source import require_pinned_verl_import


SFT1_V2_DATAPROTO_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SFT1V2UpdateInputs:
    student_batch: QwenStateTrainingBatch
    targets: SFT1V2Targets
    record_ids: tuple[str, ...]
    step_indices: tuple[int, ...]
    splits: tuple[str, ...]
    image_content_groups: tuple[str, ...]
    instruction_equivalence_groups: tuple[str, ...]
    token_counts: tuple[int, ...]
    manifest_identity: str

    @property
    def local_feasibility_valid_count(self) -> int:
        return int(self.targets.feasibility_label_valid.sum().item())


def _object_array(values: Sequence[Any]) -> np.ndarray:
    result = np.empty(len(values), dtype=object)
    result[:] = list(values)
    return result


def _runtime_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_pinned_dataproto_type() -> type[Any]:
    return require_pinned_verl_import(_runtime_repo_root())


def build_sft1_v2_dataproto(
    rows: Sequence[SFT1V2PreparedRow],
    *,
    manifest: SFT1V2Manifest,
) -> Any:
    """Build one strict transport batch without serializing student hidden/state."""

    verify_pinned_vagen_verl_source(_runtime_repo_root())
    if not rows:
        raise ValueError("SFT1-v2 DataProto batch must not be empty")
    if any(row.manifest_identity != manifest.identity for row in rows):
        raise ValueError("SFT1-v2 rows contain mixed teacher/manifest identities")
    allowed_splits = {manifest.train_split, manifest.external_validation_split}
    if any(row.split not in allowed_splits for row in rows):
        raise ValueError("SFT1-v2 row split is outside the manifest boundary")
    DataProto = _load_pinned_dataproto_type()

    movement_valid = torch.tensor(
        [row.movement_success is not None for row in rows],
        dtype=torch.bool,
    )
    movement_success = torch.tensor(
        [bool(row.movement_success) if row.movement_success is not None else False for row in rows],
        dtype=torch.float32,
    )
    tensors = {
        "dino_regions": torch.stack(tuple(row.dino_regions.detach().cpu() for row in rows)),
        "instruction_teacher": torch.stack(
            tuple(row.instruction_teacher.detach().cpu() for row in rows)
        ),
        "executed_action_indices": torch.tensor(
            [row.executed_action_index for row in rows],
            dtype=torch.long,
        ),
        "movement_success": movement_success,
        "feasibility_label_valid": movement_valid,
        "actor_teacher_log_probs": torch.stack(
            tuple(row.actor_teacher_log_probs.detach().cpu() for row in rows)
        ),
        "token_counts": torch.tensor([row.token_count for row in rows], dtype=torch.long),
        "step_indices": torch.tensor([row.step_index for row in rows], dtype=torch.long),
        "row_valid": torch.ones(len(rows), dtype=torch.bool),
    }
    if any(not torch.isfinite(value).all() for value in tensors.values() if value.is_floating_point()):
        raise ValueError("SFT1-v2 DataProto contains non-finite teacher tensors")
    return DataProto.from_dict(
        tensors=tensors,
        non_tensors={
            "encoded_rows": _object_array([dict(row.encoded_tensors) for row in rows]),
            "archived_assistant_responses": _object_array(
                [row.archived_assistant_response for row in rows]
            ),
            "response_sources": _object_array(["archived"] * len(rows)),
            "record_ids": _object_array([row.record_id for row in rows]),
            "splits": _object_array([row.split for row in rows]),
            "image_content_groups": _object_array(
                [row.image_content_group for row in rows]
            ),
            "instruction_equivalence_groups": _object_array(
                [row.instruction_equivalence_group for row in rows]
            ),
        },
        meta_info={
            "schema_version": SFT1_V2_DATAPROTO_SCHEMA_VERSION,
            "manifest_schema": SFT1_V2_MANIFEST_SCHEMA,
            "supervision_schema": SFT1_V2_SUPERVISION_SCHEMA,
            "objective_version": manifest.objective_version,
            "manifest_identity": manifest.identity,
            "query_count": manifest.query_count,
            "action_count": manifest.action_count,
            "action_token_ids": manifest.action_token_ids,
            "vagen_commit": manifest.vagen_commit,
            "verl_commit": manifest.verl_commit,
        },
    )


def sft1_v2_micro_batches(
    data: Any,
    *,
    max_padded_tokens: int,
    max_rows: int,
) -> tuple[Any, ...]:
    """First-fit-decreasing packing by exact padded-token cost; never truncate."""

    if max_padded_tokens < 1 or max_rows < 1:
        raise ValueError("SFT1-v2 token and row budgets must be positive")
    counts = tuple(int(value) for value in data.batch["token_counts"].tolist())
    if not counts or any(value < 1 for value in counts):
        raise ValueError("SFT1-v2 token counts must be positive")
    oversized = [(index, value) for index, value in enumerate(counts) if value > max_padded_tokens]
    if oversized:
        raise ValueError(
            "SFT1-v2 row exceeds max_padded_tokens: "
            + ", ".join(f"row{index}={value}" for index, value in oversized)
        )
    groups: list[list[int]] = []
    for index in sorted(range(len(counts)), key=lambda item: (-counts[item], item)):
        for group in groups:
            candidate = (*group, index)
            cost = max(counts[item] for item in candidate) * len(candidate)
            if len(candidate) <= max_rows and cost <= max_padded_tokens:
                group.append(index)
                break
        else:
            groups.append([index])
    return tuple(data[torch.tensor(group, dtype=torch.long)] for group in groups)


def sft1_v2_update_inputs(
    data: Any,
    *,
    input_builder: BackboneInputBuilder,
) -> SFT1V2UpdateInputs:
    """Validate and collate one worker-local DataProto chunk."""

    expected_meta = {
        "schema_version": SFT1_V2_DATAPROTO_SCHEMA_VERSION,
        "manifest_schema": SFT1_V2_MANIFEST_SCHEMA,
        "supervision_schema": SFT1_V2_SUPERVISION_SCHEMA,
        "objective_version": STATE_INTERFACE_OBJECTIVE_VERSION,
        "query_count": 16,
        "action_count": 8,
        "vagen_commit": PINNED_VAGEN_COMMIT,
        "verl_commit": PINNED_VERL_COMMIT,
    }
    for name, expected in expected_meta.items():
        if data.meta_info.get(name) != expected:
            raise ValueError(f"SFT1-v2 DataProto {name} mismatch")
    manifest_identity = data.meta_info.get("manifest_identity")
    if not isinstance(manifest_identity, str) or len(manifest_identity) != 64:
        raise ValueError("SFT1-v2 DataProto has no manifest identity")
    action_token_ids = data.meta_info.get("action_token_ids")
    if (
        not isinstance(action_token_ids, (list, tuple))
        or len(action_token_ids) != 8
        or len(set(action_token_ids)) != 8
    ):
        raise ValueError("SFT1-v2 DataProto action token identity is invalid")
    required_tensors = {
        "dino_regions",
        "instruction_teacher",
        "executed_action_indices",
        "movement_success",
        "feasibility_label_valid",
        "actor_teacher_log_probs",
        "token_counts",
        "step_indices",
        "row_valid",
    }
    missing = sorted(required_tensors - set(data.batch.keys()))
    if missing:
        raise ValueError("SFT1-v2 DataProto is missing tensor: " + missing[0])
    required_non_tensors = {
        "encoded_rows",
        "archived_assistant_responses",
        "response_sources",
        "record_ids",
        "splits",
        "image_content_groups",
        "instruction_equivalence_groups",
    }
    missing_non_tensor = sorted(required_non_tensors - set(data.non_tensor_batch))
    if missing_non_tensor:
        raise ValueError("SFT1-v2 DataProto is missing metadata: " + missing_non_tensor[0])
    batch_size = len(data)
    encoded_rows = tuple(data.non_tensor_batch["encoded_rows"])
    if len(encoded_rows) != batch_size:
        raise ValueError("encoded student rows do not align with DataProto")
    backbone_batch = input_builder.collate_encoded(
        list(encoded_rows),
        include_labels=False,
    )
    groups = tuple(str(value) for value in data.non_tensor_batch["instruction_equivalence_groups"])
    group_to_id = {name: index for index, name in enumerate(sorted(set(groups)))}
    targets = SFT1V2Targets(
        dino_regions=data.batch["dino_regions"],
        instruction_teacher=data.batch["instruction_teacher"],
        instruction_group_ids=torch.tensor(
            [group_to_id[group] for group in groups],
            dtype=torch.long,
            device=data.batch["instruction_teacher"].device,
        ),
        sample_valid=data.batch["row_valid"],
        executed_action_indices=data.batch["executed_action_indices"],
        movement_success=data.batch["movement_success"],
        feasibility_label_valid=data.batch["feasibility_label_valid"],
        actor_teacher_log_probs=data.batch["actor_teacher_log_probs"],
    )
    return SFT1V2UpdateInputs(
        student_batch=QwenStateTrainingBatch(
            backbone_batch=backbone_batch,
            archived_assistant_responses=tuple(
                str(value) for value in data.non_tensor_batch["archived_assistant_responses"]
            ),
            response_sources=tuple(
                str(value) for value in data.non_tensor_batch["response_sources"]
            ),
        ),
        targets=targets,
        record_ids=tuple(str(value) for value in data.non_tensor_batch["record_ids"]),
        step_indices=tuple(int(value) for value in data.batch["step_indices"].tolist()),
        splits=tuple(str(value) for value in data.non_tensor_batch["splits"]),
        image_content_groups=tuple(
            str(value) for value in data.non_tensor_batch["image_content_groups"]
        ),
        instruction_equivalence_groups=groups,
        token_counts=tuple(int(value) for value in data.batch["token_counts"].tolist()),
        manifest_identity=manifest_identity,
    )


__all__ = [
    "SFT1V2UpdateInputs",
    "SFT1_V2_DATAPROTO_SCHEMA_VERSION",
    "build_sft1_v2_dataproto",
    "sft1_v2_micro_batches",
    "sft1_v2_update_inputs",
]
