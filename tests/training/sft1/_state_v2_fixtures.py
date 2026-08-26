from __future__ import annotations

import math
from pathlib import Path

import torch

from nimloth.agent import NimlothPromptTemplate
from nimloth.rollout import RolloutTrajectory
from nimloth.rollout.record_format import STEP_REWARD_PROVENANCE
from nimloth.training.sft1.data import (
    SFT1V2PreparedRow,
    SFT1V2TeacherRow,
    prepare_sft1_v2_row,
    sha256_file,
)
from nimloth.training.sft1.manifest import (
    PINNED_VAGEN_COMMIT,
    PINNED_VERL_COMMIT,
    SFT1V2Manifest,
    SFT1_V2_MANIFEST_SCHEMA,
    SFT1_V2_SUPERVISION_SCHEMA,
    parse_sft1_v2_manifest,
)
from nimloth.training.sft1.config import STATE_INTERFACE_OBJECTIVE_VERSION


def manifest_raw() -> dict[str, object]:
    digest = "a" * 64
    return {
        "schema": SFT1_V2_MANIFEST_SCHEMA,
        "objective_version": STATE_INTERFACE_OBJECTIVE_VERSION,
        "supervision_schema": SFT1_V2_SUPERVISION_SCHEMA,
        "vagen_commit": PINNED_VAGEN_COMMIT,
        "verl_commit": PINNED_VERL_COMMIT,
        "actor_checkpoint_sha256": digest,
        "processor_sha256": "b" * 64,
        "prompt_template_sha256": "c" * 64,
        "token_table_sha256": "d" * 64,
        "dino_identity_sha256": "e" * 64,
        "trajectory_sha256": "f" * 64,
        "teacher_cache_sha256": "1" * 64,
        "latent_query_mode": "inject",
        "query_count": 16,
        "action_count": 8,
        "action_token_ids": list(range(100, 108)),
        "train_split": "train",
        "external_validation_split": "external_validation",
    }


def manifest() -> SFT1V2Manifest:
    return parse_sft1_v2_manifest(manifest_raw())


def trajectory_record(
    tmp_path: Path,
    *,
    record_id: str = "row-1",
    split: str = "train",
    action_index: int = 0,
    feedback: str = "Last action is executed successfully.",
) -> tuple[dict[str, object], Path]:
    prompt = NimlothPromptTemplate(latent_token_count=16, action_count=8)
    before = tmp_path / f"{record_id}-before.png"
    after = tmp_path / f"{record_id}-after.png"
    before.write_bytes(f"before-{record_id}".encode())
    after.write_bytes(f"after-{record_id}".encode())
    response = prompt.assistant_response(
        action_index,
        thought="The observation supports this executed action.",
    )
    trajectory = RolloutTrajectory(
        record_id=record_id,
        reward_provenance=STEP_REWARD_PROVENANCE,
        image_paths=[str(before), str(after)],
        action_indices=[action_index],
        action_names=["moveahead"],
        action_log_probs=[[-math.log(8.0)] * 8],
        instruction="Find the target object.",
        rewards=[0.0],
        terminated=True,
        split=split,
        system_prompt="Navigate safely.",
        observation_texts=[
            "Human Instruction: Find the target object.\n<image>",
            f"After your action. The environment feedback is: {feedback}\n<image>",
        ],
        assistant_responses=[response],
        terminal_assistant_prefix=prompt.assistant_prefix(
            thought="This is the real terminal thought."
        ),
        prompt_template_spec=prompt.spec,
    )
    return trajectory.to_record(), before


def teacher_row(
    image_path: Path,
    *,
    record_id: str = "row-1",
    manifest_value: SFT1V2Manifest | None = None,
    instruction_group: str = "find-target",
    image_group: str = "image-group-1",
) -> SFT1V2TeacherRow:
    bound = manifest_value or manifest()
    return SFT1V2TeacherRow(
        manifest_identity=bound.identity,
        record_id=record_id,
        step_index=0,
        original_image_sha256=sha256_file(image_path),
        image_content_group=image_group,
        instruction_equivalence_group=instruction_group,
        dino_regions=torch.randn(16, 1024),
        instruction_teacher=torch.randn(2048),
        actor_teacher_log_probs=torch.log_softmax(torch.randn(8), dim=-1),
    )


def prepared_row(
    tmp_path: Path,
    *,
    record_id: str = "row-1",
    split: str = "train",
    action_index: int = 0,
    feedback: str = "Last action is executed successfully.",
    token_count: int = 6,
    instruction_group: str = "find-target",
    image_group: str = "image-group-1",
) -> SFT1V2PreparedRow:
    bound = manifest()
    record, image = trajectory_record(
        tmp_path,
        record_id=record_id,
        split=split,
        action_index=action_index,
        feedback=feedback,
    )
    return prepare_sft1_v2_row(
        record,
        step_index=0,
        encoded_tensors={
            "input_ids": torch.arange(token_count, dtype=torch.long),
            "attention_mask": torch.ones(token_count, dtype=torch.long),
        },
        teacher=teacher_row(
            image,
            record_id=record_id,
            manifest_value=bound,
            instruction_group=instruction_group,
            image_group=image_group,
        ),
        manifest=bound,
    )
