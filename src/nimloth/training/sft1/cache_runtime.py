"""Production fresh-teacher cache execution and parity audit."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch

from nimloth.backbone.dino_grid import (
    DINOV2_LARGE_IDENTITY,
    FrozenDINOGridTargets,
)
from nimloth.backbone.qwen25vl.factory import load_backbone
from nimloth.latent import LatentActionTokens
from nimloth.training.sft1.experiment_config import SFT1V2Config
from nimloth.training.sft1.data import sha256_file
from nimloth.training.sft1.real_rows import index_early4_rows, render_early4_row
from nimloth.training.sft1.teacher_cache import (
    SFT1V2CacheSummary,
    SFT1V2TeacherCacheIdentity,
    SFT1V2TeacherCacheReader,
    finalize_teacher_cache,
    prepare_teacher_cache_shard,
)
from nimloth.training.sft1.teachers import FreshID176DINOTeacher


PARITY_REPORT_SCHEMA = "nimloth_sft1_state_v2_fresh_cache_parity_v1"


def _json_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_teacher_cache_identity(
    config: SFT1V2Config,
    *,
    repo_root: Path,
) -> SFT1V2TeacherCacheIdentity:
    renderer = Path(repo_root) / "src/nimloth/backbone/qwen25vl/batch.py"
    if not renderer.is_file():
        raise FileNotFoundError("Qwen prompt renderer source is missing")
    dino = {
        "source": config.teacher.dino_source,
        "revision": config.teacher.dino_revision,
        "processor_fingerprint": config.teacher.dino_processor_fingerprint,
        "hidden_size": 1024,
        "grid_tokens": 16,
    }
    return SFT1V2TeacherCacheIdentity(
        source_commit=config.source.expected_commit,
        actor_checkpoint_sha256=config.teacher.actor_completion_sha256,
        actor_config_sha256=config.teacher.actor_config_sha256,
        actor_model_index_sha256=config.teacher.actor_model_index_sha256,
        actor_action_head_sha256=config.teacher.actor_action_head_sha256,
        actor_shards_sha256=config.teacher.actor_model_shards_sha256,
        processor_sha256=config.teacher.processor_sha256,
        tokenizer_sha256=config.teacher.tokenizer_sha256,
        chat_template_sha256=config.teacher.prompt_template_sha256,
        prompt_renderer_sha256=sha256_file(renderer),
        token_table_sha256=config.teacher.token_table_sha256,
        query_action_contract_sha256=_json_digest({
            "query_mode": "inject",
            "query_count": 16,
            "action_token_ids": list(config.teacher.action_token_ids),
        }),
        dino_checkpoint_sha256=_json_digest(dino),
        dino_processor_sha256=_json_digest({
            "processor_fingerprint": config.teacher.dino_processor_fingerprint
        }),
        train_trajectory_sha256=config.data.train_sha256,
        validation_trajectory_sha256=config.data.validation_sha256,
    )


def _teacher_backbone_args(config: SFT1V2Config) -> SimpleNamespace:
    return SimpleNamespace(
        model=config.teacher.actor_checkpoint,
        max_pixels=config.runtime.max_pixels,
        gradient_checkpointing=False,
        attn_implementation=config.runtime.attention_implementation,
        llm_tune="freeze",
        vision_tune="freeze",
        lora=False,
        query_tune="freeze",
        resume=False,
    )


def load_fresh_teacher(
    config: SFT1V2Config,
    *,
    device: torch.device,
) -> tuple[FreshID176DINOTeacher, Any]:
    loaded = load_backbone(
        _teacher_backbone_args(config),
        device=device,
        latent_token_count=16,
        model_parallel_size=1,
        resume_dir=None,
        resume_state_path=None,
    )
    actual_action_ids = tuple(
        loaded.token_id_map[token]
        for token in LatentActionTokens().action_tokens
    )
    if actual_action_ids != config.teacher.action_token_ids:
        raise ValueError("loaded ID176 action token table differs from resolved config")
    if loaded.query_adapter is not None:
        raise RuntimeError("frozen cache teacher must not install a student query adapter")
    if (
        DINOV2_LARGE_IDENTITY.source != config.teacher.dino_source
        or DINOV2_LARGE_IDENTITY.revision != config.teacher.dino_revision
        or DINOV2_LARGE_IDENTITY.processor_fingerprint
        != config.teacher.dino_processor_fingerprint
    ):
        raise ValueError("resolved DINO identity differs from source-owned identity")
    dino = FrozenDINOGridTargets.from_pretrained(
        DINOV2_LARGE_IDENTITY,
        device=device,
        dtype=torch.bfloat16,
        grid_size=4,
        batch_size=config.runtime.teacher_batch_size,
    )
    teacher = FreshID176DINOTeacher(
        qwen_model=loaded.backbone.model,
        dino=dino,
        action_token_ids=actual_action_ids,
        pad_token_id=loaded.processor.tokenizer.pad_token_id,
        device=device,
    )
    return teacher, loaded.processor


def generate_teacher_cache(
    config: SFT1V2Config,
    *,
    repo_root: Path,
    rank: int,
    world_size: int,
    device: torch.device,
) -> SFT1V2CacheSummary | None:
    """Generate rank-owned cache shards; rank zero finalizes after a barrier."""

    if not config.runtime.launch_locked:
        raise PermissionError("fresh cache generation requires a launch-locked config")
    if world_size != config.runtime.world_size or not 0 <= rank < world_size:
        raise ValueError("cache rank/world size differs from resolved config")
    rows, audit = index_early4_rows(config)
    if len(rows) != config.selection.train_rows + config.selection.raw_validation_rows:
        raise ValueError("fresh cache row count differs from approved early-4 audit")
    identity = build_teacher_cache_identity(config, repo_root=repo_root)
    teacher, processor = load_fresh_teacher(config, device=device)
    output = Path(config.cache.output_dir)
    for shard_index in range(rank, config.cache.shard_count, world_size):
        prepare_teacher_cache_shard(
            output,
            rows,
            shard_index=shard_index,
            shard_count=config.cache.shard_count,
            identity=identity,
            teacher=teacher,
            teacher_batch_size=config.runtime.teacher_batch_size,
            render_row=lambda row: render_early4_row(
                row,
                processor=processor,
                max_length=config.runtime.max_sequence_length,
            ),
        )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    if rank != 0:
        return None
    audit_path = output / "row_audit.json"
    if audit_path.exists():
        raise FileExistsError("fresh cache row audit already exists")
    _atomic_json(audit_path, asdict(audit))
    return finalize_teacher_cache(
        output,
        identity=identity,
        shard_count=config.cache.shard_count,
        expected_row_count=len(rows),
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists():
        raise FileExistsError(f"immutable parity report exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return sha256_file(path)


def audit_fresh_cache_parity(
    config: SFT1V2Config,
    *,
    manifest_identity: str,
    output_path: Path,
) -> str:
    """Compare fresh targets to ID60/ID192 without importing old targets."""

    reader = SFT1V2TeacherCacheReader(
        Path(config.cache.output_dir),
        manifest_identity=manifest_identity,
    )
    metadata_path = Path(config.cache.parity_dino_path).with_name(
        "frozen_state_cache_metadata.json"
    )
    if not metadata_path.is_file():
        raise FileNotFoundError("ID60 parity metadata is missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(config.cache.parity_dino_path, allow_pickle=False) as old_dino, np.load(
        config.cache.parity_instruction_path, allow_pickle=False
    ) as old_instruction:
        transition_current = old_dino["transition_current_index"]
        transition_record = old_dino["transition_record_index"]
        state_step = old_dino["state_step_index"]
        dino = old_dino["dino"]
        instruction = old_instruction["instruction_embedding"]
        records = metadata.get("records")
        if not isinstance(records, list) or len(instruction) != len(records):
            raise ValueError("legacy parity record/instruction identities are misaligned")
        if len(transition_current) != len(transition_record):
            raise ValueError("legacy parity transition identities are misaligned")
        legacy_rows: dict[tuple[str, int], tuple[int, int]] = {}
        for current_value, record_value in zip(
            transition_current, transition_record, strict=True
        ):
            current_index = int(current_value)
            record_index = int(record_value)
            if (
                not 0 <= current_index < len(state_step)
                or not 0 <= record_index < len(records)
            ):
                raise ValueError("legacy parity transition index is out of bounds")
            record_id = records[record_index].get("record_id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError("legacy parity record identity is missing")
            key = (record_id, int(state_step[current_index]))
            if key in legacy_rows:
                raise ValueError("legacy parity record/step identity is duplicated")
            legacy_rows[key] = (current_index, record_index)

        dino_sq = instruction_sq = 0.0
        dino_max = instruction_max = 0.0
        count = reader.summary.row_count
        for ordinal in range(count):
            fresh = reader.load(ordinal)
            key = (fresh.record_id, int(fresh.step_index))
            try:
                current_index, record_index = legacy_rows[key]
            except KeyError as error:
                raise ValueError(
                    "fresh parity row has no legacy record/step identity"
                ) from error
            old_dino_row = torch.from_numpy(
                np.asarray(dino[current_index])
            ).float()
            old_instruction_row = torch.from_numpy(
                np.asarray(instruction[record_index])
            ).float()
            dino_diff = fresh.dino_regions.float() - old_dino_row
            instruction_diff = fresh.instruction_teacher.float() - old_instruction_row
            dino_sq += float(dino_diff.square().mean().item())
            instruction_sq += float(instruction_diff.square().mean().item())
            dino_max = max(dino_max, float(dino_diff.abs().max().item()))
            instruction_max = max(
                instruction_max, float(instruction_diff.abs().max().item())
            )
    report = {
        "schema": PARITY_REPORT_SCHEMA,
        "cache_identity": reader.summary.cache_identity,
        "row_count": count,
        "id60_dino_reference_sha256": config.cache.parity_dino_sha256,
        "id192_instruction_reference_sha256": config.cache.parity_instruction_sha256,
        "dino_rmse_mean": (dino_sq / count) ** 0.5,
        "dino_max_abs": dino_max,
        "instruction_rmse_mean": (instruction_sq / count) ** 0.5,
        "instruction_max_abs": instruction_max,
        "role": "parity_evidence_only_not_training_target",
    }
    return _atomic_json(Path(output_path), report)


__all__ = [
    "PARITY_REPORT_SCHEMA",
    "audit_fresh_cache_parity",
    "build_teacher_cache_identity",
    "generate_teacher_cache",
    "load_fresh_teacher",
]
