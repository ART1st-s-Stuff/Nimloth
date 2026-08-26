"""Explicit immutable resolution of the launch-locked SFT1-v2 config."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping

from nimloth.config import load_yaml_config
from nimloth.training.sft1.experiment_config import SFT1V2Config, parse_sft1_v2_config


@dataclass(frozen=True)
class SFT1V2LaunchResolution:
    repo: str
    expected_commit: str
    interpreter: str
    cache_output_dir: str
    run_dir: str
    wandb_run_name: str
    wandb_run_id: str
    minimum_free_bytes: int
    processor_sha256: str
    tokenizer_sha256: str
    prompt_template_sha256: str
    token_table_sha256: str
    world_size: int
    max_sequence_length: int
    max_padded_tokens: int
    max_rows_per_micro_batch: int
    rows_per_rank_update: int
    teacher_batch_size: int
    checkpoint_cadence_steps: int


def resolve_launch_config(
    template_path: Path,
    output_path: Path,
    resolution: SFT1V2LaunchResolution,
) -> SFT1V2Config:
    """Apply every launch-time value explicitly and publish JSON atomically."""

    raw = load_yaml_config(template_path)
    if not isinstance(raw, dict):
        raise ValueError("launch config template must be a mapping")
    payload: dict[str, Any] = json.loads(json.dumps(raw))
    payload["source"].update({
        "repo": resolution.repo,
        "expected_commit": resolution.expected_commit,
        "interpreter": resolution.interpreter,
    })
    payload["teacher"].update({
        "processor_sha256": resolution.processor_sha256,
        "tokenizer_sha256": resolution.tokenizer_sha256,
        "prompt_template_sha256": resolution.prompt_template_sha256,
        "token_table_sha256": resolution.token_table_sha256,
    })
    payload["cache"]["output_dir"] = resolution.cache_output_dir
    payload["runtime"].update({
        "world_size": resolution.world_size,
        "max_sequence_length": resolution.max_sequence_length,
        "max_padded_tokens": resolution.max_padded_tokens,
        "max_rows_per_micro_batch": resolution.max_rows_per_micro_batch,
        "rows_per_rank_update": resolution.rows_per_rank_update,
        "teacher_batch_size": resolution.teacher_batch_size,
        "launch_locked": True,
    })
    payload["checkpoint"]["cadence_steps"] = resolution.checkpoint_cadence_steps
    payload["output"].update({
        "run_dir": resolution.run_dir,
        "wandb_run_name": resolution.wandb_run_name,
        "wandb_run_id": resolution.wandb_run_id,
        "minimum_free_bytes": resolution.minimum_free_bytes,
    })
    config = parse_sft1_v2_config(payload)
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"resolved launch config already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return config


__all__ = ["SFT1V2LaunchResolution", "resolve_launch_config"]
