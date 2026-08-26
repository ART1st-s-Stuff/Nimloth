"""Production smoke/formal runtime wiring for the SFT1-v2 experiment."""

from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
from tempfile import NamedTemporaryFile
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch

from nimloth.training.sft1.checkpoint import export_sft1_v2_deployable
from nimloth.training.sft1.experiment_config import SFT1V2Config
from nimloth.training.sft1.data import sha256_file
from nimloth.training.sft1.driver import (
    SFT1V2DataCursor,
    SFT1V2ProductionAssembly,
    build_training_manifest,
    build_update_dataproto,
    construct_sft1_v2_production,
    deterministic_update_schedule,
    iter_schedule_updates,
    restore_training_checkpoint,
    run_sft1_v2_epochs,
    save_training_checkpoint,
)
from nimloth.training.sft1.manifest import load_sft1_v2_manifest
from nimloth.training.sft1.real_rows import index_early4_rows
from nimloth.training.sft1.teacher_cache import (
    SFT1V2TeacherCacheReader,
    inspect_teacher_cache,
)
from nimloth.training.sft1.validation import load_validation_report
from nimloth.training.sft1.validation_runtime import run_validation_epoch
from nimloth.training.verl.runtime import MixedPrecisionConfig


def _run_identity(config: SFT1V2Config, manifest_identity: str) -> str:
    return hashlib.sha256(json.dumps({
        "config_identity": config.identity,
        "manifest_identity": manifest_identity,
        "source_commit": config.source.expected_commit,
        "run_dir": config.output.run_dir,
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _backbone_args(config: SFT1V2Config) -> SimpleNamespace:
    return SimpleNamespace(
        model=config.teacher.actor_checkpoint,
        max_pixels=config.runtime.max_pixels,
        gradient_checkpointing=config.runtime.gradient_checkpointing,
        attn_implementation=config.runtime.attention_implementation,
        llm_tune="freeze",
        vision_tune="freeze",
        lora=False,
        query_tune="adapter",
        resume=False,
    )


def _seed_runtime(seed: int) -> None:
    if seed < 0:
        raise ValueError("experiment seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_production_assembly(
    config: SFT1V2Config,
    *,
    device: torch.device,
    repo_root: Path,
) -> SFT1V2ProductionAssembly:
    return construct_sft1_v2_production(
        config=config,
        backbone_args=_backbone_args(config),
        device=device,
        repo_root=repo_root,
        wrap_policy=None,
        mixed_precision=MixedPrecisionConfig(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        ),
    )


def _prepare_run_dir(path: Path, *, rank: int) -> None:
    if rank == 0:
        if path.exists():
            raise FileExistsError(f"fresh training output exists: {path}")
        path.mkdir(parents=True)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
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


def _atomic_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"immutable text artifact exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _record_step_metrics(
    directory: Path,
    *,
    epoch: int,
    global_step: int,
    metrics: Mapping[str, float],
) -> None:
    path = directory / f"step_{global_step:08d}.json"
    payload = {"epoch": epoch, "global_step": global_step, **metrics}
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("resumed step metrics differ from the durable record")
        return
    _atomic_json(path, payload)


def _materialize_step_csv(output: Path) -> None:
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output / "step_metrics").glob("step_*.json"))
    ]
    if not rows:
        raise ValueError("formal run produced no step metrics")
    fields = tuple(sorted({key for row in rows for key in row}))
    path = output / "train_step_log.csv"
    if path.exists():
        raise FileExistsError("final train_step_log.csv already exists")
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w", newline="", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_formal_training(
    config: SFT1V2Config,
    *,
    repo_root: Path,
    rank: int,
    world_size: int,
    device: torch.device,
    seed: int,
    resume_checkpoint: Path | None = None,
    metric_logger: Any | None = None,
) -> Mapping[str, Any]:
    if not config.runtime.launch_locked:
        raise PermissionError("formal training requires a launch-locked config")
    _seed_runtime(seed)
    output = Path(config.output.run_dir)
    if resume_checkpoint is None:
        _prepare_run_dir(output, rank=rank)
    elif not output.is_dir():
        raise FileNotFoundError("resume run directory is missing")
    cache_summary = inspect_teacher_cache(Path(config.cache.output_dir))
    manifest_path = Path(config.cache.output_dir) / "training_manifest.json"
    manifest = load_sft1_v2_manifest(manifest_path)
    expected_manifest = build_training_manifest(config, cache_summary)
    if manifest != expected_manifest:
        raise ValueError("training manifest differs from resolved config/cache")
    parity = Path(config.cache.output_dir) / "parity_report.json"
    if not parity.is_file():
        raise ValueError("fresh cache parity report is missing")
    rows, audit = index_early4_rows(config)
    assembly = build_production_assembly(
        config,
        device=device,
        repo_root=repo_root,
    )
    reader = SFT1V2TeacherCacheReader(
        Path(config.cache.output_dir), manifest_identity=manifest.identity
    )
    run_identity = _run_identity(config, manifest.identity)
    if rank == 0:
        metadata_path = output / "run_metadata.json"
        metadata = {
            "schema": "nimloth_sft1_state_v2_run_metadata_v1",
            "config": asdict(config),
            "config_identity": config.identity,
            "manifest_identity": manifest.identity,
            "cache_identity": cache_summary.cache_identity,
            "parity_report_sha256": sha256_file(parity),
            "run_identity": run_identity,
            "source_commit": config.source.expected_commit,
            "trainable": ["K16 query additive adapter", "fresh SharedSlotProjector", "training-only readouts"],
            "frozen": ["ID176 Qwen language body/LM head/vision", "DINO and detached teachers"],
            "interpretation_owner": "human",
            "automatic_sft2_authorization": False,
        }
        if metadata_path.exists():
            if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
                raise ValueError("resume run metadata differs from resolved contract")
        else:
            _atomic_json(metadata_path, metadata)
        readme_path = output / "README.md"
        readme_text = (
            "# SFT1-v2 early-4 report-first canary\n\n"
            "This run trains only the K16 query adapter, fresh projector, and "
            "training-only readouts. It does not authorize SFT2/WM and must be "
            "interpreted by the human after component reports.\n"
        )
        if readme_path.exists():
            if readme_path.read_text(encoding="utf-8") != readme_text:
                raise ValueError("resume run README differs from resolved contract")
        else:
            _atomic_text(readme_path, readme_text)
    resume_cursor = None
    if resume_checkpoint is not None:
        resume_cursor = restore_training_checkpoint(
            assembly,
            resume_checkpoint,
            manifest=manifest,
            config=config,
            run_identity=run_identity,
            rank=rank,
            world_size=world_size,
        )
    latest_checkpoint: dict[str, Path | None] = {"path": resume_checkpoint}
    epoch0_path = output / "validation" / "epoch_000.json"

    def checkpoint_callback(
        epoch: int,
        global_step: int,
        cursor: SFT1V2DataCursor,
    ) -> Path:
        path = output / "checkpoints" / f"epoch_{epoch:03d}_step_{global_step:08d}"
        saved = save_training_checkpoint(
            assembly,
            path,
            cursor=cursor,
            manifest=manifest,
            config=config,
            run_identity=run_identity,
            rank=rank,
            world_size=world_size,
        )
        latest_checkpoint["path"] = saved
        return saved

    def validation_callback(
        epoch: int,
        global_step: int,
        runtime_metrics: Mapping[str, float],
    ) -> tuple[Path, bool]:
        destination = output / "validation" / f"epoch_{epoch:03d}.json"
        if destination.is_file():
            existing = load_validation_report(destination)
            if (
                existing.epoch != epoch
                or existing.checkpoint_step != global_step
                or existing.config_identity != config.identity
                or existing.manifest_identity != manifest.identity
            ):
                raise ValueError("existing resume validation report identity mismatch")
            return destination, existing.safety_stop
        checkpoint = latest_checkpoint["path"]
        checkpoint_identity = (
            sha256_file(checkpoint / "control.json")
            if checkpoint is not None
            else hashlib.sha256(
                f"fresh:{config.identity}:{manifest.identity}".encode()
            ).hexdigest()
        )
        report_path, safety = run_validation_epoch(
            assembly=assembly,
            config=config,
            rows=rows,
            cache_reader=reader,
            manifest=manifest,
            repo_root=repo_root,
            rank=rank,
            world_size=world_size,
            epoch=epoch,
            checkpoint_step=global_step,
            checkpoint_identity=checkpoint_identity,
            report_path=destination,
            runtime_metrics=runtime_metrics,
            epoch0_report_path=None if epoch == 0 else epoch0_path,
        )
        if rank == 0 and metric_logger is not None:
            report = load_validation_report(report_path)
            metric_logger.log(
                {f"validation/{metric.name}": metric.value for metric in report.metrics},
                step=global_step,
            )
        return report_path, safety

    def update_callback(
        epoch: int,
        global_step: int,
        metrics: Mapping[str, float],
    ) -> None:
        if rank == 0:
            _record_step_metrics(
                output / "step_metrics",
                epoch=epoch,
                global_step=global_step,
                metrics=metrics,
            )
            if metric_logger is not None:
                metric_logger.log(dict(metrics), step=global_step)

    result = run_sft1_v2_epochs(
        assembly=assembly,
        config=config,
        rows=rows,
        cache_reader=reader,
        manifest=manifest,
        repo_root=repo_root,
        rank=rank,
        world_size=world_size,
        seed=seed,
        checkpoint_callback=checkpoint_callback,
        validation_callback=validation_callback,
        update_callback=update_callback,
        resume_cursor=resume_cursor,
    )
    if rank == 0:
        _materialize_step_csv(output)
        summary = {
            "schema": "nimloth_sft1_state_v2_formal_run_v1",
            "config_identity": config.identity,
            "manifest_identity": manifest.identity,
            "run_identity": run_identity,
            "row_audit": asdict(audit),
            "result": asdict(result),
            "interpretation_owner": "human",
            "automatic_sft2_authorization": False,
        }
        _atomic_json(output / "result.json", summary)
        return summary
    return {"rank": rank, "status": "completed"}


def _export_resume_smoke(
    assembly: SFT1V2ProductionAssembly,
    destination: Path,
    *,
    manifest_identity: str,
    config: SFT1V2Config,
    rank: int,
) -> Path | None:
    """Exercise the real full-state FSDP to restricted deployable boundary."""

    from torch.distributed.fsdp import (
        FullStateDictConfig,
        FullyShardedDataParallel as FSDP,
        StateDictType,
    )

    root = assembly.worker.root
    if not isinstance(root, FSDP):
        raise TypeError("resume-smoke export requires the real FSDP root")
    with FSDP.state_dict_type(
        root,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        full_state = root.state_dict()
    if rank == 0:
        normalized = {
            key.removeprefix("_fsdp_wrapped_module."): value
            for key, value in full_state.items()
        }
        model_prefix = "backbone.language_model."
        projector_prefix = "objective.projector."
        model_state = {
            key[len(model_prefix):]: value
            for key, value in normalized.items()
            if key.startswith(model_prefix)
        }
        projector_state = {
            key[len(projector_prefix):]: value
            for key, value in normalized.items()
            if key.startswith(projector_prefix)
        }
        if set(model_state) != set(assembly.model_state_keys):
            missing = sorted(set(assembly.model_state_keys) - set(model_state))
            unexpected = sorted(set(model_state) - set(assembly.model_state_keys))
            raise RuntimeError(
                "full FSDP actor export key mismatch: "
                f"missing={missing[:1]}, unexpected={unexpected[:1]}"
            )
        if not projector_state:
            raise RuntimeError("full FSDP state lacks projector export owner")
        export_sft1_v2_deployable(
            destination,
            actor_exporter=lambda path: assembly.loaded_backbone.backbone.save_pretrained(
                path,
                metadata={
                    "nimloth_state_interface_objective_version": config.state.objective_version,
                    "nimloth_latent_query_mode": "inject",
                    "nimloth_latent_token_count": 16,
                },
                state_dict=model_state,
            ),
            processor_exporter=lambda path: assembly.loaded_backbone.processor.save_pretrained(path),
            projector_state=projector_state,
            state_metadata={
                "manifest_identity": manifest_identity,
                "query_mode": "inject",
                "action_token_ids": list(config.teacher.action_token_ids),
                "role": "resume_smoke_export_not_model_quality_evidence",
            },
        )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    return destination if rank == 0 else None


def run_training_smoke(
    config: SFT1V2Config,
    *,
    repo_root: Path,
    rank: int,
    world_size: int,
    device: torch.device,
    seed: int,
    resume_checkpoint: Path | None = None,
) -> Mapping[str, Any]:
    """Run exactly one real update and publish an exact-resume checkpoint."""

    if not config.runtime.launch_locked:
        raise PermissionError("training smoke requires a launch-locked config")
    _seed_runtime(seed)
    output = Path(config.output.run_dir).with_name(
        f"{Path(config.output.run_dir).name}-"
        f"{'resume-smoke' if resume_checkpoint else 'smoke'}"
    )
    _prepare_run_dir(output, rank=rank)
    cache_summary = inspect_teacher_cache(Path(config.cache.output_dir))
    manifest = load_sft1_v2_manifest(
        Path(config.cache.output_dir) / "training_manifest.json"
    )
    if manifest != build_training_manifest(config, cache_summary):
        raise ValueError("smoke manifest differs from resolved config/cache")
    rows, _ = index_early4_rows(config)
    train_rows = tuple(row for row in rows if row.split == config.data.train_split)
    assembly = build_production_assembly(config, device=device, repo_root=repo_root)
    reader = SFT1V2TeacherCacheReader(
        Path(config.cache.output_dir), manifest_identity=manifest.identity
    )
    run_identity = _run_identity(config, manifest.identity)
    schedule, schedule_identity = deterministic_update_schedule(
        tuple(row.ordinal for row in train_rows),
        movement_ordinals=frozenset(
            row.ordinal for row in train_rows if row.movement_success is not None
        ),
        epoch=0,
        seed=seed,
        rank=rank,
        world_size=world_size,
        rows_per_rank_update=config.runtime.rows_per_rank_update,
    )
    consumed = 0
    global_step = 0
    if resume_checkpoint is not None:
        cursor = restore_training_checkpoint(
            assembly,
            resume_checkpoint,
            manifest=manifest,
            config=config,
            run_identity=run_identity,
            rank=rank,
            world_size=world_size,
        )
        if cursor.schedule_identity != schedule_identity:
            raise ValueError("resume-smoke schedule identity mismatch")
        consumed = cursor.consumed_rank_rows
        global_step = cursor.update_index
        schedule = schedule[consumed:]
    first = next(iter_schedule_updates(
        schedule,
        rows_per_rank_update=config.runtime.rows_per_rank_update,
    ))
    data = build_update_dataproto(
        first,
        rows_by_ordinal={row.ordinal: row for row in rows},
        padding_row=train_rows[0],
        cache_reader=reader,
        manifest=manifest,
        processor=assembly.loaded_backbone.processor,
        config=config,
        repo_root=repo_root,
    )
    update = assembly.worker.core.update(data)
    global_step += 1
    consumed += len(first)
    cursor = SFT1V2DataCursor(
        epoch=0,
        update_index=global_step,
        consumed_rank_rows=consumed,
        schedule_identity=schedule_identity,
        world_size=world_size,
        rank=rank,
    )
    checkpoint = save_training_checkpoint(
        assembly,
        output / "checkpoint",
        cursor=cursor,
        manifest=manifest,
        config=config,
        run_identity=run_identity,
        rank=rank,
        world_size=world_size,
    )
    deployable = (
        _export_resume_smoke(
            assembly,
            output / "deployable_smoke",
            manifest_identity=manifest.identity,
            config=config,
            rank=rank,
        )
        if resume_checkpoint is not None
        else None
    )
    result = {
        "kind": (
            "production_path_resume_export_smoke_not_model_evidence"
            if resume_checkpoint is not None
            else "production_path_smoke_not_model_evidence"
        ),
        "global_step": global_step,
        "checkpoint": str(checkpoint),
        "gradient_norm": update.gradient_norm,
        "micro_batch_count": update.micro_batch_count,
        "config_identity": config.identity,
        "manifest_identity": manifest.identity,
        "deployable_smoke": str(deployable) if deployable is not None else None,
    }
    if rank == 0:
        _atomic_json(output / "result.json", result)
        _atomic_text(
            output / "README.md",
            "# SFT1-v2 production-path smoke\n\n"
            "One real update plus exact checkpoint publication; resume-smoke also "
            "exercises full-state restricted export. This is not model-quality evidence.\n",
        )
    return result


__all__ = [
    "build_production_assembly",
    "run_formal_training",
    "run_training_smoke",
]
