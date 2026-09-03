"""WS8 production entry for the authoritative update6420 Query-State cache.

This owner constructs the production SFT1 root, restores model tensors only, freezes
it recursively, replays the locked Formal38 archived-response rows, and commits a
schema-distinct unsafe/nondeployable cache.  It never constructs an optimizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from nimloth.backbone.qwen25vl.factory import build_input_builder, load_backbone
from nimloth.training.reconstruction.forensic_query_state_production import (
    FORMAL38_ROW_INDEX_IDENTITY,
    Formal38ForensicStateExtractor,
    _dtype,
    _git,
    _source_contract,
)
from nimloth.training.reconstruction.update6420_forensic_comparison import (
    LOCKED_UPDATE6420_EXPECTED,
    _restore_authenticated_update6420_model_only,
    validate_checkpoint_evidence,
)
from nimloth.training.reconstruction.update6420_query_state_cache import (
    BASELINE_CACHE_PATH,
    _write_rank_payload,
    load_locked_baseline_rows,
    publish_update6420_cache_from_rank_payloads,
)
from nimloth.training.sft1.query_state_runtime import (
    construct_query_state_production_root,
)
from nimloth.training.sft1.query_state_training_config import (
    QueryStateTrainingConfig,
    parse_query_state_training_config,
)
from nimloth.training.sft1.real_rows import index_early4_rows
from nimloth.training.verl.runtime import (
    MixedPrecisionConfig,
    assert_complete_module_device,
    wrap_complete_fsdp,
)

UPDATE6420_PRODUCTION_CONFIG_SCHEMA = "nimloth_update6420_unsafe_query_state_cache_production_v1"
UPDATE6420_CPU_PREFLIGHT_SCHEMA = "nimloth_update6420_query_state_cpu_preflight_v1"
_ARCHIVED_UPDATE6420_COMPATIBILITY_ENVELOPE = (
    "archived_update6420_disabled_execution_migration_v1"
)
_ARCHIVED_UPDATE6420_TOP_LEVEL_FIELDS = frozenset({
    "schema", "mode", "lifecycle", "source", "data", "model", "objective",
    "optimizer", "runtime", "schedule", "early_stopping", "validation",
    "output", "resources", "authorization", "initialization", "tracking",
    "environment", "forensic_fork", "command", "artifacts",
})
_DISABLED_EXECUTION_MIGRATION_ENVELOPE = {
    "enabled": False,
    "anchor_run_identity": "disabled",
    "anchor_source_commit": "disabled",
    "anchor_source_manifest_path": "disabled",
    "anchor_source_manifest_identity": "disabled",
    "anchor_partition": "disabled",
    "prior_process_path": "disabled",
    "prior_process_sha256": "disabled",
    "anchor_checkpoint_path": "disabled",
    "anchor_control_sha256": "disabled",
    "anchor_index_path": "disabled",
    "anchor_index_sha256": "disabled",
    "execution_source_commit": "disabled",
    "execution_source_manifest_path": "disabled",
    "execution_source_manifest_identity": "disabled",
    "execution_partition": "disabled",
    "approval_sha256": "disabled",
}


@dataclass(frozen=True)
class Update6420ProductionConfig:
    integrated_repo_root: Path
    integrated_source_commit: str
    checkpoint_evidence_path: Path
    baseline_cache_path: Path
    output_path: Path
    backend: str
    world_size: int
    max_restarts: int


@dataclass(frozen=True)
class ArchivedUpdate6420ResolvedConfig:
    """Strict current-parser view without replacing authoritative run provenance."""

    config: QueryStateTrainingConfig
    compatibility_envelope: str
    normalized_parser_identity: str
    authoritative_run_identity: str


def _absolute(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError(f"{field} must be an explicit absolute path")
    return Path(value)


def parse_update6420_production_config(raw: Mapping[str, Any]) -> Update6420ProductionConfig:
    required = {"schema", "integrated_source", "checkpoint_evidence_path", "baseline_cache_path", "output_path", "torchrun"}
    if not isinstance(raw, Mapping) or set(raw) != required or raw.get("schema") != UPDATE6420_PRODUCTION_CONFIG_SCHEMA:
        raise ValueError("update6420 production config schema/fields mismatch")
    source, torchrun = raw.get("integrated_source"), raw.get("torchrun")
    if not isinstance(source, Mapping) or set(source) != {"repo_root", "commit"}:
        raise ValueError("update6420 integrated source fields mismatch")
    if not isinstance(torchrun, Mapping) or set(torchrun) != {"backend", "world_size", "max_restarts"}:
        raise ValueError("update6420 torchrun fields mismatch")
    commit = source.get("commit")
    if not isinstance(commit, str) or len(commit) != 40 or set(commit) - frozenset("0123456789abcdef"):
        raise ValueError("update6420 integrated source commit must be a Git SHA")
    if torchrun != {"backend": "nccl", "world_size": 8, "max_restarts": 0}:
        raise ValueError("update6420 producer requires exact WS8 NCCL and max-restarts=0")
    baseline = _absolute(raw["baseline_cache_path"], field="baseline_cache_path")
    if baseline != BASELINE_CACHE_PATH:
        raise ValueError("update6420 producer requires the locked Formal38 Stage B cache")
    return Update6420ProductionConfig(
        integrated_repo_root=_absolute(source["repo_root"], field="integrated_source.repo_root"),
        integrated_source_commit=commit,
        checkpoint_evidence_path=_absolute(raw["checkpoint_evidence_path"], field="checkpoint_evidence_path"),
        baseline_cache_path=baseline,
        output_path=_absolute(raw["output_path"], field="output_path"),
        backend="nccl", world_size=8, max_restarts=0,
    )


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a mapping")
    return value


def _read_hash_bound_json(
    path: Path, *, expected_sha256: str, label: str
) -> dict[str, Any]:
    """Parse exactly the bytes whose immutable owner hash was authenticated."""

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    try:
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError(f"{label} hash changed after owner authentication")
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a mapping")
    return value


def _verify_source(config: Update6420ProductionConfig) -> None:
    root = config.integrated_repo_root.resolve()
    if root != config.integrated_repo_root or _git(root, "rev-parse", "--show-toplevel") != str(root):
        raise ValueError("update6420 integrated source root is not canonical")
    if _git(root, "rev-parse", "HEAD") != config.integrated_source_commit:
        raise ValueError("update6420 integrated source commit drift")
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("update6420 integrated source must be clean")


def parse_archived_update6420_resolved_config(
    raw: Mapping[str, Any],
    *,
    authoritative_run_identity: str,
    expected_source_commit: str,
) -> ArchivedUpdate6420ResolvedConfig:
    """Apply the sole named compatibility envelope to the exact archived shape."""

    if not isinstance(raw, Mapping) or set(raw) != _ARCHIVED_UPDATE6420_TOP_LEVEL_FIELDS:
        raise ValueError(
            "archived update6420 config must have the exact historical top-level shape "
            "with execution_migration absent"
        )
    if (
        not isinstance(authoritative_run_identity, str)
        or len(authoritative_run_identity) != 64
        or set(authoritative_run_identity) - frozenset("0123456789abcdef")
    ):
        raise ValueError("authoritative update6420 run identity is malformed")
    compatible = deepcopy(dict(raw))
    compatible["execution_migration"] = deepcopy(
        _DISABLED_EXECUTION_MIGRATION_ENVELOPE
    )
    parsed = parse_query_state_training_config(compatible)
    if parsed.mode != "visual_only_forensic_fork":
        raise ValueError("archived update6420 config is not the visual-fork producer")
    if parsed.source["commit"] != expected_source_commit:
        raise ValueError("archived update6420 config source commit is unbound")
    return ArchivedUpdate6420ResolvedConfig(
        config=parsed,
        compatibility_envelope=_ARCHIVED_UPDATE6420_COMPATIBILITY_ENVELOPE,
        normalized_parser_identity=parsed.identity,
        authoritative_run_identity=authoritative_run_identity,
    )


def load_archived_update6420_resolved_config(
    evidence: Mapping[str, Any],
) -> ArchivedUpdate6420ResolvedConfig:
    """Authenticate immutable owners before reading and adapting the archived config."""

    validated = validate_checkpoint_evidence(evidence)
    resolved_path = Path(str(validated["files"]["resolved_config"]))
    raw = _read_hash_bound_json(
        resolved_path,
        expected_sha256=str(validated["resolved_config_sha256"]),
        label="update6420 resolved config",
    )
    return parse_archived_update6420_resolved_config(
        raw,
        authoritative_run_identity=str(validated["run_identity"]),
        expected_source_commit=str(validated["source_commit"]),
    )


def preflight_update6420_producer(
    config: Update6420ProductionConfig,
) -> Mapping[str, Any]:
    """CPU-only immutable owner/config compatibility gate; no output is created."""

    _verify_source(config)
    evidence = _read_json(
        config.checkpoint_evidence_path,
        label="update6420 checkpoint evidence",
    )
    archived = load_archived_update6420_resolved_config(evidence)
    return {
        "schema": UPDATE6420_CPU_PREFLIGHT_SCHEMA,
        "compatibility_envelope": archived.compatibility_envelope,
        "normalized_parser_identity": archived.normalized_parser_identity,
        "authoritative_run_identity": archived.authoritative_run_identity,
        "actor_unsafe": True,
        "deployable": False,
    }


def _backbone_args(resolved: Any) -> SimpleNamespace:
    return SimpleNamespace(
        model=resolved.initialization["actor_checkpoint"],
        max_pixels=resolved.runtime["max_pixels"],
        gradient_checkpointing=resolved.runtime["gradient_checkpointing"],
        attn_implementation=resolved.runtime["attention_implementation"],
        llm_tune=resolved.model["llm_tune"], vision_tune=resolved.model["vision_tune"],
        query_tune=resolved.model["query_tune"], lora=False, resume=False,
    )


def construct_update6420_producer(
    config: Update6420ProductionConfig,
    *,
    evidence: Mapping[str, Any],
    rank: int,
    device: torch.device,
) -> tuple[Formal38ForensicStateExtractor, tuple[Any, ...], tuple[dict[str, Any], ...]]:
    """Construct the real visual-fork root and bind native source rows to baseline order."""

    _verify_source(config)
    archived = load_archived_update6420_resolved_config(evidence)
    resolved = archived.config
    if (
        str(resolved.source["commit"]) != LOCKED_UPDATE6420_EXPECTED["source_commit"]
        or int(resolved.resources["world_size"]) != 8
        or resolved.runtime["fsdp_sharding"] != "full_shard"
        or resolved.runtime["fsdp_use_orig_params"] is not True
    ):
        raise ValueError("update6420 resolved visual-fork model/runtime identity mismatch")
    loaded = load_backbone(_backbone_args(resolved), device=device, latent_token_count=16, model_parallel_size=1, resume_dir=None, resume_state_path=None)
    root = construct_query_state_production_root(loaded).root
    root.to(device)
    assert_complete_module_device(root, device)
    wrapped = wrap_complete_fsdp(
        root, device=device, wrap_policy=resolved.runtime["fsdp_wrap_policy"],
        mixed_precision=MixedPrecisionConfig(
            param_dtype=_dtype(resolved.runtime["model_dtype"]),
            reduce_dtype=_dtype(resolved.runtime["model_dtype"]),
            buffer_dtype=_dtype(resolved.runtime["model_dtype"]),
        ), repo_root=config.integrated_repo_root,
    )
    _restore_authenticated_update6420_model_only(
        root=wrapped, rank=rank,
        checkpoint_path=Path(str(evidence["files"]["control"])).parent,
    )
    builder = build_input_builder(
        loaded, max_length=int(resolved.runtime["max_sequence_length"]),
        latent_token_count=16, mask_latent_query_labels=True,
    )
    source = _source_contract(resolved, row_index_identity=FORMAL38_ROW_INDEX_IDENTITY)
    native_rows, audit = index_early4_rows(source, enforce_approved_counts=False)
    if (audit.train_rows, audit.raw_validation_rows, audit.external_validation_rows) != (12_836, 1_420, 1_413):
        raise ValueError("update6420 live source audit counts differ from the locked baseline")
    by_identity = {row.identity: row for row in native_rows}
    baseline_rows = load_locked_baseline_rows(config.baseline_cache_path)
    ordered_native = tuple(by_identity.get(str(row["row_identity"])) for row in baseline_rows)
    if any(row is None for row in ordered_native) or len({row.identity for row in ordered_native if row is not None}) != 14_249:
        raise ValueError("update6420 source cannot reproduce every ordered baseline row")
    return (
        Formal38ForensicStateExtractor(
            root=wrapped, processor=loaded.processor, input_builder=builder,
            max_length=int(resolved.runtime["max_sequence_length"]),
        ),
        ordered_native,
        baseline_rows,
    )


def _validated_cache_row(
    *,
    native_row: Any,
    prepared: Any,
    baseline_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove the live renderer/archived response reproduces the locked baseline row."""

    expected_native = {
        "row_identity": native_row.identity,
        "record_id": native_row.record_id,
        "step_index": native_row.step_index,
        "original_image_path": native_row.original_image_path,
        "original_image_sha256": native_row.original_image_sha256,
        "archived_assistant_response_sha256": hashlib.sha256(
            native_row.archived_assistant_response.encode("utf-8")
        ).hexdigest(),
    }
    if any(baseline_row.get(name) != value for name, value in expected_native.items()):
        raise ValueError("update6420 live source row/response differs from the locked baseline")
    provenance = getattr(prepared, "provenance", None)
    expected_provenance = {
        name: baseline_row.get(name)
        for name in (
            "encoded_input_identity",
            "messages_identity",
            "prompt_history_identity",
            "renderer_identity",
            "template_identity",
            "response_source",
        )
    }
    if not isinstance(provenance, Mapping) or dict(provenance) != expected_provenance:
        raise ValueError(
            "update6420 live archived-response rendering differs from the locked baseline"
        )
    return dict(baseline_row)


def _all_ready(phase: str, ready: bool, detail: str, *, world_size: int) -> None:
    local = {"rank": torch.distributed.get_rank(), "phase": phase, "ready": ready, "detail": detail}
    gathered: list[Any] = [None] * world_size
    torch.distributed.all_gather_object(gathered, local)
    failed = [item for item in gathered if not isinstance(item, Mapping) or item.get("phase") != phase or item.get("ready") is not True]
    if failed:
        raise RuntimeError(f"update6420 distributed {phase} gate failed: {failed[0]}")


def run_update6420_cache(config: Update6420ProductionConfig) -> Mapping[str, Any] | None:
    required_env = {"RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK", "TORCHELASTIC_MAX_RESTARTS"}
    if not required_env <= set(os.environ):
        raise ValueError("update6420 cache must run under exact torchrun")
    rank, world, local_rank = (int(os.environ[name]) for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if world != 8 or int(os.environ["LOCAL_WORLD_SIZE"]) != 4 or int(os.environ["TORCHELASTIC_MAX_RESTARTS"]) != 0 or rank // 4 != int(os.environ["GROUP_RANK"]) or rank % 4 != local_rank:
        raise ValueError("update6420 torchrun topology/restart identity mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise RuntimeError("update6420 producer requires four visible CUDA devices per node")
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(config.backend)
    evidence = _read_json(config.checkpoint_evidence_path, label="update6420 checkpoint evidence")
    extractor, rows, baseline_rows = construct_update6420_producer(config, evidence=evidence, rank=rank, device=torch.device(f"cuda:{local_rank}"))
    staging = config.output_path.with_name(f".{config.output_path.name}.update6420-tmp")
    init_error: BaseException | None = None
    if rank == 0:
        try:
            if config.output_path.exists() or config.output_path.is_symlink() or staging.exists() or staging.is_symlink():
                raise FileExistsError("update6420 cache output/staging already exists")
            staging.mkdir(parents=True)
        except Exception as error:  # noqa: BLE001 - all ranks must observe setup failure
            init_error = error
    _all_ready("staging", init_error is None, "ready" if init_error is None else str(init_error), world_size=8)
    _all_ready("staging_visible", staging.is_dir(), "ready" if staging.is_dir() else "missing", world_size=8)
    local_states: list[torch.Tensor] = []
    local_rows: list[dict[str, Any]] = []
    steps = (len(rows) + 7) // 8
    for step in range(steps):
        ordinal = step * 8 + rank
        contributing = ordinal < len(rows)
        selected_ordinal = ordinal if contributing else 0
        prepared = None
        prepare_error: Exception | None = None
        try:
            prepared = extractor.prepare(rows[selected_ordinal])
        except Exception as error:  # noqa: BLE001 - gate before any rank enters FSDP
            prepare_error = error
        _all_ready(
            f"prepare_{step}", prepare_error is None,
            "ready" if prepare_error is None else str(prepare_error), world_size=8,
        )
        assert prepared is not None
        cache_row = None
        provenance_error: Exception | None = None
        try:
            cache_row = _validated_cache_row(
                native_row=rows[selected_ordinal],
                prepared=prepared,
                baseline_row=baseline_rows[selected_ordinal],
            )
        except Exception as error:  # noqa: BLE001 - gate before synchronized FSDP
            provenance_error = error
        _all_ready(
            f"provenance_{step}", provenance_error is None,
            "ready" if provenance_error is None else str(provenance_error), world_size=8,
        )
        assert cache_row is not None
        try:
            state = extractor.extract((prepared,))
        except Exception:
            torch.distributed.destroy_process_group()
            raise
        state_ready = (
            isinstance(state, torch.Tensor)
            and state.shape == (1, 16, 1024)
            and state.is_floating_point()
            and bool(torch.isfinite(state).all())
        )
        _all_ready(
            f"extract_{step}", state_ready,
            "ready" if state_ready else "invalid state", world_size=8,
        )
        if contributing:
            local_states.append(state[0].detach().cpu().float().contiguous())
            local_rows.append(cache_row)
    descriptor = None
    descriptor_error: Exception | None = None
    try:
        descriptor = _write_rank_payload(
            staging / f"rank_{rank:05d}_of_00008.pt", rank=rank,
            state=torch.stack(local_states).float().contiguous(), rows=local_rows,
        )
    except Exception as error:  # noqa: BLE001 - all ranks must observe shard failure
        descriptor_error = error
    _all_ready(
        "rank_payload", descriptor_error is None,
        "ready" if descriptor_error is None else str(descriptor_error), world_size=8,
    )
    assert descriptor is not None
    gathered: list[Any] = [None] * 8
    torch.distributed.all_gather_object(gathered, descriptor)
    manifest = None
    publish_error: BaseException | None = None
    if rank == 0:
        try:
            manifest = publish_update6420_cache_from_rank_payloads(
                staging=staging, output=config.output_path, rank_descriptors=gathered,
                checkpoint_evidence=evidence, baseline_rows=baseline_rows,
                producer={
                    "integrated_repo_root": str(config.integrated_repo_root),
                    "integrated_source_commit": config.integrated_source_commit,
                    "production_config_schema": UPDATE6420_PRODUCTION_CONFIG_SCHEMA,
                    "world_size": 8,
                },
            )
        except Exception as error:  # noqa: BLE001 - all ranks must observe publication failure
            publish_error = error
    _all_ready("publish", publish_error is None, "ready" if publish_error is None else str(publish_error), world_size=8)
    if rank == 0:
        shutil.rmtree(staging)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the strict unsafe/nondeployable update6420 Query-State cache under WS8 torchrun --max-restarts=0")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="authenticate immutable evidence and parse the archived config on CPU",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = parse_update6420_production_config(_read_json(args.config.resolve(), label="update6420 production config"))
    if args.preflight_only:
        print(json.dumps(preflight_update6420_producer(config), sort_keys=True))
        return 0
    try:
        manifest = run_update6420_cache(config)
        if manifest is not None:
            print(json.dumps({"schema": manifest["schema"], "cache_fingerprint": manifest["cache_fingerprint"], "actor_unsafe": True, "deployable": False}, sort_keys=True))
        return 0
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
