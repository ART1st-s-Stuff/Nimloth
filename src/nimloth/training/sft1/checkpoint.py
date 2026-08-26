"""Exact-resume rank checkpoints and minimal deployable SFT1-v2 export."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import torch

from nimloth.training.sft1.config import STATE_INTERFACE_OBJECTIVE_VERSION


SFT1_V2_CHECKPOINT_SCHEMA = "nimloth_sft1_state_v2_checkpoint_v2"
SFT1_V2_EXPORT_SCHEMA = "nimloth_state_interface_v2_deployable_v1"
_COMPLETE_MARKER = "COMPLETED"
_HEX = frozenset("0123456789abcdef")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SFT1V2ControlState:
    global_step: int
    data_cursor: Mapping[str, Any]
    manifest_identity: str
    config_identity: str
    objective_version: str
    world_size: int
    run_identity: str
    source_commit: str

    def __post_init__(self) -> None:
        if self.global_step < 0 or self.world_size < 1:
            raise ValueError("checkpoint global_step/world_size are invalid")
        if not _is_sha256(self.manifest_identity) or not _is_sha256(
            self.config_identity
        ):
            raise ValueError("checkpoint manifest/config identities must be SHA256 digests")
        if self.objective_version != STATE_INTERFACE_OBJECTIVE_VERSION:
            raise ValueError("checkpoint uses an old state objective")
        if not _is_sha256(self.run_identity):
            raise ValueError("checkpoint run identity must be a SHA256 digest")
        if len(self.source_commit) != 40 or any(char not in _HEX for char in self.source_commit):
            raise ValueError("checkpoint source commit must be a lowercase Git SHA")


@dataclass(frozen=True)
class SFT1V2RankState:
    model: Mapping[str, Any]
    optimizer: Mapping[str, Any]
    scheduler: Mapping[str, Any]
    rng: Mapping[str, Any]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_torch_save(value: Any, path: Path) -> None:
    temporary: Path | None = None
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            encoding="utf-8",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _rank_path(path: Path, rank: int, world_size: int) -> Path:
    return path / f"rank_{rank:05d}_of_{world_size:05d}.pt"


def save_sft1_v2_rank_state(
    path: Path,
    *,
    rank: int,
    world_size: int,
    state: SFT1V2RankState,
) -> None:
    checkpoint = Path(path)
    if rank < 0 or rank >= world_size or world_size < 1:
        raise ValueError("checkpoint rank/world_size are invalid")
    if (checkpoint / _COMPLETE_MARKER).exists():
        raise FileExistsError("completed SFT1-v2 checkpoint is immutable")
    _atomic_torch_save(
        {
            "schema": SFT1_V2_CHECKPOINT_SCHEMA,
            "objective_version": STATE_INTERFACE_OBJECTIVE_VERSION,
            "rank": rank,
            "world_size": world_size,
            "model": dict(state.model),
            "optimizer": dict(state.optimizer),
            "scheduler": dict(state.scheduler),
            "rng": dict(state.rng),
        },
        _rank_path(checkpoint, rank, world_size),
    )


def finalize_sft1_v2_checkpoint(
    path: Path,
    *,
    control: SFT1V2ControlState,
) -> None:
    checkpoint = Path(path)
    if (checkpoint / _COMPLETE_MARKER).exists():
        raise FileExistsError("completed SFT1-v2 checkpoint is immutable")
    missing = [
        rank
        for rank in range(control.world_size)
        if not _rank_path(checkpoint, rank, control.world_size).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"checkpoint rank shards are incomplete: {missing}")
    shard_hashes = {
        str(rank): _sha256_file(_rank_path(checkpoint, rank, control.world_size))
        for rank in range(control.world_size)
    }
    _atomic_json(
        {
            "schema": SFT1_V2_CHECKPOINT_SCHEMA,
            **asdict(control),
            "rank_shard_sha256": shard_hashes,
        },
        checkpoint / "control.json",
    )
    marker = checkpoint / _COMPLETE_MARKER
    descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, b"complete\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(checkpoint)


def load_sft1_v2_rank_state(
    path: Path,
    *,
    rank: int,
    expected_world_size: int,
    expected_manifest_identity: str,
    expected_config_identity: str,
    expected_run_identity: str,
    expected_source_commit: str,
) -> tuple[SFT1V2RankState, SFT1V2ControlState]:
    checkpoint = Path(path)
    if not (checkpoint / _COMPLETE_MARKER).is_file():
        raise ValueError("SFT1-v2 checkpoint has no atomic completion marker")
    control_raw = json.loads((checkpoint / "control.json").read_text(encoding="utf-8"))
    if control_raw.pop("schema", None) != SFT1_V2_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported or legacy SFT1-v2 checkpoint schema")
    shard_hashes = control_raw.pop("rank_shard_sha256", None)
    if not isinstance(shard_hashes, dict) or set(shard_hashes) != {str(value) for value in range(expected_world_size)}:
        raise ValueError("checkpoint shard hash index is incomplete")
    for shard_rank in range(expected_world_size):
        shard_path = _rank_path(checkpoint, shard_rank, expected_world_size)
        if not shard_path.is_file() or _sha256_file(shard_path) != shard_hashes[str(shard_rank)]:
            raise ValueError("checkpoint rank shard hash mismatch")
    control = SFT1V2ControlState(**control_raw)
    if control.world_size != expected_world_size:
        raise ValueError("checkpoint world size differs from the resume world size")
    if control.manifest_identity != expected_manifest_identity:
        raise ValueError("checkpoint manifest identity mismatch")
    if control.config_identity != expected_config_identity:
        raise ValueError("checkpoint config identity mismatch")
    if control.run_identity != expected_run_identity:
        raise ValueError("checkpoint run identity mismatch")
    if control.source_commit != expected_source_commit:
        raise ValueError("checkpoint source commit mismatch")
    payload = torch.load(
        _rank_path(checkpoint, rank, expected_world_size),
        map_location="cpu",
        weights_only=False,
    )
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != SFT1_V2_CHECKPOINT_SCHEMA
        or payload.get("objective_version") != STATE_INTERFACE_OBJECTIVE_VERSION
        or payload.get("rank") != rank
        or payload.get("world_size") != expected_world_size
    ):
        raise ValueError("SFT1-v2 checkpoint rank shard identity mismatch")
    return (
        SFT1V2RankState(
            model=payload["model"],
            optimizer=payload["optimizer"],
            scheduler=payload["scheduler"],
            rng=payload["rng"],
        ),
        control,
    )


def capture_sft1_v2_rank_state(
    root: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> SFT1V2RankState:
    """Capture only local trainable shards plus exact optimizer/RNG state."""

    model = {
        name: parameter.detach().cpu().clone()
        for name, parameter in root.named_parameters()
        if parameter.requires_grad
    }
    if not model:
        raise ValueError("SFT1-v2 checkpoint has no trainable model state")
    rng: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng["torch_cuda"] = torch.cuda.get_rng_state()
    return SFT1V2RankState(
        model=model,
        optimizer=optimizer.state_dict(),
        scheduler={},
        rng=rng,
    )


def restore_sft1_v2_rank_state(
    root: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    state: SFT1V2RankState,
) -> None:
    """Restore exact local trainable shards; frozen ID176 reloads from source."""

    current = {
        name: parameter
        for name, parameter in root.named_parameters()
        if parameter.requires_grad
    }
    if set(current) != set(state.model):
        raise ValueError("checkpoint trainable model key set mismatch")
    with torch.no_grad():
        for name, parameter in current.items():
            value = state.model[name]
            if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
                raise ValueError(f"checkpoint trainable tensor shape mismatch: {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    optimizer.load_state_dict(dict(state.optimizer))
    if state.scheduler:
        raise ValueError("SFT1-v2 constant-LR checkpoint must not contain scheduler state")
    rng = state.rng
    required = {"python", "numpy", "torch_cpu"}
    if not required <= set(rng):
        raise ValueError("checkpoint RNG state is incomplete")
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"])
    if torch.cuda.is_available():
        cuda_state = rng.get("torch_cuda")
        if not isinstance(cuda_state, torch.Tensor):
            raise ValueError("CUDA resume requires exact CUDA RNG state")
        torch.cuda.set_rng_state(cuda_state)


def export_sft1_v2_deployable(
    path: Path,
    *,
    actor_exporter: Callable[[Path], None],
    processor_exporter: Callable[[Path], None],
    projector_state: Mapping[str, Any],
    state_metadata: Mapping[str, Any],
) -> None:
    """Export only actor/query, processor, projector, and state-interface metadata."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"deployable export already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.mkdir(parents=True)
    try:
        actor_exporter(temporary / "actor")
        processor_exporter(temporary / "processor")
        if not (temporary / "actor").is_dir() or not (temporary / "processor").is_dir():
            raise ValueError("actor/processor exporters must create their owned directories")
        forbidden_terms = {
            "visual_readout",
            "instruction_readout",
            "feasibility_head",
            "state_policy_head",
            "training_heads",
            "optimizer",
            "scheduler",
            "world_model",
            "value_head",
        }
        bad_projector_keys = [
            key
            for key in projector_state
            if any(term in str(key).lower() for term in forbidden_terms)
        ]
        if bad_projector_keys:
            raise ValueError("deployable projector state contains training-only keys")
        bad_export_files = [
            file
            for owner in (temporary / "actor", temporary / "processor")
            for file in owner.rglob("*")
            if file.is_file()
            and any(term in file.name.lower() for term in forbidden_terms)
        ]
        if bad_export_files:
            raise ValueError("actor/processor export contains training-only files")
        _atomic_torch_save(dict(projector_state), temporary / "slot_projector.pt")
        reserved = {
            "schema",
            "objective_version",
            "grid_tokens",
            "state_dim",
            "ordering",
            "shared_slot_projector",
        }
        if reserved & set(state_metadata):
            raise ValueError("state metadata may not override the deployable contract")
        required_metadata = {"manifest_identity", "query_mode", "action_token_ids"}
        missing_metadata = sorted(required_metadata - set(state_metadata))
        if missing_metadata:
            raise ValueError(
                "deployable metadata is missing field: " + missing_metadata[0]
            )
        action_token_ids = state_metadata["action_token_ids"]
        if (
            not _is_sha256(state_metadata["manifest_identity"])
            or state_metadata["query_mode"] != "inject"
            or not isinstance(action_token_ids, (list, tuple))
            or len(action_token_ids) != 8
            or len(set(action_token_ids)) != 8
        ):
            raise ValueError("deployable metadata identity/query/action contract is invalid")
        metadata = {
            **dict(state_metadata),
            "schema": SFT1_V2_EXPORT_SCHEMA,
            "objective_version": STATE_INTERFACE_OBJECTIVE_VERSION,
            "grid_tokens": 16,
            "state_dim": 1024,
            "ordering": "row_major",
            "shared_slot_projector": True,
        }
        forbidden = forbidden_terms & set(metadata)
        if forbidden:
            raise ValueError(f"deployable metadata contains training-only fields: {sorted(forbidden)}")
        _atomic_json(metadata, temporary / "state_interface_config.json")
        temporary.replace(destination)
        _fsync_directory(destination.parent)
    except Exception:
        if temporary.exists():
            import shutil

            shutil.rmtree(temporary)
        raise


__all__ = [
    "SFT1V2ControlState",
    "SFT1V2RankState",
    "SFT1_V2_CHECKPOINT_SCHEMA",
    "SFT1_V2_EXPORT_SCHEMA",
    "capture_sft1_v2_rank_state",
    "export_sft1_v2_deployable",
    "finalize_sft1_v2_checkpoint",
    "load_sft1_v2_rank_state",
    "restore_sft1_v2_rank_state",
    "save_sft1_v2_rank_state",
]
