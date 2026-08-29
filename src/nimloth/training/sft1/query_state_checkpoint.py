"""Direct-state artifact and local exact-resume contract for Query-State SFT1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Mapping

import numpy as np
import torch

from nimloth.training.sft1.query_state import (
    DIRECT_STATE_ARTIFACT_SCHEMA,
    QUERY_STATE_OBJECTIVE_VERSION,
    QUERY_STATE_SCHEMA,
    SFT1QueryStateTrainingRoot,
)
from nimloth.wm.grid import DirectSlotProjector


QUERY_STATE_CHECKPOINT_SCHEMA = "nimloth_sft1_query_state_checkpoint_v1"
QUERY_STATE_RANK_CHECKPOINT_SCHEMA = "nimloth_sft1_query_state_rank_checkpoint_v1"
QUERY_STATE_DEPLOYABLE_SCHEMA = "nimloth_sft1_query_state_deployable_v1"
QUERY_STATE_DEPLOYABLE_BUNDLE_SCHEMA = "nimloth_sft1_query_state_bundle_v1"
_COMPLETE_MARKER = "COMPLETED"
_HEX = frozenset("0123456789abcdef")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _atomic_torch_save(value: Any, path: Path) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Query-State artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class QueryStateResumeIdentity:
    source_commit: str
    source_manifest_identity: str
    config_identity: str
    run_identity: str
    world_size: int
    stage: str = "sft1_query_state"
    gradient_mode: str = "full_language_direct_state"
    training_schema: str = QUERY_STATE_SCHEMA
    objective_version: str = QUERY_STATE_OBJECTIVE_VERSION
    state_artifact_schema: str = DIRECT_STATE_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        if len(self.source_commit) != 40 or any(char not in _HEX for char in self.source_commit):
            raise ValueError("Query-State source commit must be a lowercase Git SHA")
        for name in ("source_manifest_identity", "config_identity", "run_identity"):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"Query-State {name} must be SHA256")
        expected = (
            "sft1_query_state",
            "full_language_direct_state",
            QUERY_STATE_SCHEMA,
            QUERY_STATE_OBJECTIVE_VERSION,
            DIRECT_STATE_ARTIFACT_SCHEMA,
        )
        actual = (
            self.stage,
            self.gradient_mode,
            self.training_schema,
            self.objective_version,
            self.state_artifact_schema,
        )
        if (
            actual != expected
            or isinstance(self.world_size, bool)
            or not isinstance(self.world_size, int)
            or self.world_size < 1
        ):
            raise ValueError("Query-State resume identity uses an incompatible stage/schema")


@dataclass(frozen=True)
class QueryStateResumeControl:
    identity: QueryStateResumeIdentity
    global_step: int
    data_cursor: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            isinstance(self.global_step, bool)
            or not isinstance(self.global_step, int)
            or self.global_step < 0
            or not isinstance(self.data_cursor, Mapping)
        ):
            raise ValueError("Query-State resume control is invalid")


def export_direct_query_state_artifact(
    path: Path,
    *,
    projector: DirectSlotProjector,
    source_identity: QueryStateResumeIdentity,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Export only the unique no-bias direct state head and immutable metadata."""

    if not isinstance(projector, DirectSlotProjector):
        raise TypeError("Query-State export rejects legacy SharedSlotProjector artifacts")
    reserved = {
        "schema",
        "training_schema",
        "objective_version",
        "state_contract",
        "source_identity",
        "shared_slot_projector",
    }
    supplied = dict(metadata or {})
    if reserved & set(supplied):
        raise ValueError("Query-State artifact metadata may not override reserved fields")
    state_dict = {
        name: value.detach().cpu().clone()
        for name, value in projector.state_dict().items()
    }
    weight = state_dict.get("linear.weight")
    if (
        set(state_dict) != {"linear.weight"}
        or not isinstance(weight, torch.Tensor)
        or weight.shape != (1024, 2048)
        or not weight.is_floating_point()
        or not torch.isfinite(weight).all()
    ):
        raise ValueError(
            "Query-State direct artifact requires a full finite unsharded (1024,2048) weight"
        )
    payload = {
        "schema": QUERY_STATE_DEPLOYABLE_SCHEMA,
        "training_schema": QUERY_STATE_SCHEMA,
        "objective_version": QUERY_STATE_OBJECTIVE_VERSION,
        "state_contract": projector.artifact_metadata(),
        "source_identity": asdict(source_identity),
        "metadata": supplied,
        "state_dict": state_dict,
        "shared_slot_projector": False,
    }
    _atomic_torch_save(payload, Path(path))


def load_direct_query_state_artifact(
    path: Path,
    *,
    expected_source_identity: QueryStateResumeIdentity | None = None,
) -> tuple[DirectSlotProjector, Mapping[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != QUERY_STATE_DEPLOYABLE_SCHEMA:
        raise ValueError("unsupported, legacy, or SharedSlotProjector state artifact")
    if (
        payload.get("training_schema") != QUERY_STATE_SCHEMA
        or payload.get("objective_version") != QUERY_STATE_OBJECTIVE_VERSION
        or payload.get("shared_slot_projector") is not False
    ):
        raise ValueError("Query-State deployable identity mismatch")
    projector = DirectSlotProjector()
    if payload.get("state_contract") != projector.artifact_metadata():
        raise ValueError("Query-State direct state metadata mismatch")
    source_raw = payload.get("source_identity")
    if not isinstance(source_raw, dict):
        raise ValueError("Query-State artifact source identity is absent")
    source = QueryStateResumeIdentity(**source_raw)
    if expected_source_identity is not None and source != expected_source_identity:
        raise ValueError("Query-State artifact source identity mismatch")
    state = payload.get("state_dict")
    weight = state.get("linear.weight") if isinstance(state, dict) else None
    if (
        not isinstance(state, dict)
        or set(state) != {"linear.weight"}
        or not isinstance(weight, torch.Tensor)
        or weight.shape != (1024, 2048)
        or not weight.is_floating_point()
        or not torch.isfinite(weight).all()
    ):
        raise ValueError("Query-State artifact requires one full finite direct-state weight")
    projector.load_state_dict(state, strict=True)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Query-State artifact metadata is invalid")
    return projector, metadata


def save_query_state_resume_checkpoint(
    path: Path,
    *,
    root: SFT1QueryStateTrainingRoot,
    optimizer: torch.optim.Optimizer,
    control: QueryStateResumeControl,
    scheduler_state: Mapping[str, Any],
) -> None:
    """Save one atomic local resume payload with strict new-stage identity."""

    root.assert_trainable_contract()
    model = {
        name: parameter.detach().cpu().clone()
        for name, parameter in root.named_parameters()
        if parameter.requires_grad
    }
    if not model or "objective.projector.linear.weight" not in model:
        raise ValueError("Query-State checkpoint is missing the direct state head")
    rng: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng["torch_cuda"] = torch.cuda.get_rng_state()
    _atomic_torch_save(
        {
            "schema": QUERY_STATE_CHECKPOINT_SCHEMA,
            "control": {
                "identity": asdict(control.identity),
                "global_step": control.global_step,
                "data_cursor": dict(control.data_cursor),
            },
            "trainable_parameter_names": sorted(model),
            "model": model,
            "optimizer": optimizer.state_dict(),
            "scheduler": dict(scheduler_state),
            "rng": rng,
        },
        Path(path),
    )


def _validate_model_tensor_set_before_restore(
    current: Mapping[str, torch.nn.Parameter],
    state: Mapping[str, Any],
    *,
    owner: str,
) -> None:
    """Validate every model tensor before any live parameter is mutated."""

    for name, parameter in current.items():
        value = state[name]
        if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
            raise ValueError(f"Query-State {owner} tensor shape mismatch: {name}")
        if value.dtype != parameter.dtype:
            raise ValueError(
                f"Query-State {owner} tensor dtype mismatch: "
                f"{name}: {value.dtype} != {parameter.dtype}"
            )


def load_query_state_resume_checkpoint(
    path: Path,
    *,
    root: SFT1QueryStateTrainingRoot,
    optimizer: torch.optim.Optimizer,
    expected_identity: QueryStateResumeIdentity,
) -> tuple[QueryStateResumeControl, Mapping[str, Any]]:
    """Restore only an exact same-stage/schema Query-State checkpoint."""

    root.assert_trainable_contract()
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != QUERY_STATE_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported legacy/cross-stage Query-State resume checkpoint")
    control_raw = payload.get("control")
    if not isinstance(control_raw, dict) or not isinstance(control_raw.get("identity"), dict):
        raise ValueError("Query-State resume control is invalid")
    identity = QueryStateResumeIdentity(**control_raw["identity"])
    if identity != expected_identity:
        raise ValueError("Query-State resume identity mismatch")
    if set(control_raw) != {"identity", "global_step", "data_cursor"}:
        raise ValueError("Query-State resume control has unknown/missing fields")
    control = QueryStateResumeControl(
        identity=identity,
        global_step=control_raw["global_step"],
        data_cursor=control_raw["data_cursor"],
    )
    current = {
        name: parameter
        for name, parameter in root.named_parameters()
        if parameter.requires_grad
    }
    state = payload.get("model")
    names = payload.get("trainable_parameter_names")
    if (
        not isinstance(state, dict)
        or names != sorted(current)
        or set(state) != set(current)
        or "objective.projector.linear.weight" not in state
    ):
        raise ValueError("Query-State checkpoint trainable parameter set mismatch")
    _validate_model_tensor_set_before_restore(current, state, owner="checkpoint")
    with torch.no_grad():
        for name, parameter in current.items():
            parameter.copy_(state[name].to(device=parameter.device))
    optimizer.load_state_dict(payload.get("optimizer"))
    scheduler = payload.get("scheduler")
    if not isinstance(scheduler, dict):
        raise ValueError("Query-State checkpoint scheduler state is invalid")
    rng = payload.get("rng")
    if not isinstance(rng, dict) or not {"python", "numpy", "torch_cpu"} <= set(rng):
        raise ValueError("Query-State checkpoint RNG state is incomplete")
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"])
    if torch.cuda.is_available():
        cuda_state = rng.get("torch_cuda")
        if not isinstance(cuda_state, torch.Tensor):
            raise ValueError("Query-State CUDA resume requires CUDA RNG state")
        torch.cuda.set_rng_state(cuda_state)
    return control, scheduler


@dataclass(frozen=True)
class QueryStateDistributedControl:
    """Rank-checkpoint control state including both data and metric cursors."""

    identity: QueryStateResumeIdentity
    global_step: int
    data_cursor: Mapping[str, Any]
    metric_cursor: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            isinstance(self.global_step, bool)
            or not isinstance(self.global_step, int)
            or self.global_step < 0
            or not isinstance(self.data_cursor, Mapping)
            or not isinstance(self.metric_cursor, Mapping)
        ):
            raise ValueError("Query-State distributed control is invalid")


@dataclass(frozen=True)
class QueryStateRankState:
    identity: QueryStateResumeIdentity
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Query-State immutable JSON already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
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
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _rank_path(path: Path, rank: int, world_size: int) -> Path:
    return Path(path) / f"rank_{rank:05d}_of_{world_size:05d}.pt"


def _rank_manifest_path(path: Path, rank: int, world_size: int) -> Path:
    return Path(path) / f"rank_{rank:05d}_of_{world_size:05d}.json"


def _validated_rank_manifest(
    path: Path,
    *,
    rank: int,
    world_size: int,
    expected_identity: QueryStateResumeIdentity,
) -> str:
    manifest_path = _rank_manifest_path(path, rank, world_size)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Query-State rank checkpoint manifest is invalid") from error
    if not isinstance(raw, dict) or set(raw) != {
        "schema",
        "rank",
        "world_size",
        "identity",
        "shard_sha256",
    }:
        raise ValueError("Query-State rank checkpoint manifest contract is invalid")
    if (
        raw["schema"] != QUERY_STATE_RANK_CHECKPOINT_SCHEMA
        or raw["rank"] != rank
        or raw["world_size"] != world_size
        or not isinstance(raw["identity"], dict)
        or QueryStateResumeIdentity(**raw["identity"]) != expected_identity
        or not _is_sha256(raw["shard_sha256"])
    ):
        raise ValueError("Query-State rank checkpoint manifest identity mismatch")
    shard = _rank_path(path, rank, world_size)
    if not shard.is_file() or _sha256_file(shard) != raw["shard_sha256"]:
        raise ValueError("Query-State rank checkpoint shard hash mismatch")
    return str(raw["shard_sha256"])


def capture_query_state_rank_state(
    root: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_state: Mapping[str, Any],
    identity: QueryStateResumeIdentity,
) -> QueryStateRankState:
    """Capture local trainable shards plus optimizer/scheduler and exact RNG."""

    model = {
        name: parameter.detach().cpu().clone()
        for name, parameter in root.named_parameters()
        if parameter.requires_grad
    }
    if not model or not any(
        name.endswith("objective.projector.linear.weight")
        or name == "objective.projector.linear.weight"
        for name in model
    ):
        raise ValueError("Query-State rank state is missing the direct state head")
    rng: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng["torch_cuda"] = torch.cuda.get_rng_state()
    return QueryStateRankState(
        identity=identity,
        model=model,
        optimizer=optimizer.state_dict(),
        scheduler=dict(scheduler_state),
        rng=rng,
    )


def restore_query_state_rank_state(
    root: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    state: QueryStateRankState,
) -> Mapping[str, Any]:
    """Restore the exact same local trainable shard and runtime state."""

    current = {
        name: parameter
        for name, parameter in root.named_parameters()
        if parameter.requires_grad
    }
    if set(current) != set(state.model):
        raise ValueError("Query-State rank checkpoint trainable key set mismatch")
    _validate_model_tensor_set_before_restore(
        current,
        state.model,
        owner="rank checkpoint",
    )
    with torch.no_grad():
        for name, parameter in current.items():
            parameter.copy_(state.model[name].to(device=parameter.device))
    optimizer.load_state_dict(dict(state.optimizer))
    required_rng = {"python", "numpy", "torch_cpu"}
    if not required_rng <= set(state.rng):
        raise ValueError("Query-State rank checkpoint RNG state is incomplete")
    random.setstate(state.rng["python"])
    np.random.set_state(state.rng["numpy"])
    torch.set_rng_state(state.rng["torch_cpu"])
    if torch.cuda.is_available():
        cuda_state = state.rng.get("torch_cuda")
        if not isinstance(cuda_state, torch.Tensor):
            raise ValueError("Query-State CUDA resume requires CUDA RNG state")
        torch.cuda.set_rng_state(cuda_state)
    return dict(state.scheduler)


def save_query_state_rank_state(
    path: Path,
    *,
    rank: int,
    world_size: int,
    state: QueryStateRankState,
) -> None:
    """Atomically publish one immutable worker-local shard."""

    checkpoint = Path(path)
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Query-State checkpoint rank/world-size is invalid")
    if (checkpoint / _COMPLETE_MARKER).exists():
        raise FileExistsError("completed Query-State checkpoint is immutable")
    if state.identity.world_size != world_size:
        raise ValueError("Query-State rank shard identity/world-size mismatch")
    if (
        not isinstance(state.model, Mapping)
        or not any(
            str(name).endswith("objective.projector.linear.weight")
            or str(name) == "objective.projector.linear.weight"
            for name in state.model
        )
    ):
        raise ValueError("Query-State rank shard is missing the direct state head")
    if not {"python", "numpy", "torch_cpu"} <= set(state.rng):
        raise ValueError("Query-State rank shard RNG state is incomplete")
    _atomic_torch_save(
        {
            "schema": QUERY_STATE_RANK_CHECKPOINT_SCHEMA,
            "training_schema": QUERY_STATE_SCHEMA,
            "objective_version": QUERY_STATE_OBJECTIVE_VERSION,
            "state_artifact_schema": DIRECT_STATE_ARTIFACT_SCHEMA,
            "rank": rank,
            "world_size": world_size,
            "identity": asdict(state.identity),
            "model": dict(state.model),
            "optimizer": dict(state.optimizer),
            "scheduler": dict(state.scheduler),
            "rng": dict(state.rng),
        },
        _rank_path(checkpoint, rank, world_size),
    )
    _atomic_json(
        {
            "schema": QUERY_STATE_RANK_CHECKPOINT_SCHEMA,
            "rank": rank,
            "world_size": world_size,
            "identity": asdict(state.identity),
            "shard_sha256": _sha256_file(_rank_path(checkpoint, rank, world_size)),
        },
        _rank_manifest_path(checkpoint, rank, world_size),
    )


def finalize_query_state_rank_checkpoint(
    path: Path,
    *,
    control: QueryStateDistributedControl,
) -> None:
    """Publish control metadata only after every immutable rank shard exists."""

    checkpoint = Path(path)
    world_size = control.identity.world_size
    if (checkpoint / _COMPLETE_MARKER).exists():
        raise FileExistsError("completed Query-State checkpoint is immutable")
    missing = [
        rank
        for rank in range(world_size)
        if not _rank_path(checkpoint, rank, world_size).is_file()
        or not _rank_manifest_path(checkpoint, rank, world_size).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Query-State checkpoint rank shards are incomplete: {missing}")
    shard_hashes = {
        str(rank): _validated_rank_manifest(
            checkpoint,
            rank=rank,
            world_size=world_size,
            expected_identity=control.identity,
        )
        for rank in range(world_size)
    }
    control_path = checkpoint / "control.json"
    _atomic_json(
        {
            "schema": QUERY_STATE_RANK_CHECKPOINT_SCHEMA,
            "training_schema": QUERY_STATE_SCHEMA,
            "objective_version": QUERY_STATE_OBJECTIVE_VERSION,
            "state_artifact_schema": DIRECT_STATE_ARTIFACT_SCHEMA,
            "identity": asdict(control.identity),
            "global_step": control.global_step,
            "data_cursor": dict(control.data_cursor),
            "metric_cursor": dict(control.metric_cursor),
            "rank_shard_sha256": shard_hashes,
        },
        control_path,
    )
    marker = checkpoint / _COMPLETE_MARKER
    descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        marker_payload = f"control_sha256={_sha256_file(control_path)}\n".encode()
        os.write(descriptor, marker_payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(checkpoint)


def load_query_state_rank_state(
    path: Path,
    *,
    rank: int,
    expected_identity: QueryStateResumeIdentity,
) -> tuple[QueryStateRankState, QueryStateDistributedControl]:
    """Load only a complete exact-identity Query-State rank transaction."""

    checkpoint = Path(path)
    marker = checkpoint / _COMPLETE_MARKER
    control_path = checkpoint / "control.json"
    if not marker.is_file():
        raise ValueError("Query-State checkpoint has no atomic completion marker")
    try:
        marker_text = marker.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError("Query-State completion marker is unreadable") from error
    expected_marker = (
        f"control_sha256={_sha256_file(control_path)}\n"
        if control_path.is_file()
        else ""
    )
    if marker_text != expected_marker:
        raise ValueError("Query-State checkpoint control hash mismatch")
    try:
        raw = json.loads(control_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Query-State rank checkpoint control is invalid") from error
    if not isinstance(raw, dict):
        raise ValueError("Query-State rank checkpoint control must be a mapping")
    expected_metadata = {
        "schema": QUERY_STATE_RANK_CHECKPOINT_SCHEMA,
        "training_schema": QUERY_STATE_SCHEMA,
        "objective_version": QUERY_STATE_OBJECTIVE_VERSION,
        "state_artifact_schema": DIRECT_STATE_ARTIFACT_SCHEMA,
    }
    for name, expected in expected_metadata.items():
        if raw.pop(name, None) != expected:
            raise ValueError("unsupported legacy/cross-stage Query-State rank checkpoint")
    identity_raw = raw.pop("identity", None)
    if not isinstance(identity_raw, dict):
        raise ValueError("Query-State rank checkpoint identity is absent")
    identity = QueryStateResumeIdentity(**identity_raw)
    if identity != expected_identity:
        raise ValueError("Query-State rank checkpoint identity mismatch")
    if not 0 <= rank < identity.world_size:
        raise ValueError("Query-State rank checkpoint requested rank is invalid")
    hashes = raw.pop("rank_shard_sha256", None)
    if not isinstance(hashes, dict) or set(hashes) != {
        str(value) for value in range(identity.world_size)
    }:
        raise ValueError("Query-State rank checkpoint shard hash index is incomplete")
    for shard_rank in range(identity.world_size):
        digest = _validated_rank_manifest(
            checkpoint,
            rank=shard_rank,
            world_size=identity.world_size,
            expected_identity=identity,
        )
        if digest != hashes[str(shard_rank)]:
            raise ValueError("Query-State rank checkpoint shard hash mismatch")
    control = QueryStateDistributedControl(
        identity=identity,
        global_step=raw.pop("global_step", None),
        data_cursor=raw.pop("data_cursor", None),
        metric_cursor=raw.pop("metric_cursor", None),
    )
    if raw:
        raise ValueError("Query-State rank checkpoint control has unknown fields")
    payload = torch.load(
        _rank_path(checkpoint, rank, identity.world_size),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise ValueError("Query-State rank checkpoint shard is invalid")
    for name, expected in {
        **expected_metadata,
        "rank": rank,
        "world_size": identity.world_size,
    }.items():
        if payload.pop(name, None) != expected:
            raise ValueError("Query-State rank checkpoint shard identity mismatch")
    shard_identity = payload.pop("identity", None)
    if (
        not isinstance(shard_identity, dict)
        or QueryStateResumeIdentity(**shard_identity) != identity
    ):
        raise ValueError("Query-State rank checkpoint shard source/run identity mismatch")
    required = {"model", "optimizer", "scheduler", "rng"}
    if set(payload) != required or not all(
        isinstance(payload[name], dict) for name in required
    ):
        raise ValueError("Query-State rank checkpoint shard payload is incomplete")
    return QueryStateRankState(identity=identity, **payload), control


def export_query_state_deployable_bundle(
    path: Path,
    *,
    actor_exporter: Callable[[Path], None],
    processor_exporter: Callable[[Path], None],
    projector: DirectSlotProjector,
    source_identity: QueryStateResumeIdentity,
    metadata: Mapping[str, Any],
) -> None:
    """Publish separate Qwen, processor, and direct-state deployment owners.

    The actor callback is the boundary at which a production caller must pass a
    full FSDP state-dict-backed Qwen exporter. The projector must likewise be
    materialized under an official full-parameter context; direct-state export
    rejects local/zero-sized FSDP shards by exact shape. This function does not
    gather or approximate model shards itself.
    """

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Query-State deployable bundle exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Query-State temporary deployable exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        actor_exporter(temporary / "actor")
        processor_exporter(temporary / "processor")
        if not (temporary / "actor").is_dir() or not (
            temporary / "processor"
        ).is_dir():
            raise ValueError("Query-State actor/processor exporters must create directories")
        forbidden_terms = (
            "optimizer",
            "scheduler",
            "world_model",
            "value_head",
            "training_head",
            "shared_slot_projector",
        )
        bad = [
            item
            for owner in (temporary / "actor", temporary / "processor")
            for item in owner.rglob("*")
            if item.is_file()
            and any(term in item.name.lower() for term in forbidden_terms)
        ]
        if bad:
            raise ValueError("Query-State deployable contains training-only files")
        export_direct_query_state_artifact(
            temporary / "direct_state.pt",
            projector=projector,
            source_identity=source_identity,
            metadata={"bundle_role": "direct_state_only"},
        )
        reserved = {
            "schema",
            "training_schema",
            "objective_version",
            "direct_state_schema",
            "source_identity",
        }
        if reserved & set(metadata):
            raise ValueError("Query-State bundle metadata overrides reserved identity")
        _atomic_json(
            {
                "schema": QUERY_STATE_DEPLOYABLE_BUNDLE_SCHEMA,
                "training_schema": QUERY_STATE_SCHEMA,
                "objective_version": QUERY_STATE_OBJECTIVE_VERSION,
                "direct_state_schema": QUERY_STATE_DEPLOYABLE_SCHEMA,
                "source_identity": asdict(source_identity),
                "metadata": dict(metadata),
                "owners": {
                    "actor": "full_qwen_actor",
                    "processor": "qwen_processor_tokenizer",
                    "direct_state": "no_bias_linear_2048_to_1024",
                },
            },
            temporary / "bundle.json",
        )
        temporary.replace(destination)
        _fsync_directory(destination.parent)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


__all__ = [
    "QUERY_STATE_CHECKPOINT_SCHEMA",
    "QUERY_STATE_DEPLOYABLE_BUNDLE_SCHEMA",
    "QUERY_STATE_DEPLOYABLE_SCHEMA",
    "QUERY_STATE_RANK_CHECKPOINT_SCHEMA",
    "QueryStateDistributedControl",
    "QueryStateRankState",
    "QueryStateResumeControl",
    "QueryStateResumeIdentity",
    "capture_query_state_rank_state",
    "export_direct_query_state_artifact",
    "export_query_state_deployable_bundle",
    "finalize_query_state_rank_checkpoint",
    "load_direct_query_state_artifact",
    "load_query_state_rank_state",
    "load_query_state_resume_checkpoint",
    "restore_query_state_rank_state",
    "save_query_state_rank_state",
    "save_query_state_resume_checkpoint",
]
