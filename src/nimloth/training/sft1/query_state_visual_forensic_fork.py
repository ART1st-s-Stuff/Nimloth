"""Strict model-only initialization and event contract for the visual forensic fork.

The full launch schema belongs to :mod:`query_state_training_config`; this module
only authenticates the Formal38 ancestor, performs the model-only load, and
builds the fixed diagnostic event plan.  It has no alternate launch path.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nimloth.training.sft1.query_state_checkpoint import (
    QueryStateResumeIdentity,
    load_query_state_forensic_model_for_debug,
)


@dataclass(frozen=True)
class QueryStateVisualForkEvent:
    update: int
    kind: str
    report_only: bool


@dataclass(frozen=True)
class QueryStateVisualForkInitializationReceipt:
    ancestor_checkpoint: str
    ancestor_control_sha256: str
    model_loaded: bool
    optimizer_restored: bool
    scheduler_restored: bool
    rng_restored: bool
    data_cursor_restored: bool
    wandb_cursor_restored: bool


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ancestor_fields(config: Any) -> Mapping[str, Any]:
    forensic = getattr(config, "forensic_fork", None)
    source = getattr(config, "source", None)
    resources = getattr(config, "resources", None)
    if (
        getattr(config, "mode", None) != "visual_only_forensic_fork"
        or not isinstance(forensic, Mapping)
        or not isinstance(source, Mapping)
        or not isinstance(resources, Mapping)
    ):
        raise TypeError("visual fork requires the strict production training config")
    return {
        "checkpoint_path": forensic["ancestor_checkpoint_path"],
        "failure_manifest_path": forensic["ancestor_failure_manifest_path"],
        "control_sha256": forensic["ancestor_control_sha256"],
        "source_commit": forensic["ancestor_source_commit"],
        "current_source_commit": source["commit"],
        "source_manifest_identity": forensic["ancestor_source_manifest_identity"],
        "source_config_identity": forensic["ancestor_source_config_identity"],
        "run_identity": forensic["ancestor_run_identity"],
        "world_size": resources["world_size"],
        "update": forensic["ancestor_update"],
    }


def build_visual_fork_event_plan(
    *,
    schedule_start_update: int,
    epoch_updates: int,
    checkpoint_cadence_updates: int,
    fixed_additional_epochs: int,
) -> tuple[QueryStateVisualForkEvent, ...]:
    if (
        schedule_start_update != 1605
        or epoch_updates != 1605
        or checkpoint_cadence_updates != 321
        or fixed_additional_epochs != 4
    ):
        raise ValueError("visual fork event plan differs from the fixed four-epoch contract")
    end = schedule_start_update + epoch_updates * fixed_additional_epochs
    events = [
        QueryStateVisualForkEvent(schedule_start_update, "calibration_parity", True)
    ]
    for update in range(
        schedule_start_update + checkpoint_cadence_updates,
        end + 1,
        checkpoint_cadence_updates,
    ):
        events.append(QueryStateVisualForkEvent(update, "checkpoint", False))
        if update % epoch_updates == 0:
            events.append(QueryStateVisualForkEvent(update, "calibration", True))
            if update == end:
                events.append(QueryStateVisualForkEvent(update, "holdout", True))
                events.append(
                    QueryStateVisualForkEvent(update, "fixed_budget_complete", True)
                )
    return tuple(events)


def authenticate_visual_fork_ancestor(
    config: Any,
    *,
    verify_payload_hashes: bool = True,
) -> QueryStateResumeIdentity:
    """Authenticate Formal38 control/failure and all eight immutable rank shards."""

    ancestor = _ancestor_fields(config)
    checkpoint = Path(str(ancestor["checkpoint_path"])).resolve()
    failure_path = Path(str(ancestor["failure_manifest_path"])).resolve()
    try:
        control_path = checkpoint / "control.json"
        control = json.loads(control_path.read_text(encoding="utf-8"))
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("visual fork ancestor evidence is unreadable") from error
    control_hash = _file_sha256(control_path)
    identity_raw = control.get("identity") if isinstance(control, dict) else None
    forensic = failure.get("forensic_checkpoint") if isinstance(failure, dict) else None
    if ancestor["source_commit"] == ancestor["current_source_commit"]:
        raise ValueError("visual fork ancestor source must differ from runtime source")
    expected_identity = QueryStateResumeIdentity(
        source_commit=str(ancestor["source_commit"]),
        source_manifest_identity=str(ancestor["source_manifest_identity"]),
        config_identity=str(ancestor["source_config_identity"]),
        run_identity=str(ancestor["run_identity"]),
        world_size=int(ancestor["world_size"]),
        experiment_mode="formal",
    )
    rank_hashes = control.get("rank_shard_sha256") if isinstance(control, dict) else None
    if (
        control_hash != ancestor["control_sha256"]
        or not isinstance(identity_raw, dict)
        or QueryStateResumeIdentity(**identity_raw) != expected_identity
        or control.get("global_step") != ancestor["update"]
        or control.get("forensic_only") is not True
        or control.get("terminal_primary") is not False
        or not isinstance(rank_hashes, Mapping)
        or set(rank_hashes) != {str(rank) for rank in range(8)}
        or failure.get("run_identity") != ancestor["run_identity"]
        or failure.get("mode") != "formal"
        or failure.get("end_update") != ancestor["update"]
        or failure.get("resumable") is not False
        or not isinstance(forensic, Mapping)
        or forensic.get("path") != str(checkpoint)
        or forensic.get("control_sha256") != control_hash
        or forensic.get("forensic_only") is not True
        or forensic.get("resumable") is not False
        or forensic.get("authoritative") is not False
    ):
        raise ValueError("visual fork ancestor control/failure identity mismatch")
    for rank in range(8):
        shard = checkpoint / f"rank_{rank:05d}_of_00008.pt"
        manifest_path = checkpoint / f"rank_{rank:05d}_of_00008.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("visual fork ancestor rank manifest is invalid") from error
        actual_hash = (
            _file_sha256(shard)
            if verify_payload_hashes and shard.is_file() and not shard.is_symlink()
            else rank_hashes[str(rank)]
            if shard.is_file() and not shard.is_symlink()
            else None
        )
        if (
            not shard.is_file()
            or shard.is_symlink()
            or not isinstance(manifest, Mapping)
            or manifest.get("rank") != rank
            or manifest.get("world_size") != 8
            or manifest.get("shard_sha256") != rank_hashes[str(rank)]
            or actual_hash != rank_hashes[str(rank)]
        ):
            raise ValueError("visual fork ancestor rank shard inventory mismatch")
    return expected_identity


def initialize_visual_fork_from_forensic_model(
    config: Any,
    *,
    root: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    rank: int,
    expected_ancestor_identity: QueryStateResumeIdentity,
) -> QueryStateVisualForkInitializationReceipt:
    """Load only ancestor model shards and prove fresh runtime state was untouched."""

    if expected_ancestor_identity.experiment_mode != "formal":
        raise ValueError("visual fork ancestor must retain formal forensic identity")
    ancestor = _ancestor_fields(config)
    if (
        expected_ancestor_identity.source_commit != ancestor["source_commit"]
        or expected_ancestor_identity.source_manifest_identity
        != ancestor["source_manifest_identity"]
        or expected_ancestor_identity.config_identity
        != ancestor["source_config_identity"]
        or expected_ancestor_identity.run_identity != ancestor["run_identity"]
        or expected_ancestor_identity.world_size != ancestor["world_size"]
    ):
        raise ValueError("visual fork expected ancestor identity mismatch")
    checkpoint = Path(str(ancestor["checkpoint_path"]))
    control_digest = _file_sha256(checkpoint / "control.json")
    if control_digest != ancestor["control_sha256"]:
        raise ValueError("visual fork ancestor control SHA mismatch")
    if optimizer.state:
        raise ValueError("visual fork optimizer must be fresh before model-only load")
    optimizer_groups_before = tuple(
        tuple(id(parameter) for parameter in group["params"])
        for group in optimizer.param_groups
    )
    scheduler_before = scheduler.state_dict()
    python_rng_before = random.getstate()
    numpy_rng_before = np.random.get_state()
    torch_rng_before = torch.get_rng_state().clone()
    cuda_rng_before = torch.cuda.get_rng_state().clone() if torch.cuda.is_available() else None
    control = load_query_state_forensic_model_for_debug(
        checkpoint,
        root=root,
        rank=rank,
        expected_identity=expected_ancestor_identity,
        failure_manifest_path=Path(str(ancestor["failure_manifest_path"])),
    )
    if control.global_step != 1605 or not control.forensic_only:
        raise ValueError("visual fork model loader did not authenticate Formal38 forensic update")
    if (
        optimizer.state
        or optimizer_groups_before
        != tuple(
            tuple(id(parameter) for parameter in group["params"])
            for group in optimizer.param_groups
        )
        or scheduler.state_dict() != scheduler_before
        or random.getstate() != python_rng_before
        or np.random.get_state()[0] != numpy_rng_before[0]
        or not np.array_equal(np.random.get_state()[1], numpy_rng_before[1])
        or np.random.get_state()[2:] != numpy_rng_before[2:]
        or not torch.equal(torch.get_rng_state(), torch_rng_before)
        or (
            cuda_rng_before is not None
            and not torch.equal(torch.cuda.get_rng_state(), cuda_rng_before)
        )
    ):
        raise RuntimeError("visual fork model-only load mutated fresh runtime state")
    return QueryStateVisualForkInitializationReceipt(
        ancestor_checkpoint=str(checkpoint.resolve()),
        ancestor_control_sha256=control_digest,
        model_loaded=True,
        optimizer_restored=False,
        scheduler_restored=False,
        rng_restored=False,
        data_cursor_restored=False,
        wandb_cursor_restored=False,
    )


__all__ = [
    "QueryStateVisualForkEvent",
    "QueryStateVisualForkInitializationReceipt",
    "authenticate_visual_fork_ancestor",
    "build_visual_fork_event_plan",
    "initialize_visual_fork_from_forensic_model",
]
