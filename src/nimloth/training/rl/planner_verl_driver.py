"""Driver-owned transaction boundary for Planner VERL/FSDP updates."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch


PLANNER_FSDP_CHECKPOINT_SCHEMA_VERSION = 1


class PlannerWorkerGroup(Protocol):
    world_size: int

    def begin_planner_update(self, update_id: str) -> list[bool]: ...

    def backward_planner_micro_batch(
        self,
        rank_batches: list[Any],
    ) -> list[dict[str, float]]: ...

    def finish_planner_update(
        self,
        update_id: str,
    ) -> list[dict[str, float]]: ...

    def save_planner_checkpoint(
        self,
        path: str,
        update_id: str,
        global_step: int,
    ) -> list[bool]: ...

    def mark_planner_checkpoint_succeeded(
        self,
        update_id: str,
    ) -> list[bool]: ...


class FreshConsumptionOwner(Protocol):
    def begin_consumption(self, *, output_dir: Path, global_step: int) -> str: ...

    def commit_consumption(
        self,
        consumption_id: str,
        *,
        checkpoint_path: Path,
        global_step: int,
    ) -> None: ...


@dataclass(frozen=True)
class PlannerVERLDriverResult:
    global_step: int
    update_id: str
    checkpoint_path: Path
    rank_metrics: tuple[dict[str, float], ...]


def _checkpoint_shard_path(
    path: Path,
    *,
    kind: str,
    world_size: int,
    rank: int,
) -> Path:
    return path / f"{kind}_world_size_{world_size}_rank_{rank}.pt"


def validate_planner_fsdp_checkpoint(
    path: Path,
    *,
    world_size: int,
    global_step: int,
    update_id: str,
) -> dict[str, Any]:
    """Validate every FSDP shard and the authoritative update sidecar."""

    checkpoint = Path(path)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"planner checkpoint directory is missing: {checkpoint}")
    if world_size < 1:
        raise ValueError("planner checkpoint world_size must be positive")
    for rank in range(world_size):
        for kind in ("model", "optim", "extra_state"):
            shard = _checkpoint_shard_path(
                checkpoint,
                kind=kind,
                world_size=world_size,
                rank=rank,
            )
            if not shard.is_file() or shard.stat().st_size < 1:
                raise FileNotFoundError(
                    f"missing planner rank checkpoint shard: {shard}"
                )
    state_path = checkpoint / "rl_state.pt"
    if not state_path.is_file() or state_path.stat().st_size < 1:
        raise FileNotFoundError(
            f"planner checkpoint is missing rl_state.pt: {state_path}"
        )
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    expected = {
        "checkpoint_schema_version": PLANNER_FSDP_CHECKPOINT_SCHEMA_VERSION,
        "optimizer_state_layout": "rank_sharded_fsdp",
        "optimizer_world_size": world_size,
        "training_world_size": world_size,
        "global_step": global_step,
        "update_id": update_id,
    }
    for name, value in expected.items():
        if state.get(name) != value:
            raise ValueError(
                "planner checkpoint metadata mismatch for "
                f"{name}: saved={state.get(name)!r}, expected={value!r}"
            )
    completed = state.get("completed_update_ids")
    if not isinstance(completed, list) or update_id not in completed:
        raise ValueError(
            "planner checkpoint must persist the completed update identity"
        )
    if any(not isinstance(item, str) or not item for item in completed):
        raise ValueError("planner checkpoint completed identities are invalid")
    return state


def _require_all_rank_acks(
    values: Any,
    *,
    world_size: int,
    operation: str,
) -> None:
    if not isinstance(values, list) or values != [True] * world_size:
        raise RuntimeError(
            f"planner worker {operation} did not acknowledge every rank: {values!r}"
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class PlannerVERLUpdateDriver:
    """Run one fresh update and commit it only after atomic checkpoint publish."""

    def __init__(
        self,
        *,
        worker_group: PlannerWorkerGroup,
        collector: FreshConsumptionOwner,
    ) -> None:
        self._workers = worker_group
        self._collector = collector

    def _validate_rounds(
        self,
        *,
        rank_rounds: tuple[tuple[Any, ...], ...],
    ) -> None:
        if not rank_rounds:
            raise ValueError("planner update requires at least one backward round")
        world_size = int(self._workers.world_size)
        if world_size < 1:
            raise ValueError("planner worker group must not be empty")
        for round_index, rank_batches in enumerate(rank_rounds):
            if len(rank_batches) != world_size:
                raise ValueError(
                    "planner backward round requires one nonempty batch per rank: "
                    f"round={round_index}, batches={len(rank_batches)}, "
                    f"world_size={world_size}"
                )
            row_counts = tuple(
                len(batch) if hasattr(batch, "__len__") else None
                for batch in rank_batches
            )
            known_counts = tuple(count for count in row_counts if count is not None)
            if known_counts and (
                len(known_counts) != world_size or len(set(known_counts)) != 1
            ):
                raise ValueError(
                    "planner nested FSDP requires equal row counts on every rank: "
                    f"round={round_index}, row_counts={row_counts}"
                )
            provisional_ids: set[str] = set()
            batch_signatures: set[tuple[Any, ...]] = set()
            for rank, batch in enumerate(rank_batches):
                if batch is None or (
                    hasattr(batch, "__len__") and len(batch) < 1
                ):
                    raise ValueError(
                        "planner backward round requires one nonempty batch per rank: "
                        f"round={round_index}, rank={rank}"
                    )
                meta_info = getattr(batch, "meta_info", None)
                provisional_id = (
                    meta_info.get("update_id")
                    if isinstance(meta_info, dict)
                    else None
                )
                if not isinstance(provisional_id, str) or not provisional_id:
                    raise ValueError(
                        "planner rank batch requires a provisional update identity: "
                        f"round={round_index}, rank={rank}"
                    )
                provisional_ids.add(provisional_id)
                batch_signatures.add(
                    (
                        meta_info.get("schema_version"),
                        meta_info.get("objective"),
                        meta_info.get("total_transitions"),
                        meta_info.get("has_dino_grid_targets"),
                        meta_info.get("behavior_matched"),
                        meta_info.get("diagnostic_only"),
                    )
                )
                if (
                    meta_info.get("behavior_matched") is not True
                    or meta_info.get("diagnostic_only") is not False
                ):
                    raise ValueError(
                        "transactional planner updates reject nonbehavior "
                        f"diagnostics: round={round_index}, rank={rank}"
                    )
            if len(provisional_ids) != 1:
                raise ValueError(
                    "planner rank batches disagree on provisional update identity: "
                    f"round={round_index}"
                )
            if len(batch_signatures) != 1:
                raise ValueError(
                    "planner rank batches disagree on objective metadata: "
                    f"round={round_index}"
                )

    @staticmethod
    def _bind_consumption_identity(
        rank_rounds: tuple[tuple[Any, ...], ...],
        consumption_id: str,
    ) -> None:
        if not consumption_id:
            raise RuntimeError("fresh collector returned an empty consumption identity")
        for rank_batches in rank_rounds:
            for batch in rank_batches:
                batch.meta_info = {
                    **batch.meta_info,
                    "update_id": consumption_id,
                }

    def run_update(
        self,
        *,
        output_dir: Path,
        current_global_step: int,
        rank_rounds: tuple[tuple[Any, ...], ...],
    ) -> PlannerVERLDriverResult:
        """Execute exactly one optimizer step and publish its durable checkpoint.

        Any exception after the fresh claim is created leaves that claim in
        progress. In particular, this method never attempts a distributed abort
        after a Ray/FSDP call may have partially executed.
        """

        if current_global_step < 0:
            raise ValueError("current_global_step must not be negative")
        self._validate_rounds(rank_rounds=rank_rounds)
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        next_global_step = current_global_step + 1
        checkpoint = output / f"global_step_{next_global_step:06d}"
        if checkpoint.exists():
            raise FileExistsError(
                f"planner checkpoint already exists before update: {checkpoint}"
            )
        temporary = output / f".{checkpoint.name}.tmp-{uuid.uuid4().hex}"
        consumption_id = self._collector.begin_consumption(
            output_dir=output,
            global_step=current_global_step,
        )
        update_id = consumption_id
        self._bind_consumption_identity(rank_rounds, update_id)

        begin_acks = self._workers.begin_planner_update(update_id)
        _require_all_rank_acks(
            begin_acks,
            world_size=self._workers.world_size,
            operation="begin",
        )
        for rank_batches in rank_rounds:
            # Pinned RayWorkerGroup only slices list arguments. A tuple would be
            # broadcast whole to every worker and violate rank ownership.
            self._workers.backward_planner_micro_batch(list(rank_batches))
        rank_metrics = self._workers.finish_planner_update(update_id)
        if not isinstance(rank_metrics, list) or len(rank_metrics) != self._workers.world_size:
            raise RuntimeError(
                "planner optimizer finish did not return one metric row per rank"
            )

        save_acks = self._workers.save_planner_checkpoint(
            str(temporary),
            update_id,
            next_global_step,
        )
        _require_all_rank_acks(
            save_acks,
            world_size=self._workers.world_size,
            operation="checkpoint save",
        )
        validate_planner_fsdp_checkpoint(
            temporary,
            world_size=self._workers.world_size,
            global_step=next_global_step,
            update_id=update_id,
        )
        _fsync_directory(temporary)
        temporary.replace(checkpoint)
        _fsync_directory(output)

        self._collector.commit_consumption(
            consumption_id,
            checkpoint_path=checkpoint,
            global_step=next_global_step,
        )
        mark_acks = self._workers.mark_planner_checkpoint_succeeded(update_id)
        _require_all_rank_acks(
            mark_acks,
            world_size=self._workers.world_size,
            operation="checkpoint completion",
        )
        return PlannerVERLDriverResult(
            global_step=next_global_step,
            update_id=update_id,
            checkpoint_path=checkpoint,
            rank_metrics=tuple(rank_metrics),
        )


__all__ = [
    "PLANNER_FSDP_CHECKPOINT_SCHEMA_VERSION",
    "PlannerVERLDriverResult",
    "PlannerVERLUpdateDriver",
    "validate_planner_fsdp_checkpoint",
]
