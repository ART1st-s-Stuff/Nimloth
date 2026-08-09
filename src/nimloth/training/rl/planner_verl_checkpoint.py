"""Sharded checkpoint I/O for the composite Planner FSDP root."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    StateDictType,
)

from nimloth.training.rl.planner_verl_driver import (
    PLANNER_FSDP_CHECKPOINT_SCHEMA_VERSION,
    validate_planner_fsdp_checkpoint,
)


class PlannerFSDPCheckpointManager:
    """Save/load exact-world-size FSDP model, optimizer, and RNG shards."""

    def __init__(
        self,
        *,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        if not isinstance(model, FSDP):
            raise TypeError("planner checkpoint model must be an FSDP root")
        if not dist.is_initialized():
            raise RuntimeError("planner checkpoint requires distributed initialization")
        self.model = model
        self.optimizer = optimizer
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def _rng_state(self) -> dict[str, Any]:
        return {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(torch.cuda.current_device()),
        }

    def _load_rng_state(self, state: dict[str, Any]) -> None:
        torch.set_rng_state(state["cpu"])
        torch.cuda.set_rng_state(
            state["cuda"],
            device=torch.cuda.current_device(),
        )

    @staticmethod
    def _atomic_torch_save(payload: Any, path: Path) -> None:
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _raise_if_rank_error(
        self,
        error: str | None,
        *,
        operation: str,
    ) -> None:
        errors: list[str | None] = [None] * self.world_size
        dist.all_gather_object(errors, error)
        failures = tuple(
            f"rank{rank}={message}"
            for rank, message in enumerate(errors)
            if message is not None
        )
        if failures:
            raise RuntimeError(
                f"planner checkpoint {operation} failed: " + "; ".join(failures)
            )

    def save(
        self,
        path: Path,
        *,
        update_id: str,
        global_step: int,
        completed_update_ids: tuple[str, ...],
    ) -> None:
        checkpoint = Path(path)
        destination_error = (
            f"planner checkpoint destination already exists: {checkpoint}"
            if self.rank == 0 and checkpoint.exists()
            else None
        )
        errors = [destination_error]
        dist.broadcast_object_list(errors, src=0)
        if errors[0] is not None:
            raise FileExistsError(str(errors[0]))
        mkdir_error: str | None = None
        if self.rank == 0:
            try:
                checkpoint.mkdir(parents=True)
            except Exception as error:
                mkdir_error = repr(error)
        self._raise_if_rank_error(mkdir_error, operation="directory creation")

        state_config = ShardedStateDictConfig(offload_to_cpu=True)
        optim_config = ShardedOptimStateDictConfig(offload_to_cpu=True)
        with FSDP.state_dict_type(
            self.model,
            StateDictType.SHARDED_STATE_DICT,
            state_config,
            optim_config,
        ):
            model_state = self.model.state_dict()
            optimizer_state = FSDP.optim_state_dict(
                self.model,
                self.optimizer,
            )
        shard_error: str | None = None
        try:
            self._atomic_torch_save(
                model_state,
                checkpoint
                / f"model_world_size_{self.world_size}_rank_{self.rank}.pt",
            )
            self._atomic_torch_save(
                optimizer_state,
                checkpoint
                / f"optim_world_size_{self.world_size}_rank_{self.rank}.pt",
            )
            self._atomic_torch_save(
                {"rng": self._rng_state()},
                checkpoint
                / f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt",
            )
        except Exception as error:
            shard_error = repr(error)
        self._raise_if_rank_error(shard_error, operation="rank shard write")

        sidecar_error: str | None = None
        if self.rank == 0:
            try:
                self._atomic_torch_save(
                    {
                        "checkpoint_schema_version": (
                            PLANNER_FSDP_CHECKPOINT_SCHEMA_VERSION
                        ),
                        "optimizer_state_layout": "rank_sharded_fsdp",
                        "optimizer_world_size": self.world_size,
                        "training_world_size": self.world_size,
                        "global_step": int(global_step),
                        "update_id": update_id,
                        "completed_update_ids": list(completed_update_ids),
                    },
                    checkpoint / "rl_state.pt",
                )
                descriptor = os.open(checkpoint, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            except Exception as error:
                sidecar_error = repr(error)
        self._raise_if_rank_error(sidecar_error, operation="sidecar write")

    def load(self, path: Path) -> dict[str, Any]:
        checkpoint = Path(path)
        state: dict[str, Any] | None = None
        model_state: dict[str, Any] | None = None
        optimizer_state: dict[str, Any] | None = None
        extra_state: dict[str, Any] | None = None
        load_error: str | None = None
        try:
            raw_state = torch.load(
                checkpoint / "rl_state.pt",
                map_location="cpu",
                weights_only=False,
            )
            if not isinstance(raw_state, dict):
                raise ValueError("planner checkpoint rl_state must be a mapping")
            state = validate_planner_fsdp_checkpoint(
                checkpoint,
                world_size=self.world_size,
                global_step=int(raw_state["global_step"]),
                update_id=str(raw_state["update_id"]),
            )
            model_state = torch.load(
                checkpoint
                / f"model_world_size_{self.world_size}_rank_{self.rank}.pt",
                map_location="cpu",
                weights_only=False,
            )
            optimizer_state = torch.load(
                checkpoint
                / f"optim_world_size_{self.world_size}_rank_{self.rank}.pt",
                map_location="cpu",
                weights_only=False,
            )
            extra_state = torch.load(
                checkpoint
                / f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt",
                map_location="cpu",
                weights_only=False,
            )
        except Exception as error:
            load_error = repr(error)
        self._raise_if_rank_error(load_error, operation="rank shard preflight")
        if not all(
            isinstance(item, dict)
            for item in (state, model_state, optimizer_state, extra_state)
        ):
            raise RuntimeError("planner checkpoint preflight produced invalid state")

        state_config = ShardedStateDictConfig(offload_to_cpu=True)
        optim_config = ShardedOptimStateDictConfig(offload_to_cpu=True)
        with FSDP.state_dict_type(
            self.model,
            StateDictType.SHARDED_STATE_DICT,
            state_config,
            optim_config,
        ):
            self.model.load_state_dict(model_state)
            optimizer_to_load = FSDP.optim_state_dict_to_load(
                self.model,
                self.optimizer,
                optimizer_state,
            )
            self.optimizer.load_state_dict(optimizer_to_load)
        self._load_rng_state(extra_state["rng"])
        dist.barrier()
        return state


__all__ = ["PlannerFSDPCheckpointManager"]
