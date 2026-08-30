#!/usr/bin/env python3
"""Preflight or run one approved Query-State pilot/formal torchrun process.

The Python owner never submits Slurm, retries a failed run, extends pilot into
formal, enters SFT2, or materializes a deployable artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
    raise RuntimeError("Query-State training requires PYTHONDONTWRITEBYTECODE=1")
_PYCACHE = Path(os.environ.get("PYTHONPYCACHEPREFIX", ""))
if not _PYCACHE.is_absolute() or _PYCACHE == _REPO_ROOT or _REPO_ROOT in _PYCACHE.parents:
    raise RuntimeError("Query-State training requires an absolute external pycache")
sys.dont_write_bytecode = True
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Query-State resolved launch config must be immutable JSON") from error
    if not isinstance(value, dict):
        raise ValueError("Query-State resolved launch config must be a mapping")
    return value


def _canonical_run_argv(config_path: Path) -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        str(Path(__file__).resolve()),
        "--config",
        str(config_path.resolve()),
        "--phase",
        "run",
    ]


def _distributed_device(config: Any) -> tuple[int, int, Any]:
    import torch
    import torch.distributed as dist

    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise ValueError("Query-State training must run under torchrun: " + missing[0])
    if not torch.cuda.is_available():
        raise RuntimeError("Query-State training backend requires CUDA")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    group_rank = int(os.environ["GROUP_RANK"])
    if (
        world_size != int(config.resources["world_size"])
        or local_world_size != int(config.resources["gpus_per_node"])
        or torch.cuda.device_count() != local_world_size
        or not 0 <= rank < world_size
        or not 0 <= local_rank < local_world_size
        or not 0 <= group_rank < int(config.resources["nodes"])
    ):
        raise ValueError("Query-State torchrun topology differs from launch lock")
    device_name = torch.cuda.get_device_name(local_rank)
    if device_name not in config.resources["gpu_model_allowlist"]:
        raise ValueError("Query-State GPU model is outside the approved allowlist")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(str(config.resources["backend"]))
    if dist.get_rank() != rank or dist.get_world_size() != world_size:
        raise ValueError("Query-State process-group identity mismatch")
    return rank, world_size, torch.device(f"cuda:{local_rank}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("preflight", "run"), required=True)
    args = parser.parse_args(argv)

    from nimloth.training.sft1.query_state_training_config import (
        parse_query_state_training_config,
        reapply_locked_wandb_environment,
    )
    from nimloth.training.sft1.query_state_training_preflight import (
        assert_query_state_training_backend_ready,
        verify_query_state_training_preflight,
    )

    config_path = args.config.resolve()
    config = parse_query_state_training_config(_load(config_path))
    canonical_run = _canonical_run_argv(config_path)
    if canonical_run != list(config.command["argv"]):
        raise ValueError("Query-State entry command differs from resolved launch contract")
    preflight = verify_query_state_training_preflight(
        config,
        repo_root=Path(str(config.source["repo_root"])),
        current_argv=canonical_run,
        environ=os.environ,
    )
    if args.phase == "preflight":
        print(json.dumps({
            "config_identity": config.identity,
            "lifecycle_state": config.lifecycle_state,
            "mode": config.mode,
            "verified_file_count": preflight.verified_file_count,
            "launch_executed": False,
            "cuda_entered": False,
        }, sort_keys=True))
        return 0

    assert_query_state_training_backend_ready(config, preflight=preflight)
    if config.mode == "pilot":
        os.environ["WANDB_MODE"] = "disabled"
    else:
        os.environ.update(reapply_locked_wandb_environment(config, os.environ))
    import torch.distributed as dist

    from nimloth.training.sft1.query_state_training_backend import (
        run_query_state_training,
    )

    rank, world_size, device = _distributed_device(config)
    try:
        result = run_query_state_training(
            config,
            repo_root=Path(str(config.source["repo_root"])),
            device=device,
            rank=rank,
            world_size=world_size,
        )
        if world_size > 1:
            dist.barrier()
        if rank == 0:
            print(json.dumps({
                "mode": result.mode,
                "final_update": result.final_update,
                "final_checkpoint": result.final_checkpoint,
                "validation_cursor": result.validation_cursor,
                "log_cursor": result.log_cursor,
                "tracking_cursor": result.tracking_cursor,
                "tracking_incomplete": result.tracking_incomplete,
                "terminal_epoch": result.terminal_epoch,
                "terminal_reason": result.terminal_reason,
                "automatic_formal_extension": False,
                "automatic_sft2_authorization": False,
                "automatic_export": False,
            }, sort_keys=True))
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
