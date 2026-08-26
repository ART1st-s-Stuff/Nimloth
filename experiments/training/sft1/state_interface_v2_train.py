#!/usr/bin/env python3
"""Run launch-locked production smoke/resume/formal SFT1-v2 phases."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
import torch.distributed as dist

from nimloth.training.sft1.controller import assert_clean_resolved_source
from nimloth.training.sft1.experiment_config import load_sft1_v2_config
from nimloth.training.sft1.train_runtime import (
    run_formal_training,
    run_training_smoke,
)


def _distributed_device(expected_world_size: int) -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("SFT1-v2 smoke/formal training requires CUDA")
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != expected_world_size or not 0 <= rank < world_size:
        raise ValueError("torchrun world differs from resolved SFT1-v2 config")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    return rank, world_size, torch.device(f"cuda:{local_rank}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "resume-smoke", "formal"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_sft1_v2_config(args.config)
    assert_clean_resolved_source(config, args.repo_root)
    if args.mode == "resume-smoke" and args.resume_checkpoint is None:
        raise ValueError("resume-smoke requires --resume-checkpoint")
    if args.mode == "smoke" and args.resume_checkpoint is not None:
        raise ValueError("fresh smoke may not receive --resume-checkpoint")
    rank, world_size, device = _distributed_device(config.runtime.world_size)
    wandb_run = None
    try:
        if args.mode == "formal":
            if rank == 0:
                import wandb

                wandb_run = wandb.init(
                    project=config.output.wandb_project,
                    name=config.output.wandb_run_name,
                    id=config.output.wandb_run_id,
                    resume="must" if args.resume_checkpoint else "never",
                    config={
                        "config_identity": config.identity,
                        "source_commit": config.source.expected_commit,
                    },
                )
                if (
                    wandb_run.project != config.output.wandb_project
                    or wandb_run.id != config.output.wandb_run_id
                ):
                    raise RuntimeError("initialized W&B identity differs from resolved config")
            result = run_formal_training(
                config,
                repo_root=args.repo_root,
                rank=rank,
                world_size=world_size,
                device=device,
                seed=args.seed,
                resume_checkpoint=args.resume_checkpoint,
                metric_logger=wandb_run,
            )
        else:
            result = run_training_smoke(
                config,
                repo_root=args.repo_root,
                rank=rank,
                world_size=world_size,
                device=device,
                seed=args.seed,
                resume_checkpoint=(
                    args.resume_checkpoint if args.mode == "resume-smoke" else None
                ),
            )
        if rank == 0:
            print(result)
        dist.barrier()
        return 0
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
