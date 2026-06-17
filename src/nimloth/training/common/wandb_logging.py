"""Wandb helpers for SFT2 training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def is_main() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def maybe_init_wandb(args: argparse.Namespace) -> Any | None:
    if not is_main() or getattr(args, "no_wandb", False):
        return None
    if not os.environ.get("WANDB_API_KEY"):
        print(json.dumps({"wandb": "skipped", "reason": "WANDB_API_KEY not set"}))
        return None

    import wandb

    prefix = os.environ.get("WANDB_RUN_PREFIX", "")
    run_name = getattr(args, "wandb_run_name", None) or f"{prefix}sft2-latentwm-value"
    wandb_dir = Path(args.output_dir) / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", str(wandb_dir))

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "nimloth"),
        entity=os.environ.get("WANDB_ENTITY"),
        name=run_name,
        mode=os.environ.get("WANDB_MODE", "online"),
        dir=str(wandb_dir),
        config=vars(args),
    )
    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    return run


def log_train_step(run: Any | None, global_step: int, metrics: dict[str, float]) -> None:
    if run is None:
        return
    payload = {f"train/{k}": v for k, v in metrics.items()}
    payload["global_step"] = global_step
    run.log(payload, step=global_step)


def log_val_epoch(run: Any | None, epoch: int, metrics: dict[str, float]) -> None:
    if run is None:
        return
    payload = {f"val/{k}": v for k, v in metrics.items()}
    payload["epoch"] = epoch
    run.log(payload, step=epoch)
