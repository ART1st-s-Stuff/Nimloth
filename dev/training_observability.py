"""Shared realtime training logging helpers.

All training entrypoints should stream metrics to stdout and W&B at optimizer-step
granularity so long-running Slurm jobs are observable while they are running.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


_DEFAULT_ENV_FILES = (
    Path("/project/peilab/atst/flower/.env"),
    Path("/project/peilab/atst/.env"),
)


def load_wandb_env_if_needed() -> Path | None:
    """Load WANDB_* variables from the first project .env that defines a key.

    Values are never printed. Existing environment variables win, so Slurm or a
    caller can override the .env safely.
    """
    if os.getenv("WANDB_API_KEY"):
        return None
    for env_path in _DEFAULT_ENV_FILES:
        if not env_path.exists():
            continue
        loaded = _load_dotenv(env_path)
        if loaded and os.getenv("WANDB_API_KEY"):
            return env_path
    return None


def _load_dotenv(env_path: Path) -> bool:
    loaded = False
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        if not key.startswith("WANDB_"):
            continue
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
        loaded = True
    return loaded


def add_observability_args(parser: argparse.ArgumentParser, *, default_project: str, default_run_name: str) -> None:
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", default_project))
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY", ""))
    parser.add_argument("--wandb-run-name", default=os.getenv("WANDB_RUN_NAME", default_run_name))
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=os.getenv("WANDB_MODE", "online"))
    parser.add_argument("--disable-wandb", action="store_true", help="Emergency override only; training jobs should normally log online.")
    parser.add_argument("--log-every-steps", type=int, default=1, help="Print/log every N optimizer steps. Default 1 = every step.")


def init_wandb(args: argparse.Namespace, *, task_name: str, config: dict[str, Any], output_dir: str | Path) -> Any | None:
    if getattr(args, "disable_wandb", False) or getattr(args, "wandb_mode", "online") == "disabled":
        return None
    load_wandb_env_if_needed()
    import wandb

    wandb_dir = Path(output_dir) / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", str(wandb_dir))
    os.environ["WANDB_MODE"] = str(getattr(args, "wandb_mode", "online"))
    run = wandb.init(
        project=str(getattr(args, "wandb_project", "flower")),
        entity=str(getattr(args, "wandb_entity", "")) or None,
        name=str(getattr(args, "wandb_run_name", "")) or task_name,
        mode=str(getattr(args, "wandb_mode", "online")),
        config=config,
    )
    return run


def emit_metrics(metrics: dict[str, Any], *, wandb_run: Any | None = None, step: int | None = None, prefix: str = "") -> None:
    row = {k: _json_safe(v) for k, v in metrics.items()}
    print(json.dumps(row, ensure_ascii=False), flush=True)
    if wandb_run is not None:
        log_row = {f"{prefix}{k}": v for k, v in row.items() if isinstance(v, (int, float, bool))}
        if log_row:
            wandb_run.log(log_row, step=step)


def _json_safe(value: Any) -> Any:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.detach().cpu().item())
            return value.detach().cpu().tolist()
    except Exception:
        pass
    return value
