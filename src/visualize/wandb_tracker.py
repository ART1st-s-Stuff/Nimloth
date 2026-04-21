"""实验观测：W&B + 统一日志接口。"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os

import wandb

from src.utils.console import info, warn
from src.utils.env import get_env, require_env
from src.utils.io import ensure_dir


class ExperimentTracker:
    def __init__(self, run: wandb.sdk.wandb_run.Run) -> None:
        self.run = run

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        wandb.log(metrics, step=step)

    def log_artifact_path(self, name: str, path: str | Path, artifact_type: str = "file") -> None:
        target = Path(path)
        if not target.exists():
            raise RuntimeError(f"artifact 路径不存在: {target}")
        artifact = wandb.Artifact(name=name, type=artifact_type)
        if target.is_dir():
            artifact.add_dir(str(target))
        else:
            artifact.add_file(str(target))
        self.run.log_artifact(artifact)

    def log_rollout_bundle(
        self,
        run_name: str,
        observation_dir: str | Path,
        prompt_path: str | Path,
        cot_path: str | Path,
    ) -> None:
        """未来 VLM rollout 默认上传入口。"""
        self.log_artifact_path(f"{run_name}-observations", observation_dir, artifact_type="rollout")
        self.log_artifact_path(f"{run_name}-prompt", prompt_path, artifact_type="rollout")
        self.log_artifact_path(f"{run_name}-cot", cot_path, artifact_type="rollout")

    def finish(self) -> None:
        wandb.finish()


def init_tracker(task_name: str, config: dict[str, Any]) -> ExperimentTracker:
    project = get_env("WANDB_PROJECT", "flower")
    entity = get_env("WANDB_ENTITY", None)
    mode = get_env("WANDB_MODE", "online")
    run_name_prefix = get_env("WANDB_RUN_PREFIX", "exp")

    if mode not in {"online", "offline", "disabled"}:
        raise RuntimeError(f"WANDB_MODE 非法值: {mode}")

    if mode == "online":
        api_key = get_env("WANDB_API_KEY", None)
        if not api_key:
            warn("WANDB_API_KEY 未配置，W&B 自动切换为 offline。")
            mode = "offline"
        else:
            require_env("WANDB_API_KEY")

    wandb_root = Path(get_env("WANDB_DIR", "wandb") or "wandb")
    ensure_dir(wandb_root)
    ensure_dir(wandb_root / "artifacts")
    ensure_dir(wandb_root / "cache")
    ensure_dir(wandb_root / "tmp")
    os.environ["WANDB_DIR"] = str(wandb_root)
    os.environ["WANDB_ARTIFACT_DIR"] = str(wandb_root / "artifacts")
    os.environ["WANDB_CACHE_DIR"] = str(wandb_root / "cache")
    os.environ["WANDB_DATA_DIR"] = str(wandb_root / "cache")
    os.environ["TMPDIR"] = str(wandb_root / "tmp")

    run = wandb.init(
        project=project,
        entity=entity,
        mode=mode,
        name=f"{run_name_prefix}-{task_name}",
        config=config,
    )
    info(f"W&B run 已启动: task={task_name}, mode={mode}")
    return ExperimentTracker(run=run)

