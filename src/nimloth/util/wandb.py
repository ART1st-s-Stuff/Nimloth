"""可恢复的 W&B 初始化和带 namespace 的指标记录。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


MetricDefinition = tuple[str, str | None]


def init_wandb_run(
    *,
    rank: int,
    output_dir: Path,
    enabled: bool,
    default_project: str,
    run_name: str | None,
    config: dict[str, Any],
    metric_definitions: Iterable[MetricDefinition] = (),
) -> Any | None:
    """只在 rank 0 初始化一个可恢复的 W&B run。

    project、entity、name、mode、目录以及显式 run id 仍以环境变量为准；
    未指定 run id 时，从实验目录保存的 id 恢复。
    """

    if rank != 0 or not enabled:
        return None
    mode = os.environ.get("WANDB_MODE", "online")
    if mode != "disabled" and not os.environ.get("WANDB_API_KEY"):
        print(json.dumps({"wandb": "skipped", "reason": "WANDB_API_KEY not set"}))
        return None

    import wandb

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_dir = Path(os.environ.get("WANDB_DIR", output_dir / "wandb"))
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", str(wandb_dir))

    run_id_path = output_dir / "wandb_run_id.txt"
    requested_run_id = os.environ.get("WANDB_RUN_ID")
    if requested_run_id is None and run_id_path.is_file():
        requested_run_id = run_id_path.read_text(encoding="utf-8").strip() or None

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", default_project),
        entity=os.environ.get("WANDB_ENTITY"),
        name=os.environ.get("WANDB_RUN_NAME") or run_name,
        id=requested_run_id,
        resume="allow" if requested_run_id is not None else None,
        mode=mode,
        dir=str(wandb_dir),
        config=config,
    )
    run_id_path.write_text(f"{run.id}\n", encoding="utf-8")
    for metric_name, step_metric in metric_definitions:
        if step_metric is None:
            wandb.define_metric(metric_name)
        else:
            wandb.define_metric(metric_name, step_metric=step_metric)
    print(
        json.dumps(
            {
                "wandb": "initialized",
                "run_id": run.id,
                "resume": requested_run_id is not None,
            }
        )
    )
    return run


def log_metrics(
    run: Any | None,
    *,
    namespace: str,
    metrics: dict[str, float],
    step: int,
    context: dict[str, int | float | str] | None = None,
) -> None:
    if run is None:
        return
    payload: dict[str, Any] = {
        f"{namespace}/{key}": value for key, value in metrics.items()
    }
    if context:
        payload.update(context)
    run.log(payload, step=step)
