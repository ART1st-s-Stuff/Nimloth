"""统一管理任务输出目录。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.utils.io import ensure_dir


def _find_latest_run_dir(task_root: Path) -> Path | None:
    candidates = [p for p in task_root.glob("*/*") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def build_run_output_dir(outputs_root: str, phase: str, task: str, resume: bool = False) -> Path:
    """按 outputs/phase/task/datetime 构建并创建输出目录。"""
    task_root = ensure_dir(Path(outputs_root) / phase / task)
    if resume:
        latest = _find_latest_run_dir(task_root)
        if latest is not None:
            return latest
    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    return ensure_dir(task_root / timestamp)

