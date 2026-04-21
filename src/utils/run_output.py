"""统一管理任务输出目录。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.utils.io import ensure_dir


def build_run_output_dir(outputs_root: str, phase: str, task: str) -> Path:
    """按 outputs/phase/task/datetime 构建并创建输出目录。"""
    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    return ensure_dir(Path(outputs_root) / phase / task / timestamp)

