"""文件系统实现的 StorageProvider / ModelProvider。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.core.interfaces import ModelProvider
from src.utils.io import ensure_dir
from src.utils.run_output import (
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    RUN_STATUS_RUNNING,
    read_run_status,
    resolve_training_run_dir,
    write_run_status,
)


class FileSystemModelProvider(ModelProvider):
    """基于目录结构管理 run 状态与模型权重。"""

    def __init__(
        self,
        *,
        path_segments: list[str],
        checkpoint_name: str = "checkpoint_last.pt",
        final_checkpoint_name: str = "checkpoint_final.pt",
    ) -> None:
        self.path_segments = path_segments
        self.checkpoint_name = checkpoint_name
        self.final_checkpoint_name = final_checkpoint_name

    def resolve_run_dir(self, *, force_new: bool = False) -> tuple[Path, bool]:
        return resolve_training_run_dir(path_segments=self.path_segments, force_new=force_new)

    def mark_running(self, run_dir: str | Path, **extra: object) -> None:
        write_run_status(run_dir, RUN_STATUS_RUNNING, **extra)

    def mark_completed(self, run_dir: str | Path, **extra: object) -> None:
        write_run_status(run_dir, RUN_STATUS_COMPLETED, **extra)

    def mark_failed(self, run_dir: str | Path, **extra: object) -> None:
        write_run_status(run_dir, RUN_STATUS_FAILED, **extra)

    def read_status(self, run_dir: str | Path) -> dict[str, Any]:
        return read_run_status(run_dir)

    def save(self, run_dir: str | Path, artifacts: dict[str, Any]) -> dict[str, Path]:
        root = ensure_dir(run_dir)
        saved: dict[str, Path] = {}
        for name, value in artifacts.items():
            if isinstance(value, Path):
                saved[name] = value
                continue
            target = root / f"{name}.pt"
            torch.save(value, target)
            saved[name] = target
        return saved

    def save_checkpoint(self, run_dir: str | Path, state: dict[str, Any], *, is_final: bool = False) -> Path:
        root = ensure_dir(run_dir)
        filename = self.final_checkpoint_name if is_final else self.checkpoint_name
        path = root / filename
        torch.save(state, path)
        return path

    def load_checkpoint(self, run_dir: str | Path, *, prefer_final: bool = False) -> dict[str, Any] | None:
        root = Path(run_dir)
        candidates: list[Path] = []
        if prefer_final:
            candidates.extend([root / self.final_checkpoint_name, root / self.checkpoint_name])
        else:
            candidates.extend([root / self.checkpoint_name, root / self.final_checkpoint_name])
        for path in candidates:
            if path.exists():
                payload = torch.load(path, map_location="cpu")
                if isinstance(payload, dict):
                    return payload
        return None

    def is_completed(self, run_dir: str | Path) -> bool:
        return str(self.read_status(run_dir).get("status", "")).lower() == RUN_STATUS_COMPLETED
