"""训练数据 Provider 实现。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from src.core.interfaces import DataProvider
from src.utils.run_output import read_run_status, resolve_training_run_dir, write_run_status


class WMDataProvider(DataProvider):
    """为 WM 训练提供 train/val/test 统一访问。"""

    def __init__(
        self,
        *,
        train_loader: Iterable[Any],
        val_loader: Iterable[Any] | None = None,
        test_loader: Iterable[Any] | None = None,
        path_segments: list[str] | None = None,
    ) -> None:
        self._train_loader = train_loader
        self._val_loader = val_loader if val_loader is not None else ()
        self._test_loader = test_loader if test_loader is not None else ()
        self._path_segments = path_segments if path_segments else []

    def resolve_run_dir(self, *, force_new: bool = False) -> tuple[Path, bool]:
        return resolve_training_run_dir(path_segments=self._path_segments, force_new=force_new)

    def mark_running(self, run_dir: str | Path, **extra: object) -> None:
        write_run_status(run_dir, "running", **extra)

    def mark_completed(self, run_dir: str | Path, **extra: object) -> None:
        write_run_status(run_dir, "completed", **extra)

    def mark_failed(self, run_dir: str | Path, **extra: object) -> None:
        write_run_status(run_dir, "failed", **extra)

    def read_status(self, run_dir: str | Path) -> dict[str, Any]:
        return read_run_status(run_dir)

    def train(self) -> Iterable[Any]:
        return self._train_loader

    def val(self) -> Iterable[Any]:
        return self._val_loader

    def test(self) -> Iterable[Any]:
        return self._test_loader
