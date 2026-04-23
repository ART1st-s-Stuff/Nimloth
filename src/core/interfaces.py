"""训练接口抽象：Storage/Data/Model/ModelProvider。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable


class StorageProvider(ABC):
    """提供运行目录与状态管理能力。"""

    @abstractmethod
    def resolve_run_dir(self, *, force_new: bool = False) -> tuple[Path, bool]:
        """返回 (run_dir, resumed)。"""

    @abstractmethod
    def mark_running(self, run_dir: str | Path, **extra: object) -> None:
        pass

    @abstractmethod
    def mark_completed(self, run_dir: str | Path, **extra: object) -> None:
        pass

    @abstractmethod
    def mark_failed(self, run_dir: str | Path, **extra: object) -> None:
        pass

    @abstractmethod
    def read_status(self, run_dir: str | Path) -> dict[str, Any]:
        pass


class DataProvider(StorageProvider, ABC):
    """统一的数据切分访问契约。"""

    @abstractmethod
    def train(self) -> Iterable[Any]:
        pass

    @abstractmethod
    def val(self) -> Iterable[Any]:
        pass

    @abstractmethod
    def test(self) -> Iterable[Any]:
        pass


class Model(ABC):
    """统一模型训练接口。"""

    @abstractmethod
    def train_step(self, batch: Any) -> dict[str, Any]:
        pass

    @abstractmethod
    def eval_step(self, batch: Any) -> dict[str, Any]:
        pass

    def train(self, data_provider: DataProvider) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for batch in data_provider.train():
            metrics = self.train_step(batch)
        return metrics

    def test(self, data_provider: DataProvider) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for batch in data_provider.test():
            metrics = self.eval_step(batch)
        return metrics


class ModelProvider(StorageProvider, ABC):
    """统一模型产物与 checkpoint 管理。"""

    @abstractmethod
    def save(self, run_dir: str | Path, artifacts: dict[str, Any]) -> dict[str, Path]:
        pass

    @abstractmethod
    def save_checkpoint(self, run_dir: str | Path, state: dict[str, Any], *, is_final: bool = False) -> Path:
        pass

    @abstractmethod
    def load_checkpoint(self, run_dir: str | Path, *, prefer_final: bool = False) -> dict[str, Any] | None:
        pass
