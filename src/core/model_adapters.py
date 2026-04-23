"""各模块统一 Model 适配器（WM/PM/VLM）。"""

from __future__ import annotations

from typing import Any

from src.core.interfaces import Model


class WMModelAdapter(Model):
    """将 WM 训练步骤包装为统一接口。"""

    def __init__(self, train_step_fn: Any, eval_step_fn: Any | None = None) -> None:
        self._train_step_fn = train_step_fn
        self._eval_step_fn = eval_step_fn or train_step_fn

    def train_step(self, batch: Any) -> dict[str, Any]:
        return dict(self._train_step_fn(batch))

    def eval_step(self, batch: Any) -> dict[str, Any]:
        return dict(self._eval_step_fn(batch))


class PMModelAdapter(Model):
    """PM 接口占位实现，供后续训练入口平滑接入。"""

    def train_step(self, batch: Any) -> dict[str, Any]:
        return {"status": "pm-train-step-not-implemented", "batch": batch}

    def eval_step(self, batch: Any) -> dict[str, Any]:
        return {"status": "pm-eval-step-not-implemented", "batch": batch}


class VLMModelAdapter(Model):
    """VLM 接口占位实现，供后续训练入口平滑接入。"""

    def train_step(self, batch: Any) -> dict[str, Any]:
        return {"status": "vlm-train-step-not-implemented", "batch": batch}

    def eval_step(self, batch: Any) -> dict[str, Any]:
        return {"status": "vlm-eval-step-not-implemented", "batch": batch}
