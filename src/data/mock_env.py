"""Mock 采集环境。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.data.env_adapter import StepResult


@dataclass
class MockEnvConfig:
    image_height: int
    image_width: int
    seed: int


class MockEnvAdapter:
    def __init__(self, cfg: MockEnvConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed=cfg.seed)

    def _render(self) -> np.ndarray:
        return (self.rng.random((self.cfg.image_height, self.cfg.image_width, 3)) * 255).astype(
            np.uint8
        )

    def _metadata(self, step_id: int) -> dict[str, Any]:
        return {
            "target_distance": max(0.0, 1.0 - 0.06 * step_id),
            "collided": bool(step_id % 7 == 0 and step_id > 0),
            "grasped": bool(step_id >= 6),
            "visible_objects": ["Cup", "Table", "Door"],
        }

    def reset(self, episode_id: int) -> StepResult:
        del episode_id
        return StepResult(frame=self._render(), metadata=self._metadata(step_id=0))

    def step(self, action: str, step_id: int) -> StepResult:
        del action
        return StepResult(frame=self._render(), metadata=self._metadata(step_id=step_id))

    def close(self) -> None:
        return None

