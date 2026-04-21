"""采集环境适配器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass
class StepResult:
    """单步执行结果。"""

    frame: np.ndarray
    metadata: dict[str, Any]


class EnvAdapter(Protocol):
    """统一采集后端接口。"""

    def reset(self, episode_id: int) -> StepResult:
        ...

    def step(self, action: str, step_id: int) -> StepResult:
        ...

    def close(self) -> None:
        ...

