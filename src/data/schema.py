"""Phase 1 数据结构定义。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class FrameState:
    """单帧状态信息。"""

    episode_id: int
    step_id: int
    image_path: str
    action: str
    action_id: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

