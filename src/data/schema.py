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
    move_ahead_distance: float
    delta_yaw: float
    delta_pitch: float
    recovery_active: bool
    recovery_stage: str
    near_wall: bool
    pitch_control_active: bool
    pitch_over_limit_streak: int
    agent_horizon: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

