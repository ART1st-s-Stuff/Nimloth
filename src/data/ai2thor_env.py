"""AI2THOR 采集环境适配器。"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np

from src.data.env_adapter import StepResult

try:
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering, Linux64
except ImportError:  # pragma: no cover - 运行时依赖
    Controller = None
    CloudRendering = None
    Linux64 = None


@dataclass
class Ai2ThorEnvConfig:
    scene: str
    seed: int
    image_height: int
    image_width: int
    visibility_distance: float
    grid_size: float
    render_depth_image: bool
    render_instance_segmentation: bool
    platform: str
    cache_dir: str


class Ai2ThorEnvAdapter:
    @staticmethod
    def _resolve_platform(name: str):
        normalized = name.strip().lower()
        if normalized == "cloudrendering":
            return CloudRendering
        if normalized == "linux64":
            return Linux64
        raise ValueError(f"不支持的 ai2thor 平台: {name}")

    def __init__(self, cfg: Ai2ThorEnvConfig) -> None:
        if Controller is None:
            raise RuntimeError("未安装 ai2thor，请先通过 uv 安装依赖。")
        self.cfg = cfg
        os.environ["AI2THOR_BASE_DIR"] = cfg.cache_dir
        self.controller = Controller(
            scene=cfg.scene,
            width=cfg.image_width,
            height=cfg.image_height,
            visibilityDistance=cfg.visibility_distance,
            gridSize=cfg.grid_size,
            renderDepthImage=cfg.render_depth_image,
            renderInstanceSegmentation=cfg.render_instance_segmentation,
            platform=self._resolve_platform(cfg.platform),
        )
        self._rng = np.random.default_rng(cfg.seed)

    def _extract_metadata(self, event: Any) -> dict[str, Any]:
        metadata = event.metadata
        agent_pos = metadata.get("agent", {}).get("position", {})
        visible_objects = [
            obj.get("name", "")
            for obj in metadata.get("objects", [])
            if obj.get("visible", False)
        ]
        target_distance = None
        if metadata.get("objects"):
            # 最小化实现：用首个可见物体距离近似目标距离。
            visible = [obj for obj in metadata["objects"] if obj.get("visible", False)]
            if visible:
                target_distance = float(visible[0].get("distance", 0.0))
        return {
            "target_distance": target_distance,
            "collided": bool(not metadata.get("lastActionSuccess", True)),
            "grasped": any(obj.get("isPickedUp", False) for obj in metadata.get("objects", [])),
            "visible_objects": visible_objects[:20],
            "agent_position": agent_pos,
            "last_action": metadata.get("lastAction", ""),
        }

    def reset(self, episode_id: int) -> StepResult:
        event = self.controller.reset(scene=self.cfg.scene)
        # 每个 episode 随机初始化到可达位置，避免数据集中视角高度重复。
        reachable = self.controller.step(action="GetReachablePositions")
        candidates = reachable.metadata.get("actionReturn") or []
        if candidates:
            idx = int(self._rng.integers(len(candidates)))
            position = candidates[idx]
            yaw = float(self._rng.choice([0.0, 90.0, 180.0, 270.0]))
            pitch = float(self._rng.choice([-30.0, 0.0, 30.0]))
            event = self.controller.step(
                action="TeleportFull",
                position=position,
                rotation={"x": 0.0, "y": yaw, "z": 0.0},
                horizon=pitch,
                standing=True,
                forceAction=True,
            )
            if not event.metadata.get("lastActionSuccess", False):
                event = self.controller.step(action="Pass")
        else:
            event = self.controller.step(action="Pass")
        frame = np.asarray(event.frame, dtype=np.uint8)
        return StepResult(frame=frame, metadata=self._extract_metadata(event))

    def step(self, action: str, step_id: int) -> StepResult:
        del step_id
        event = self.controller.step(action=action)
        frame = np.asarray(event.frame, dtype=np.uint8)
        return StepResult(frame=frame, metadata=self._extract_metadata(event))

    def close(self) -> None:
        if self.controller is not None:
            self.controller.stop()

