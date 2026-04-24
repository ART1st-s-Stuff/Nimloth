"""AI2THOR 采集环境适配器。"""

from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess
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


def _available_gpu_count() -> int:
    """通过 nvidia-smi 探测当前可见的 GPU 数量。"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return max(1, result.stdout.count("\n"))
        return 1
    except Exception:
        return 1


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
    gpu_device: int | None = None


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
            gpu_device=cfg.gpu_device,
            server_timeout=60.0,
            server_start_timeout=120.0,
        )
        self._rng = np.random.default_rng(cfg.seed)
        self._reachable_cache: list[dict[str, float]] | None = None

    def _extract_metadata(self, event: Any) -> dict[str, Any]:
        metadata = event.metadata
        agent_pos = metadata.get("agent", {}).get("position", {})
        agent_rot = metadata.get("agent", {}).get("rotation", {})
        agent_horizon = float(metadata.get("agent", {}).get("cameraHorizon", 0.0))
        visible_objects = [
            obj.get("name", "")
            for obj in metadata.get("objects", [])
            if obj.get("visible", False)
        ]
        target_distance = None
        if metadata.get("objects"):
            visible = [obj for obj in metadata["objects"] if obj.get("visible", False)]
            if visible:
                target_distance = float(visible[0].get("distance", 0.0))
        return {
            "target_distance": target_distance,
            "collided": bool(not metadata.get("lastActionSuccess", True)),
            "grasped": any(obj.get("isPickedUp", False) for obj in metadata.get("objects", [])),
            "visible_objects": visible_objects[:20],
            "agent_position": agent_pos,
            "agent_rotation": agent_rot,
            "agent_horizon": agent_horizon,
            "last_action": metadata.get("lastAction", ""),
        }

    @staticmethod
    def _get_center_depth_mean(depth_frame: np.ndarray | None) -> float | None:
        if depth_frame is None:
            return None
        if depth_frame.ndim != 2:
            return None
        h, w = depth_frame.shape
        h0, h1 = int(h * 0.4), int(h * 0.6)
        w0, w1 = int(w * 0.4), int(w * 0.6)
        center = depth_frame[h0:h1, w0:w1]
        if center.size == 0:
            return None
        return float(np.nanmean(center))

    def reset(self, episode_id: int) -> StepResult:
        # GPU 均匀分配：按 episode_id 将 worker 轮转绑定到各 GPU，
        # 降低 96 workers 同时争抢同一 GPU 的概率，提升初始化速度。
        if self.cfg.gpu_device is not None:
            gpu_index = self.cfg.gpu_device
        else:
            gpu_index = episode_id % _available_gpu_count()
        self.controller.gpu_device = gpu_index
        event = self.controller.reset(scene=self.cfg.scene)
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
        if isinstance(action, dict):
            move = float(action.get("move_ahead_distance", 0.0))
            delta_yaw = float(action.get("delta_yaw", 0.0))
            delta_pitch = float(action.get("delta_pitch", 0.0))
            move_back = bool(action.get("move_back", False))
            event = self.controller.step(action="Pass")
            if abs(move) > 1e-6:
                if move_back:
                    event = self.controller.step(action="MoveBack", moveMagnitude=max(0.0, abs(move)))
                else:
                    event = self.controller.step(action="MoveAhead", moveMagnitude=max(0.0, move))
            if abs(delta_yaw) > 1e-6:
                action_name = "RotateRight" if delta_yaw > 0 else "RotateLeft"
                event = self.controller.step(action=action_name, degrees=abs(delta_yaw))
            if abs(delta_pitch) > 1e-6:
                horizon = float(event.metadata.get("agent", {}).get("cameraHorizon", 0.0))
                target_horizon = horizon + delta_pitch
                event = self.controller.step(
                    action="TeleportFull",
                    position=event.metadata.get("agent", {}).get("position", {}),
                    rotation=event.metadata.get("agent", {}).get("rotation", {}),
                    horizon=target_horizon,
                    standing=True,
                    forceAction=True,
                )
        else:
            event = self.controller.step(action=action)
        frame = np.asarray(event.frame, dtype=np.uint8)
        step_metadata = self._extract_metadata(event)
        center_depth = self._get_center_depth_mean(getattr(event, "depth_frame", None))
        if center_depth is not None:
            step_metadata["center_depth_m"] = center_depth
        return StepResult(frame=frame, metadata=step_metadata)

    def get_reachable_positions(self) -> list[dict[str, float]]:
        if self._reachable_cache is not None:
            return self._reachable_cache
        event = self.controller.step(action="GetReachablePositions")
        positions = event.metadata.get("actionReturn") or []
        self._reachable_cache = [p for p in positions if isinstance(p, dict)]
        return self._reachable_cache

    def close(self) -> None:
        if self.controller is not None:
            self.controller.stop()