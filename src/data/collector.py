"""Phase 1 数据采集主逻辑。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from src.data.ai2thor_env import Ai2ThorEnvAdapter, Ai2ThorEnvConfig
from src.data.env_adapter import EnvAdapter
from src.data.labeler import build_label_text
from src.data.mock_env import MockEnvAdapter, MockEnvConfig
from src.data.schema import FrameState
from src.utils.io import append_jsonl, ensure_dir


@dataclass
class CollectConfig:
    backend: str
    scene: str
    seed: int
    output_dir: str
    num_episodes: int
    max_steps_per_episode: int
    image_height: int
    image_width: int
    visibility_distance: float
    grid_size: float
    render_depth_image: bool
    render_instance_segmentation: bool
    ai2thor_platform: str
    ai2thor_cache_dir: str
    fallback_to_mock_on_error: bool


ACTION_SPACE = ["MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown", "Done"]


def _build_env(cfg: CollectConfig) -> EnvAdapter:
    if cfg.backend == "mock":
        return MockEnvAdapter(
            MockEnvConfig(
                image_height=cfg.image_height,
                image_width=cfg.image_width,
                seed=cfg.seed,
            )
        )
    if cfg.backend == "ai2thor":
        try:
            return Ai2ThorEnvAdapter(
                Ai2ThorEnvConfig(
                    scene=cfg.scene,
                    image_height=cfg.image_height,
                    image_width=cfg.image_width,
                    visibility_distance=cfg.visibility_distance,
                    grid_size=cfg.grid_size,
                    render_depth_image=cfg.render_depth_image,
                    render_instance_segmentation=cfg.render_instance_segmentation,
                    platform=cfg.ai2thor_platform,
                    cache_dir=cfg.ai2thor_cache_dir,
                )
            )
        except Exception as exc:
            if cfg.fallback_to_mock_on_error:
                print(f"[collect] ai2thor 初始化失败，回退 mock: {exc}")
                return MockEnvAdapter(
                    MockEnvConfig(
                        image_height=cfg.image_height,
                        image_width=cfg.image_width,
                        seed=cfg.seed,
                    )
                )
            raise RuntimeError(f"ai2thor 初始化失败: {exc}") from exc
    raise ValueError(f"不支持的采集后端: {cfg.backend}")


def run_collection(cfg: CollectConfig) -> Path:
    """执行最小数据采集并输出 manifest。"""
    out_dir = ensure_dir(cfg.output_dir)
    img_dir = ensure_dir(out_dir / "images")
    manifest = out_dir / "manifest.jsonl"
    if manifest.exists():
        manifest.unlink()

    env = _build_env(cfg)
    try:
        for episode_id in range(cfg.num_episodes):
            _ = env.reset(episode_id=episode_id)
            for step_id in range(cfg.max_steps_per_episode):
                action_id = step_id % len(ACTION_SPACE)
                action = ACTION_SPACE[action_id]
                step_result = env.step(action=action, step_id=step_id)
                metadata: dict[str, Any] = step_result.metadata
                label_text = build_label_text(metadata)
                image_name = f"ep{episode_id:04d}_step{step_id:04d}.png"
                image_path = img_dir / image_name
                Image.fromarray(step_result.frame).save(image_path)
                sample = FrameState(
                    episode_id=episode_id,
                    step_id=step_id,
                    image_path=str(image_path),
                    action=action,
                    action_id=action_id,
                    metadata=metadata,
                    label_text=label_text,
                )
                append_jsonl(manifest, sample.to_dict())
    finally:
        env.close()
    return manifest

