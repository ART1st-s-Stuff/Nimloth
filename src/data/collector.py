"""Phase 1 数据采集主逻辑。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
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
    scenes: list[str]
    seed: int
    output_dir: str
    num_episodes_per_scene: int
    max_steps_per_episode: int
    num_workers: int
    image_height: int
    image_width: int
    visibility_distance: float
    grid_size: float
    render_depth_image: bool
    render_instance_segmentation: bool
    ai2thor_platform: str
    ai2thor_cache_dir: str


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
                    scene=cfg.scenes[0],
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
            raise RuntimeError(f"ai2thor 初始化失败: {exc}") from exc
    raise ValueError(f"不支持的采集后端: {cfg.backend}")


def _collect_scene(cfg: CollectConfig, scene: str, worker_id: int) -> tuple[Path, int]:
    out_dir = ensure_dir(cfg.output_dir)
    img_dir = ensure_dir(out_dir / "images")
    manifest = out_dir / f"manifest_worker_{worker_id}_{scene}.jsonl"
    if manifest.exists():
        manifest.unlink()

    scene_cfg = CollectConfig(
        backend=cfg.backend,
        scenes=[scene],
        seed=cfg.seed + worker_id,
        output_dir=cfg.output_dir,
        num_episodes_per_scene=cfg.num_episodes_per_scene,
        max_steps_per_episode=cfg.max_steps_per_episode,
        num_workers=1,
        image_height=cfg.image_height,
        image_width=cfg.image_width,
        visibility_distance=cfg.visibility_distance,
        grid_size=cfg.grid_size,
        render_depth_image=cfg.render_depth_image,
        render_instance_segmentation=cfg.render_instance_segmentation,
        ai2thor_platform=cfg.ai2thor_platform,
        ai2thor_cache_dir=cfg.ai2thor_cache_dir,
    )
    env = _build_env(scene_cfg)
    sample_count = 0
    try:
        scene_tag = scene.lower()
        for episode_id in range(cfg.num_episodes_per_scene):
            _ = env.reset(episode_id=episode_id)
            for step_id in range(cfg.max_steps_per_episode):
                action_id = step_id % len(ACTION_SPACE)
                action = ACTION_SPACE[action_id]
                step_result = env.step(action=action, step_id=step_id)
                metadata: dict[str, Any] = step_result.metadata
                metadata["scene"] = scene
                label_text = build_label_text(metadata)
                image_name = f"{scene_tag}_ep{episode_id:04d}_step{step_id:04d}.png"
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
                sample_count += 1
    finally:
        env.close()
    return manifest, sample_count


def run_collection(cfg: CollectConfig) -> Path:
    """执行多场景采集并输出合并 manifest。"""
    if not cfg.scenes:
        raise RuntimeError("未配置 scene 列表：data.env.scenes 不能为空。")
    out_dir = ensure_dir(cfg.output_dir)
    final_manifest = out_dir / "manifest.jsonl"
    if final_manifest.exists():
        final_manifest.unlink()

    if cfg.num_workers <= 1 or len(cfg.scenes) == 1:
        partials = [_collect_scene(cfg, scene, idx) for idx, scene in enumerate(cfg.scenes)]
    else:
        with ThreadPoolExecutor(max_workers=cfg.num_workers) as pool:
            futures = [
                pool.submit(_collect_scene, cfg, scene, idx) for idx, scene in enumerate(cfg.scenes)
            ]
            partials = [future.result() for future in futures]

    for partial_manifest, _ in partials:
        for line in partial_manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                append_jsonl(final_manifest, data=json.loads(line))
        partial_manifest.unlink(missing_ok=True)
    return final_manifest

