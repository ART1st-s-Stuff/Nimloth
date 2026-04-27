"""Encoder server：独立进程，持续从请求队列读取缺失图像路径，批量编码后写入共享 cache 文件。

与训练进程通过 multiprocessing.Manager.Queue() 和 .pt cache 文件通信。
启动方式（示例）：
    CUDA_VISIBLE_DEVICES=0 python -m src.train.encoder_server \
        wm=cfm_dinov2m \
        dataset.manifests.train=/path/to/run_dir \
        pipeline.train.encoder_batch_size=64

分块模式：
    分块 latent cache 目录结构：
        {run_dir}/{stem}.latents.{wm_name}/
            episode_{scene}_{id}.pt      # 每个 episode 独立文件，使用 scene_episode_id 作为键
            episode_{scene}_{id}.ready   # episode 就绪 marker
            ...
"""

from __future__ import annotations

import json
from collections import defaultdict
import os

import logging
import signal
import time
from pathlib import Path
from typing import Any

import torch
import hydra
from omegaconf import DictConfig

from src.train.manifest_resolver import resolve_manifest_for_split
from src.infrastructure.encoding.cache_protocol import (
    append_episode_latents as append_episode_latents_file,
    append_single_file_latents,
    write_json_state,
)
from src.train.latent_cache import (
    build_latent_cache_dir,
    build_episode_cache_path,
    build_episode_ready_path,
    list_completed_episodes,
)
from src.train.encoder_control_server import EncoderControlServer
from src.utils.env import load_project_env
from src.utils.seed import set_seed
from src.utils.terminal_ui import create_dashboard, LiveDashboard
from src.wm.encoder import build_wm_image_encoder

logger = logging.getLogger(__name__)


def _append_episode_latents(
    cache_dir: Path,
    episode_key: str,
    new_latents: dict[str, torch.Tensor],
    latent_dim: int,
) -> None:
    """将新 latents 追加到 episode 分块文件（先写临时文件再 rename）。"""
    episode_path = build_episode_cache_path(cache_dir, episode_key)
    append_episode_latents_file(
        episode_path=episode_path,
        episode_key=episode_key,
        new_latents=new_latents,
        latent_dim=latent_dim,
    )


def _mark_episode_ready(cache_dir: Path, episode_key: str) -> None:
    """写入 episode 就绪 marker。"""
    ready_path = build_episode_ready_path(cache_dir, episode_key)
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    ready_path.write_text(str(time.time()), encoding="utf-8")
    logger.info("Episode %s 完成，就绪 marker 已写入", episode_key)


def _parse_manifest_episodes(manifest_path: Path) -> dict[str, list[str]]:
    """解析 manifest，按 episode 分组图像路径。

    使用 scene + episode_id 作为唯一键，确保跨 scene 的 episode 不冲突。
    支持两种 manifest 格式：
    1. 单个 manifest.jsonl 文件
    2. 包含多个 manifest_worker_*.jsonl 文件的 run 目录

    Returns:
        dict[episode_key -> list of image_paths], episode_key 格式为 "{scene}_{episode_id}"
    """
    episode_images: dict[str, list[str]] = defaultdict(list)
    manifest_files = list(manifest_path.glob("manifest_worker_*.jsonl")) if manifest_path.is_dir() else []
    if manifest_files:
        # 目录模式：读取所有 worker manifest
        for wf in sorted(manifest_files):
            for line in wf.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    sample = json.loads(line)
                    image_path = sample.get("image_path")
                    if image_path:
                        metadata = sample.get("metadata", {})
                        scene = metadata.get("scene", "unknown")
                        episode_id = int(sample.get("episode_id", 0))
                        episode_key = f"{scene}_{episode_id}"
                        episode_images[episode_key].append(str(image_path))
                except (json.JSONDecodeError, ValueError):
                    continue
    elif manifest_path.is_file():
        # 单文件模式（向后兼容）
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
                image_path = sample.get("image_path")
                if image_path:
                    metadata = sample.get("metadata", {})
                    scene = metadata.get("scene", "unknown")
                    episode_id = int(sample.get("episode_id", 0))
                    episode_key = f"{scene}_{episode_id}"
                    episode_images[episode_key].append(str(image_path))
            except (json.JSONDecodeError, ValueError):
                continue
    return episode_images


def _append_latents_to_cache(cache_path: Path, new_latents: dict[str, torch.Tensor], latent_dim: int) -> None:
    """将新 latents 原子追加到 cache 文件（先写临时文件再 rename）- 单文件模式。"""
    append_single_file_latents(cache_path, new_latents, latent_dim)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    load_project_env()
    set_seed(int(cfg.project.seed))
    train_cfg = cfg.pipeline.train
    dataset_cfg = cfg.dataset
    wm_cfg = cfg.wm

    # 分块模式开关（可通过配置覆盖）
    chunk_mode = bool(train_cfg.get("lazy_episode_chunk", True))
    wait_first_episode = bool(train_cfg.get("lazy_wait_first_episode", True))

    run_dir = resolve_manifest_for_split(
        manifests_cfg=dataset_cfg.get("manifests", {}),
        split="train",
        outputs_root=str(train_cfg.operation.outputs_root),
        dataset_name=str(dataset_cfg.name),
    )
    if not run_dir.exists():
        raise FileNotFoundError(f"run directory not found: {run_dir}")

    latent_dim = int(wm_cfg.latent_dim)

    if chunk_mode:
        _run_chunk_mode(run_dir, wm_cfg, train_cfg, latent_dim, wait_first_episode)
    else:
        _run_single_file_mode(run_dir, wm_cfg, train_cfg, latent_dim)


def _write_encoder_state(
    cache_dir: Path,
    current_episode_key: str,
    total_episodes: int,
    episode_progress: int,
    episode_total: int,
    encoded_images: int,
    total_images: int,
    episodes_completed: int,
    is_first_episode_done: bool,
) -> None:
    """写入 encoder 状态到共享文件。"""
    state_file = cache_dir / "encoder_state.json"
    write_json_state(
        state_file,
        {
            "current_episode_key": current_episode_key,
            "total_episodes": total_episodes,
            "episode_progress": episode_progress,
            "episode_total": episode_total,
            "encoded_images": encoded_images,
            "total_images": total_images,
            "episodes_completed": episodes_completed,
            "is_first_episode_done": is_first_episode_done,
            "timestamp": time.time(),
        },
    )


def _run_chunk_mode(
    run_dir: Path,
    wm_cfg: DictConfig,
    train_cfg: DictConfig,
    latent_dim: int,
    wait_first_episode: bool,
) -> None:
    """分块模式：按 episode 分组编码，支持并行训练。"""
    cache_dir = build_latent_cache_dir(run_dir, str(wm_cfg.name))
    cache_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_path = cache_dir / "encoder_heartbeat"
    done_path = cache_dir / "done"
    control_socket_path = os.environ.get("ENCODER_CONTROL_SOCKET", "").strip()
    control_server = EncoderControlServer(control_socket_path)

    # 尽早启动控制面，避免主线程在大模型加载期间等待超时。
    control_server.start()

    device = torch.device("cuda")
    encoder = build_wm_image_encoder(wm_cfg=wm_cfg)
    if encoder is None:
        raise RuntimeError("encoder 未构建成功，确认 wm 配置中启用了 image_encoder。")
    encoder.to(device)
    encoder.eval()

    batch_size = int(train_cfg.encoder_batch_size)

    # 解析 manifest 并按 episode 分组（使用 scene_episode_id 作为键）
    episode_images = _parse_manifest_episodes(run_dir)
    episode_keys = sorted(episode_images.keys())
    total_images = sum(len(paths) for paths in episode_images.values())
    total_episodes = len(episode_keys)

    # 检查已完成的 episode
    completed = list_completed_episodes(cache_dir)
    encoded_images: set[str] = set()
    for episode_key in completed:
        ep_path = build_episode_cache_path(cache_dir, episode_key)
        if not ep_path.exists():
            continue
        try:
            payload = torch.load(ep_path, map_location="cpu")
            latents = payload.get("latents", {}) if isinstance(payload, dict) else {}
            if isinstance(latents, dict):
                encoded_images.update(str(k) for k in latents.keys())
        except Exception:
            continue

    # 待编码的 episode（排除已完成的）
    remaining_episodes = [ek for ek in episode_keys if ek not in completed]
    image_to_episode: dict[str, str] = {}
    for ep_key, paths in episode_images.items():
        for image_path in paths:
            image_to_episode[image_path] = ep_key
    control_server.mark_encoded(list(encoded_images))
    logger.info(
        "Encoder server 启动（分块模式）: run_dir=%s, cache_dir=%s, latent_dim=%d, batch_size=%d",
        run_dir,
        cache_dir,
        latent_dim,
        batch_size,
    )
    logger.info("  总 episode: %d, 已完成: %d, 待编码: %d", total_episodes, len(completed), len(remaining_episodes))

    # 启动实时仪表板
    dashboard = create_dashboard(title="Encoder Server", show_gpu=True, refresh_rate=0.5)
    dashboard.start()

    try:
        # 构建轮转队列：每个 episode 剩余的图像路径
        episode_remaining: dict[str, list[str]] = {ek: list(episode_images[ek]) for ek in remaining_episodes}
        episode_remaining_set: dict[str, set[str]] = {ek: set(episode_images[ek]) for ek in remaining_episodes}
        encoded_count = len(encoded_images)
        first_episode_done = len(completed) > 0 or not wait_first_episode
        if wait_first_episode and remaining_episodes:
            logger.info("阶段 1：优先保证首个 episode 就绪，再并行编码其余 episode。")
        logger.info("阶段 2：并行编码剩余 %d 个 episode（轮转 + priority queue）...", len(remaining_episodes))

        while any(episode_remaining.get(ek) for ek in remaining_episodes):
            if control_server.should_shutdown():
                logger.info("收到控制面 shutdown 指令，准备退出 encoder server")
                break

            priority_batch = control_server.pop_priority_batch(batch_size)
            selected: list[tuple[str, str]] = []
            if priority_batch:
                for image_path in priority_batch:
                    ep_key = image_to_episode.get(image_path, "")
                    if not ep_key:
                        continue
                    rem = episode_remaining_set.get(ep_key)
                    if rem is None or image_path not in rem:
                        continue
                    selected.append((ep_key, image_path))
                if selected:
                    logger.info("处理优先编码队列: batch=%d", len(selected))

            if not selected:
                for ep_key in list(remaining_episodes):
                    rem_list = episode_remaining.get(ep_key, [])
                    if not rem_list:
                        continue
                    for image_path in rem_list[:batch_size]:
                        selected.append((ep_key, image_path))
                    break

            if not selected:
                time.sleep(0.1)
                continue

            batch = [x[1] for x in selected]
            heartbeat_path.write_text(str(time.time()), encoding="utf-8")
            with torch.no_grad(), torch.autocast("cuda"):
                outputs = encoder.encode_image_paths(batch)

            per_episode_latents: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
            for (ep_key, image_path), out in zip(selected, outputs, strict=True):
                per_episode_latents[ep_key][image_path] = out.z.detach().cpu()
                rem_set = episode_remaining_set.get(ep_key)
                if rem_set is not None:
                    rem_set.discard(image_path)
                rem_list = episode_remaining.get(ep_key)
                if rem_list is not None and image_path in rem_list:
                    rem_list.remove(image_path)

            for ep_key, latents_out in per_episode_latents.items():
                _append_episode_latents(cache_dir, ep_key, latents_out, latent_dim)
                if wait_first_episode and not first_episode_done and remaining_episodes and ep_key == remaining_episodes[0]:
                    if not episode_remaining.get(ep_key):
                        _mark_episode_ready(cache_dir, ep_key)
                        first_episode_done = True
                        logger.info("阶段 1 完成：episode %s 就绪，训练可以开始。", ep_key)
                elif not episode_remaining.get(ep_key):
                    _mark_episode_ready(cache_dir, ep_key)
                    logger.info("Episode %s 完成（%d 张图像）", ep_key, len(episode_images[ep_key]))

            control_server.mark_encoded(batch)
            encoded_count += len(batch)
            completed_so_far = len([ek for ek in remaining_episodes if not episode_remaining.get(ek)])
            current_episode_key = selected[0][0]
            ep_done = len(episode_images[current_episode_key]) - len(episode_remaining.get(current_episode_key, []))

            dashboard.update_encoder(
                current_episode=current_episode_key,
                total_episodes=total_episodes,
                episode_progress=ep_done,
                episode_total=len(episode_images[current_episode_key]),
                encoded_images=encoded_count,
                total_images=total_images,
                episodes_completed=completed_so_far,
                is_first_episode_done=first_episode_done,
            )
            _write_encoder_state(
                cache_dir,
                current_episode_key=current_episode_key,
                total_episodes=total_episodes,
                episode_progress=ep_done,
                episode_total=len(episode_images[current_episode_key]),
                encoded_images=encoded_count,
                total_images=total_images,
                episodes_completed=completed_so_far,
                is_first_episode_done=first_episode_done,
            )

        # 全量缓存命中时，优先队列中的任务不会进入编码循环，需主动清空避免训练侧等待超时。
        if not remaining_episodes:
            dropped = control_server.clear_priority_queue()
            if dropped > 0:
                logger.info("待编码 episode 为 0，已清空 %d 条优先编码任务。", dropped)

        done_path.write_text(str(len(episode_keys)))
        logger.info("Encoder server 完成：已编码 %d/%d 张图像。", encoded_count, total_images)
    finally:
        control_server.stop()
        dashboard.stop()


def _run_single_file_mode(
    run_dir: Path,
    wm_cfg: DictConfig,
    train_cfg: DictConfig,
    latent_dim: int,
) -> None:
    """单文件模式（向后兼容）：所有 latent 编码到单一 cache 文件。"""
    from src.train.latent_cache import build_latent_cache_path

    cache_path = build_latent_cache_path(run_dir, str(wm_cfg.name))
    heartbeat_path = cache_path.with_suffix(".encoder_heartbeat")
    done_path = cache_path.with_suffix(".done")

    device = torch.device("cuda")
    encoder = build_wm_image_encoder(wm_cfg=wm_cfg)
    if encoder is None:
        raise RuntimeError("encoder 未构建成功，确认 wm 配置中启用了 image_encoder。")
    encoder.to(device)
    encoder.eval()

    batch_size = int(train_cfg.encoder_batch_size)

    # 加载已有 cache
    existing_paths: set[str] = set()
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        latents = payload.get("latents", {}) if isinstance(payload, dict) else {}
        if isinstance(latents, dict):
            existing_paths = set(latents.keys())

    # 收集所有图像路径（支持 run_dir 和单个 manifest 文件）
    manifest_paths: set[str] = set()
    if run_dir.is_dir():
        # 目录模式：读取所有 worker manifest
        for wf in sorted(run_dir.glob("manifest_worker_*.jsonl")):
            for line in wf.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        sample = json.loads(line)
                        if "image_path" in sample:
                            manifest_paths.add(str(sample["image_path"]))
                    except json.JSONDecodeError:
                        continue
    elif run_dir.is_file():
        # 单文件模式（向后兼容）
        for line in run_dir.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    sample = json.loads(line)
                    if "image_path" in sample:
                        manifest_paths.add(str(sample["image_path"]))
                except json.JSONDecodeError:
                    continue

    precompute_paths = sorted(manifest_paths - existing_paths)
    logger.info(
        "Encoder server 启动（单文件模式）: run_dir=%s, cache=%s, latent_dim=%d, batch_size=%d, 待编码=%d, 已命中=%d",
        run_dir,
        cache_path,
        latent_dim,
        batch_size,
        len(precompute_paths),
        len(existing_paths),
    )

    pending_batch: list[str] = []
    done_count = 0

    if precompute_paths:
        pending_batch = precompute_paths[:batch_size]
        precompute_idx = batch_size
        precompute_exhausted = False
    else:
        pending_batch = []
        precompute_idx = 0
        precompute_exhausted = True

    def flush_batch(batch: list[str]) -> None:
        nonlocal done_count
        if not batch:
            return
        heartbeat_path.write_text(str(time.time()), encoding="utf-8")
        with torch.no_grad(), torch.autocast("cuda"):
            outputs = encoder.encode_image_paths(batch)
        latents_out = {p: o.z.detach().cpu() for p, o in zip(batch, outputs, strict=True)}
        _append_latents_to_cache(cache_path, latents_out, latent_dim)
        done_count += len(batch)
        logger.info("encoder server 进度: %d / %d", done_count, len(precompute_paths))

    def signal_handler(signum: int, frame: Any) -> None:
        logger.info("收到信号 %d，正在 flush 剩余 batch 后退出...", signum)
        if pending_batch:
            flush_batch(pending_batch)
        done_path.write_text(str(done_count))
        raise SystemExit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    running = True
    while running and not precompute_exhausted:
        heartbeat_path.write_text(str(time.time()), encoding="utf-8")
        try:
            if len(pending_batch) < batch_size and precompute_idx < len(precompute_paths):
                pending_batch.append(precompute_paths[precompute_idx])
                precompute_idx += 1

            if len(pending_batch) >= batch_size:
                flush_batch(pending_batch[:batch_size])
                pending_batch = pending_batch[batch_size:]
            else:
                time.sleep(0.1)
        except SystemExit:
            running = False
        except Exception as exc:
            logger.error("encoder server 异常: %s", exc, exc_info=True)
            time.sleep(1)

    if pending_batch:
        flush_batch(pending_batch)
    done_path.write_text(str(done_count))
    logger.info("Encoder server 退出，完成 %d 个 latent 编码。", done_count)


if __name__ == "__main__":
    main()