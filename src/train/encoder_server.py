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

import fcntl
import logging
import signal
import time
from pathlib import Path
from typing import Any, TextIO

import torch
import hydra
from omegaconf import DictConfig

from src.train.manifest_resolver import resolve_manifest_for_split
from src.train.latent_cache import (
    build_latent_cache_dir,
    build_episode_cache_path,
    build_episode_ready_path,
    list_completed_episodes,
)
from src.utils.env import load_project_env
from src.utils.seed import set_seed
from src.utils.terminal_ui import create_dashboard, LiveDashboard
from src.wm.encoders import build_wm_image_encoder

logger = logging.getLogger(__name__)


def _acquire_lock(lock_path: Path, *, timeout_sec: float = 30.0, poll_sec: float = 0.2) -> TextIO:
    """获取独占文件锁（非阻塞重试），返回文件描述符。"""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = open(lock_path, "w")
    start = time.time()
    while True:
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            waited = time.time() - start
            if waited >= timeout_sec:
                fd.close()
                raise TimeoutError(f"获取缓存锁超时: lock={lock_path}, waited={waited:.1f}s, timeout={timeout_sec:.1f}s")
            time.sleep(poll_sec)


def _release_lock(fd: TextIO) -> None:
    """释放文件锁。"""
    fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
    fd.close()


def _append_episode_latents(
    cache_dir: Path,
    episode_key: str,
    new_latents: dict[str, torch.Tensor],
    latent_dim: int,
) -> None:
    """将新 latents 追加到 episode 分块文件（先写临时文件再 rename）。"""
    episode_path = build_episode_cache_path(cache_dir, episode_key)
    lock_path = episode_path.with_suffix(".lock")
    lock_fd: TextIO | None = None
    try:
        try:
            lock_fd = _acquire_lock(lock_path, timeout_sec=30.0, poll_sec=0.2)
        except TimeoutError as exc:
            logger.warning("episode %s 缓存锁超时，回退为无锁写入: %s", episode_key, exc)

        if episode_path.exists():
            payload = torch.load(episode_path, map_location="cpu")
            existing: dict[str, torch.Tensor] = payload.get("latents", {}) if isinstance(payload, dict) else {}
        else:
            existing = {}
        existing.update(new_latents)
        tmp = episode_path.with_suffix(".tmp")
        torch.save({"latent_dim": latent_dim, "episode_key": episode_key, "latents": existing}, tmp)
        tmp.rename(episode_path)
    finally:
        if lock_fd is not None:
            _release_lock(lock_fd)


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
    lock_path = cache_path.with_suffix(".lock")
    lock_fd: TextIO | None = None
    try:
        try:
            lock_fd = _acquire_lock(lock_path, timeout_sec=30.0, poll_sec=0.2)
        except TimeoutError as exc:
            logger.warning("缓存锁超时，回退为无锁写入: %s", exc)

        if cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu")
            existing: dict[str, torch.Tensor] = payload.get("latents", {}) if isinstance(payload, dict) else {}
        else:
            existing = {}
        existing.update(new_latents)
        tmp = cache_path.with_suffix(".tmp")
        torch.save({"latent_dim": latent_dim, "latents": existing}, tmp)
        tmp.rename(cache_path)
    finally:
        if lock_fd is not None:
            _release_lock(lock_fd)


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
    try:
        state_file.write_text(
            json.dumps({
                "current_episode_key": current_episode_key,
                "total_episodes": total_episodes,
                "episode_progress": episode_progress,
                "episode_total": episode_total,
                "encoded_images": encoded_images,
                "total_images": total_images,
                "episodes_completed": episodes_completed,
                "is_first_episode_done": is_first_episode_done,
                "timestamp": time.time(),
            }, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass  # 非关键操作，失败不影响主流程


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

    # 待编码的 episode（排除已完成的）
    remaining_episodes = [ek for ek in episode_keys if ek not in completed]
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

    # 阶段 1：先编码第一个 episode（用于启动训练）
    if wait_first_episode and remaining_episodes:
        first_ep_key = remaining_episodes[0]
        first_ep_paths = episode_images[first_ep_key]
        logger.info("阶段 1：预编码 episode %s（%d 张图像）...", first_ep_key, len(first_ep_paths))

        for start in range(0, len(first_ep_paths), batch_size):
            batch = first_ep_paths[start:start + batch_size]
            heartbeat_path.write_text(str(time.time()), encoding="utf-8")
            with torch.no_grad(), torch.autocast("cuda"):
                outputs = encoder.encode_image_paths(batch)
            latents_out = {p: o.z.detach().cpu() for p, o in zip(batch, outputs, strict=True)}
            _append_episode_latents(cache_dir, first_ep_key, latents_out, latent_dim)

            # 更新仪表板和状态文件
            total_done = sum(len(episode_images[e]) for e in remaining_episodes[:remaining_episodes.index(first_ep_key)])
            progress_in_ep = (start + batch_size) / len(first_ep_paths)
            dashboard.update_encoder(
                current_episode=first_ep_key,
                total_episodes=total_episodes,
                episode_progress=start + batch_size,
                episode_total=len(first_ep_paths),
                encoded_images=total_done + start + batch_size,
                total_images=total_images,
                episodes_completed=len(completed),
                is_first_episode_done=False,
            )
            _write_encoder_state(
                cache_dir,
                current_episode_key=first_ep_key,
                total_episodes=total_episodes,
                episode_progress=start + batch_size,
                episode_total=len(first_ep_paths),
                encoded_images=total_done + start + batch_size,
                total_images=total_images,
                episodes_completed=len(completed),
                is_first_episode_done=False,
            )
            logger.info("  episode %s 进度: %d / %d", first_ep_key, min(start + batch_size, len(first_ep_paths)), len(first_ep_paths))

        _mark_episode_ready(cache_dir, first_ep_key)
        dashboard.update_encoder(
            current_episode=first_ep_key,
            total_episodes=total_episodes,
            episode_progress=len(first_ep_paths),
            episode_total=len(first_ep_paths),
            encoded_images=len(first_ep_paths),
            total_images=total_images,
            episodes_completed=len(completed) + 1,
            is_first_episode_done=True,
        )
        _write_encoder_state(
            cache_dir,
            current_episode_key=first_ep_key,
            total_episodes=total_episodes,
            episode_progress=len(first_ep_paths),
            episode_total=len(first_ep_paths),
            encoded_images=len(first_ep_paths),
            total_images=total_images,
            episodes_completed=len(completed) + 1,
            is_first_episode_done=True,
        )
        logger.info("阶段 1 完成：episode %s 就绪，训练可以开始。", first_ep_key)

        # 移除已完成的 episode，继续剩余编码
        remaining_episodes = remaining_episodes[1:]

    # 阶段 2：编码剩余 episode（与训练并行）
    # 使用轮转编码策略：每个 episode 轮流取一个 batch
    # 这样可以确保所有 episode 均匀推进，训练采样时可以覆盖更多 episode
    logger.info("阶段 2：并行编码剩余 %d 个 episode（轮转策略）...", len(remaining_episodes))

    completed_count = len(completed) + (1 if wait_first_episode and episode_keys else 0)

    # 构建轮转队列：每个 episode 剩余的图像路径
    episode_remaining: dict[str, list[str]] = {ek: list(episode_images[ek]) for ek in remaining_episodes}

    # 轮转编码直到所有 episode 完成
    while any(episode_remaining[ek] for ek in remaining_episodes):
        for ep_key in list(remaining_episodes):
            if not episode_remaining[ep_key]:
                continue  # 这个 episode 已经完成，跳过

            # 取一个 batch
            batch = episode_remaining[ep_key][:batch_size]
            episode_remaining[ep_key] = episode_remaining[ep_key][batch_size:]

            heartbeat_path.write_text(str(time.time()), encoding="utf-8")
            with torch.no_grad(), torch.autocast("cuda"):
                outputs = encoder.encode_image_paths(batch)
            latents_out = {p: o.z.detach().cpu() for p, o in zip(batch, outputs, strict=True)}
            _append_episode_latents(cache_dir, ep_key, latents_out, latent_dim)

            # 更新进度
            total_done = sum(len(episode_images[e]) for e in episode_keys) - sum(len(v) for v in episode_remaining.values())
            completed_so_far = len([e for e in remaining_episodes if not episode_remaining.get(e, [])])

            dashboard.update_encoder(
                current_episode=ep_key,
                total_episodes=total_episodes,
                episode_progress=len(episode_images[ep_key]) - len(episode_remaining[ep_key]),
                episode_total=len(episode_images[ep_key]),
                encoded_images=total_done,
                total_images=total_images,
                episodes_completed=completed_so_far,
                is_first_episode_done=True,
            )
            _write_encoder_state(
                cache_dir,
                current_episode_key=ep_key,
                total_episodes=total_episodes,
                episode_progress=len(episode_images[ep_key]) - len(episode_remaining[ep_key]),
                episode_total=len(episode_images[ep_key]),
                encoded_images=total_done,
                total_images=total_images,
                episodes_completed=completed_so_far,
                is_first_episode_done=True,
            )

            # 检查这个 episode 是否完成
            if not episode_remaining[ep_key]:
                _mark_episode_ready(cache_dir, ep_key)
                completed_count += 1
                logger.info("Episode %s 完成（%d 张图像）", ep_key, len(episode_images[ep_key]))
                del episode_remaining[ep_key]

    # 停止仪表板
    dashboard.stop()

    done_path.write_text(str(len(episode_keys)))
    logger.info("Encoder server 完成：所有 %d 个 episode 已编码。", len(episode_keys))


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