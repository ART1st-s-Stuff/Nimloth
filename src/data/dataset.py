"""训练所需数据集定义。"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset
from src.wm.encoder import WMImageEncoder

logger = logging.getLogger(__name__)


def _read_jsonl_lines(path: Path) -> list[dict[str, Any]]:
    """安全读取 JSONL，忽略不完整的尾行（采集进行中时可能出现）。"""
    lines = []
    if not path.exists():
        return lines
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            lines.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("跳过不完整的 manifest 行（采集进行中属正常）: %s", line[:80])
    return lines


def read_worker_manifests(run_dir: Path) -> list[dict[str, Any]]:
    """读取 run_dir 下所有 manifest_worker_*.jsonl 文件，返回合并后的样本列表。

    与旧版合并脚本不同，这里直接读取原始 worker manifest 文件，
    不做 scene 级别的 episode 合并。
    每个 worker manifest 文件格式为 manifest_worker_{worker_id}_{scene}.jsonl
    """
    import re

    worker_files: list[Path] = []
    for p in run_dir.iterdir():
        if p.is_file() and p.suffix == ".jsonl" and re.match(r"^manifest_worker_\d+_.+\.jsonl$", p.name):
            worker_files.append(p)

    if not worker_files:
        logger.warning("run 目录中未找到 worker manifest 文件: %s", run_dir)
        return []

    all_samples: list[dict[str, Any]] = []
    for worker_path in sorted(worker_files):
        samples = _read_jsonl_lines(worker_path)
        all_samples.extend(samples)

    logger.info("从 %d 个 worker manifest 文件读取了 %d 条样本", len(worker_files), len(all_samples))
    return all_samples


def resolve_run_dir(base_path: str) -> Path | None:
    """解析 run 目录路径。

    支持以下输入格式：
    - 指向 run 目录的路径（如 datasets/ai2thor/train/2026-04-24_14-47-16）
    - 指向 split 目录的路径（如 datasets/ai2thor/train）
      此时从 metadata.json 的 latest 字段获取最新 run
    """
    base = Path(base_path)
    if not base.exists():
        return None

    if base.is_file():
        return None

    # 检查是否已经是 run 目录（包含 manifest_worker_*.jsonl）
    if any(p.match("manifest_worker_*.jsonl") for p in base.iterdir()):
        return base

    # 查找 latest run
    metadata_path = base / "metadata.json"
    if metadata_path.exists():
        try:
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            latest = meta.get("latest")
            if isinstance(latest, str):
                latest_dir = base / latest
                if latest_dir.is_dir():
                    return latest_dir
        except Exception:
            pass

    # 退化为按目录时间戳取最新
    candidates = [p for p in base.iterdir() if p.is_dir() and p.name not in ("images", "val", "test")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def build_env_context(metadata: dict[str, Any]) -> str:
    """基于 metadata 生成任务无关环境信息描述。"""
    scene = metadata.get("scene")
    distance = metadata.get("target_distance")
    collided = metadata.get("collided")
    grasped = metadata.get("grasped")
    parts: list[str] = []
    if scene is not None:
        parts.append(f"scene={scene}")
    if isinstance(distance, (int, float)):
        parts.append(f"target_distance={float(distance):.3f}m")
    if isinstance(collided, bool):
        parts.append(f"collided={'yes' if collided else 'no'}")
    if isinstance(grasped, bool):
        parts.append(f"grasped={'yes' if grasped else 'no'}")
    if not parts:
        return "env=unknown"
    return " | ".join(parts)


class WMDataset(Dataset):
    """从 manifest 构造序列训练样本。"""

    def __init__(
        self,
        manifest_path: str,
        latent_dim: int,
        action_dim: int,
        history_len: int,
        temporal_stride: int | tuple[int, int] = 1,
        image_encoder: WMImageEncoder | None = None,
        latent_cache_path: str | None = None,
        encoder_num_workers: int = 1,
        encoder_batch_size: int = 32,
        expected_num_patches: int = 0,
        expected_token_dim: int = 0,
        on_latent_progress: Callable[[int, int], None] | None = None,
        lazy_mode: bool = False,
        encoder_queue: Any = None,
    ) -> None:
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.history_len = max(1, history_len)
        self.temporal_stride_min, self.temporal_stride_max = self._normalize_temporal_stride(temporal_stride)
        self.temporal_stride_steps = self._sample_stride(max_valid_stride=self.temporal_stride_max)
        self.image_encoder = image_encoder
        self.latent_cache_path = Path(latent_cache_path) if latent_cache_path else None
        self.encoder_num_workers = max(1, int(encoder_num_workers))
        self.encoder_batch_size = max(1, int(encoder_batch_size))
        self.expected_num_patches = max(0, int(expected_num_patches))
        self.expected_token_dim = max(0, int(expected_token_dim))
        self.on_latent_progress = on_latent_progress
        self.lazy_mode = lazy_mode
        self.encoder_queue = encoder_queue
        # 分块模式相关（初始为 False，会在 _init_lazy_mode 中覆盖）
        self._chunk_mode = False
        self._cache_dir: Path | None = None
        # 如果 latent_cache_path 是目录，在这里提前设置（会被 _init_lazy_mode 再次确认）
        if self.latent_cache_path and self.latent_cache_path.is_dir():
            self._chunk_mode = True
            self._cache_dir = self.latent_cache_path
        self._sample_episode_map: list[int] = []  # 样本索引 -> episode_id
        self._episode_latent_cache: dict[int, dict[str, torch.Tensor]] = {}  # episode_id -> latents
        self._episode_ready: dict[int, bool] = {}  # episode_id -> 是否就绪
        # 原始 lazy mode 字段（兼容单文件模式）
        self._pending_request_file: Path | None = None
        self._encoder_heartbeat_file: Path | None = None
        self._latent_cache: dict[str, torch.Tensor] = {}
        self._pending_latents: set[str] = set()
        self._cache_lock = threading.Lock()
        self._cache_mtime_sec: float | None = None
        # 自动检测 manifest_path 是文件还是目录
        manifest_path_obj = Path(manifest_path)
        if manifest_path_obj.is_file():
            # 单文件模式（manifest.jsonl）
            self.samples = _read_jsonl_lines(manifest_path_obj)
            self._run_dir = manifest_path_obj.parent
        elif manifest_path_obj.is_dir():
            # 目录模式：读取所有 manifest_worker_*.jsonl 文件
            self.samples = read_worker_manifests(manifest_path_obj)
            self._run_dir = manifest_path_obj
        else:
            # 尝试解析为 split 目录，自动找 latest run
            resolved = resolve_run_dir(manifest_path)
            if resolved is not None and resolved.is_dir():
                self.samples = read_worker_manifests(resolved)
                self._run_dir = resolved
            else:
                self.samples = []
                self._run_dir = manifest_path_obj
                logger.warning("manifest_path 不存在或无效: %s", manifest_path)
        self._training_indices: list[dict[str, list[int]]] = []
        # 使用 scene + episode_id 作为唯一标识（不同 scene 的 episode_id 可能重复）
        episode_to_indices: dict[str, list[int]] = {}
        for idx, sample in enumerate(self.samples):
            metadata = sample.get("metadata", {})
            scene = metadata.get("scene", "unknown")
            episode_id = int(sample.get("episode_id", -1))
            episode_key = f"{scene}_{episode_id}"
            # 存储完整的 episode_key（包含 scene），用于区分不同 scene 的相同 episode_id
            self._sample_episode_map.append(episode_key)
            episode_to_indices.setdefault(episode_key, []).append(idx)
        for episode_indices in episode_to_indices.values():
            min_history_last = self.history_len - 1
            max_history_last = len(episode_indices) - 2
            for history_last_local_idx in range(min_history_last, max_history_last + 1):
                max_valid_steps = len(episode_indices) - 1 - history_last_local_idx
                if max_valid_steps < self.temporal_stride_steps:
                    continue
                history_start_local_idx = history_last_local_idx - (self.history_len - 1)
                history_global_indices = episode_indices[history_start_local_idx : history_last_local_idx + 1]
                future_global_indices = [
                    episode_indices[history_last_local_idx + step]
                    for step in range(1, self.temporal_stride_steps + 1)
                ]
                action_source_indices = [
                    episode_indices[history_last_local_idx + step - 1]
                    for step in range(1, self.temporal_stride_steps + 1)
                ]
                self._training_indices.append(
                    {
                        "history_indices": history_global_indices,
                        "future_indices": future_global_indices,
                        "action_source_indices": action_source_indices,
                        "temporal_stride": [self.temporal_stride_steps],
                    }
                )
        if lazy_mode:
            self._init_lazy_mode()
        else:
            self._warmup_latent_cache()

    @staticmethod
    def _normalize_temporal_stride(temporal_stride: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(temporal_stride, tuple):
            if len(temporal_stride) != 2:
                raise ValueError("temporal_stride 区间配置必须是二元组 (min, max)。")
            stride_min = int(temporal_stride[0])
            stride_max = int(temporal_stride[1])
        else:
            stride_min = int(temporal_stride)
            stride_max = int(temporal_stride)
        stride_min = max(1, stride_min)
        stride_max = max(stride_min, stride_max)
        return stride_min, stride_max

    def _sample_stride(self, *, max_valid_stride: int) -> int:
        hi = min(self.temporal_stride_max, int(max_valid_stride))
        lo = min(self.temporal_stride_min, hi)
        if lo >= hi:
            return lo
        return int(torch.randint(low=lo, high=hi + 1, size=(1,)).item())

    def __len__(self) -> int:
        return len(self._training_indices)

    def _get_ready_episode_indices(self) -> list[int]:
        """返回已就绪 episode 的训练样本索引。"""
        if not self._episode_ready:
            return []
        ready_episodes = set(self._episode_ready.keys())
        return [i for i, idx_data in enumerate(self._training_indices)
                if idx_data["history_indices"] and
                self._sample_episode_map[idx_data["history_indices"][0]] in ready_episodes]

    def _init_lazy_mode(self) -> None:
        """Lazy 模式初始化：检测分块模式并加载已有 cache，不阻塞等待 encoder server。"""
        from src.train.latent_cache import build_episode_cache_path, build_latent_cache_dir, list_completed_episodes

        cache_path = self.latent_cache_path
        if cache_path is None:
            return

        cache_path_obj = Path(cache_path)

        # 如果 cache_path 是目录，直接使用分块模式
        if cache_path_obj.is_dir():
            self._chunk_mode = True
            self._cache_dir = cache_path_obj
            self._reload_chunked_cache()
            # 启动分块轮询线程
            self._cache_poll_thread = threading.Thread(target=self._poll_chunked_cache, daemon=True)
            self._cache_poll_thread.start()
            # 心跳文件在分块模式下位于 cache_dir
            self._encoder_heartbeat_file = cache_path_obj / "encoder_heartbeat"
            self._pending_request_file = None  # 分块模式不需要请求文件
            logger.info(
                "LazyWMDataset 初始化完成（分块模式）: cache_dir=%s, 已就绪=%d episode",
                self._cache_dir,
                len([e for e in self._episode_ready.values() if e]),
            )
        else:
            # cache_path 是文件 - 检测是否为分块目录（从文件名推断）
            cache_stem = cache_path_obj.stem
            # 从文件名解析 wm_name: e.g., "train.latents.cfm_dinov2m.pt" -> "cfm_dinov2m"
            wm_name = ".".join(cache_stem.split(".")[2:]) if cache_stem.count(".") >= 2 else cache_stem
            cache_dir = cache_path_obj.parent / f"{'.'.join(cache_stem.split('.')[:2])}.latents.{wm_name}"

            # 检查分块模式：目录存在且包含 episode 文件
            if cache_dir.exists() and list(cache_dir.glob("episode_*.pt")):
                self._chunk_mode = True
                self._cache_dir = cache_dir
                self._reload_chunked_cache()
                # 启动分块轮询线程
                self._cache_poll_thread = threading.Thread(target=self._poll_chunked_cache, daemon=True)
                self._cache_poll_thread.start()
                # 心跳文件在分块模式下位于 cache_dir
                self._encoder_heartbeat_file = cache_dir / "encoder_heartbeat"
                self._pending_request_file = None  # 分块模式不需要请求文件
                logger.info(
                    "LazyWMDataset 初始化完成（分块模式）: cache_dir=%s, 已就绪=%d episode",
                    self._cache_dir,
                    len([e for e in self._episode_ready.values() if e]),
                )
            else:
                # 单文件模式（向后兼容）
                self._chunk_mode = False
                self._cache_dir = None
                if cache_path_obj.exists():
                    self._reload_cache_from_file(cache_path_obj)
                # 启动单文件轮询线程
                self._cache_poll_thread = threading.Thread(target=self._cache_poll_loop, daemon=True)
                self._cache_poll_thread.start()
                # 设置文件请求路径（无 Manager.Queue 时的降级）
                if self.latent_cache_path:
                    self._pending_request_file = cache_path_obj.with_suffix(".request_queue")
                    self._encoder_heartbeat_file = cache_path_obj.with_suffix(".encoder_heartbeat")
                logger.info(
                    "LazyWMDataset 初始化完成（单文件模式）: cache=%s, 已命中=%d",
                    cache_path,
                    len(self._latent_cache),
                )

    def _reload_chunked_cache(self) -> None:
        """从分块 cache 目录加载已完成 episode 的 latents。

        支持两种文件命名格式：
        1. 整数格式：episode_{id:04d}.pt（如 episode_0000.pt）
        2. 场景格式：episode_{scene}_{id}.pt（如 episode_FloorPlan1_0.pt）

        .ready marker 文件始终使用场景格式（由 encoder 写入）。
        """
        from src.train.latent_cache import build_episode_ready_path, list_completed_episodes

        if self._cache_dir is None:
            return

        completed = list_completed_episodes(self._cache_dir)
        for episode_key in completed:
            if episode_key in self._episode_ready and self._episode_ready[episode_key]:
                continue  # 已加载

            # episode_key 格式："{scene}_{episode_id}" 或纯整数（如 "FloorPlan1_0" 或 "42"）
            # 尝试解析为整数 episode_id
            try:
                _, ep_id_str = episode_key.rsplit("_", 1)
                ep_id = int(ep_id_str)
            except (ValueError, IndexError):
                # 如果不是 scene_id 格式，可能是纯整数
                try:
                    ep_id = int(episode_key)
                except ValueError:
                    ep_id = -1

            # 尝试多种可能的文件名格式
            possible_paths = []
            if ep_id >= 0:
                # 整数格式：episode_0000.pt
                possible_paths.append(self._cache_dir / f"episode_{ep_id:04d}.pt")
            # episode_key 格式（保留原始格式）
            safe_key = episode_key.replace(" ", "_").replace("/", "_").replace("\\", "_")
            possible_paths.append(self._cache_dir / f"episode_{safe_key}.pt")

            ep_path = None
            for p in possible_paths:
                if p.exists():
                    ep_path = p
                    break

            if ep_path is None:
                continue

            try:
                payload = torch.load(ep_path, map_location="cpu")
                latents = payload.get("latents", {})
                if isinstance(latents, dict):
                    self._episode_latent_cache[episode_key] = latents
                    self._episode_ready[episode_key] = True
                    logger.debug("加载 episode_key=%s: %d 个 latent", episode_key, len(latents))
            except Exception as exc:
                logger.warning("加载 episode_key=%s cache 失败: %s", episode_key, exc)

    def _poll_chunked_cache(self) -> None:
        """后台线程：定期检查分块 cache 目录，补充 _episode_latent_cache。"""
        from src.train.latent_cache import build_episode_ready_path, list_completed_episodes

        poll_interval = 1.0
        encoder_state_path = self._cache_dir / "encoder_state.json" if self._cache_dir else None

        while True:
            if self._cache_dir is not None:
                # 读取 encoder 进度
                current_episode = -1
                if encoder_state_path and encoder_state_path.exists():
                    try:
                        import json as json_lib
                        state = json_lib.loads(encoder_state_path.read_text(encoding="utf-8"))
                        current_episode = state.get("current_episode", -1)
                    except Exception:
                        pass

                completed = list_completed_episodes(self._cache_dir)
                for episode_key in completed:
                    if episode_key in self._episode_ready and self._episode_ready[episode_key]:
                        continue
                    # 从 episode_key 提取整数 episode_id（格式："{scene}_{episode_id}"）
                    try:
                        _, ep_id_str = episode_key.rsplit("_", 1)
                        ep_id = int(ep_id_str)
                    except (ValueError, IndexError):
                        try:
                            ep_id = int(episode_key)
                        except ValueError:
                            ep_id = -1

                    # 尝试多种可能的文件名格式
                    possible_paths = []
                    if ep_id >= 0:
                        possible_paths.append(self._cache_dir / f"episode_{ep_id:04d}.pt")
                    safe_key = episode_key.replace(" ", "_").replace("/", "_").replace("\\", "_")
                    possible_paths.append(self._cache_dir / f"episode_{safe_key}.pt")

                    ep_path = None
                    for p in possible_paths:
                        if p.exists():
                            ep_path = p
                            break

                    if ep_path:
                        try:
                            payload = torch.load(ep_path, map_location="cpu")
                            latents = payload.get("latents", {})
                            if isinstance(latents, dict):
                                self._episode_latent_cache[episode_key] = latents
                                self._episode_ready[episode_key] = True
                                logger.info("检测到 episode_key=%s 就绪（%d latent）", episode_key, len(latents))
                        except Exception:
                            pass

                # 检测 encoder 是否卡住（encoder 当前在编码的 episode 远落后于训练需要的）
                # 找出最大的已就绪 episode_id
                max_ready_ep_id = -1
                for k, ready in self._episode_ready.items():
                    if ready:
                        try:
                            _, ep_str = k.rsplit("_", 1)
                            ep_num = int(ep_str)
                            if ep_num > max_ready_ep_id:
                                max_ready_ep_id = ep_num
                        except (ValueError, IndexError):
                            pass
                if current_episode >= 0 and max_ready_ep_id >= 0 and current_episode < max_ready_ep_id - 10:
                    logger.warning(
                        "encoder 进度落后: encoder 在 episode %d, 已就绪到 episode %d, 差距=%d",
                        current_episode, max_ready_ep_id, max_ready_ep_id - current_episode
                    )

            time.sleep(poll_interval)

    def _reload_cache_from_file(self, cache_path: Path, *, force: bool = False) -> None:
        """从 .pt 文件重新加载 cache（用于接收 encoder server 写入的新数据）。"""
        try:
            mtime_sec = cache_path.stat().st_mtime
            if not force and self._cache_mtime_sec is not None and mtime_sec <= self._cache_mtime_sec:
                return
            payload = torch.load(cache_path, map_location="cpu")
            latents = payload.get("latents", {}) if isinstance(payload, dict) else {}
            if isinstance(latents, dict):
                loaded = 0
                with self._cache_lock:
                    for key, value in latents.items():
                        if key not in self._latent_cache and isinstance(value, torch.Tensor):
                            self._latent_cache[key] = value.detach().cpu()
                            loaded += 1
                    # 先拍快照再批量删除，避免迭代同一个 set 时发生 size 变化。
                    resolved_pending = [k for k in self._pending_latents if k in self._latent_cache]
                    if resolved_pending:
                        self._pending_latents.difference_update(resolved_pending)
                self._cache_mtime_sec = mtime_sec
                if loaded > 0:
                    logger.debug("从 cache 文件加载了 %d 个新 latent（累计 %d）", loaded, len(self._latent_cache))
        except Exception as exc:
            logger.warning("reload cache 文件失败: %s", exc)

    def _cache_poll_loop(self) -> None:
        """后台线程：定期检查 cache 文件，补充 _latent_cache。"""
        poll_interval = 1.0
        while True:
            if self.latent_cache_path and self.latent_cache_path.exists():
                self._reload_cache_from_file(self.latent_cache_path)
            time.sleep(poll_interval)

    def _request_encode(self, image_path: str) -> None:
        """向 encoder server 发送编码请求。"""
        if self.encoder_queue is not None:
            try:
                self.encoder_queue.put(image_path)
            except Exception as exc:
                logger.warning("encoder_queue.put 失败: %s", exc)
        # 降级：写请求文件
        if self._pending_request_file is not None:
            try:
                with open(self._pending_request_file, "a") as f:
                    f.write(image_path + "\n")
            except Exception as exc:
                logger.warning("request_file write 失败: %s", exc)

    def _encode_latent(self, sample: dict[str, Any]) -> torch.Tensor:
        """阻塞式获取 latent：先查缓存，再等待 encoder server。"""
        image_path = str(sample["image_path"])

        # 分块模式：按 episode 查找
        if self._chunk_mode and self._cache_dir is not None:
            metadata = sample.get("metadata", {})
            scene = metadata.get("scene", "unknown")
            episode_id = int(sample.get("episode_id", 0))
            episode_key = f"{scene}_{episode_id}"
            return self._encode_latent_chunked(image_path, episode_key)

        # 单文件模式（向后兼容）
        with self._cache_lock:
            cached = self._latent_cache.get(image_path)
        if cached is not None:
            return cached
        # 发送请求（只发一次）
        if image_path not in self._pending_latents:
            self._pending_latents.add(image_path)
            self._request_encode(image_path)
        # Spin-wait：轮询 cache，直到 encoder server 写入
        poll_interval = 0.1
        waited = 0.0
        last_file_reload = 0.0
        file_reload_interval = 0.5
        next_requeue_at = 10.0
        stall_timeout = 45.0
        hard_timeout = 180.0
        while True:
            time.sleep(poll_interval)
            waited += poll_interval
            # DataLoader worker 进程里后台线程可能失活，这里做同步兜底重载。
            if self.latent_cache_path is not None and self.latent_cache_path.exists():
                if waited - last_file_reload >= file_reload_interval:
                    self._reload_cache_from_file(self.latent_cache_path)
                    last_file_reload = waited
            with self._cache_lock:
                cached = self._latent_cache.get(image_path)
            if cached is not None:
                logger.debug("latent ready after %.1fs: %s", waited, image_path)
                return cached
            # 可选：每 10s 重新触发一次请求（防止请求丢失）。
            # 注意必须做边沿触发，避免 10.0~10.9s 期间重复洪泛请求。
            if waited >= next_requeue_at:
                self._request_encode(image_path)
                next_requeue_at += 10.0
            heartbeat_age = None
            if self._encoder_heartbeat_file is not None and self._encoder_heartbeat_file.exists():
                try:
                    heartbeat_age = max(0.0, time.time() - self._encoder_heartbeat_file.stat().st_mtime)
                except OSError:
                    heartbeat_age = None
            if waited >= stall_timeout and (heartbeat_age is None or heartbeat_age > stall_timeout):
                raise RuntimeError(
                    "检测到 encoder server 疑似卡死（心跳停滞）。"
                    f" waited={waited:.1f}s, heartbeat_age={heartbeat_age}, image={image_path}"
                )
            if waited >= hard_timeout:
                raise RuntimeError(
                    "encoder 返回超时（非心跳停滞，但单样本等待超过硬阈值）。"
                    f" waited={waited:.1f}s, image={image_path}"
                )

    def _encode_latent_chunked(self, image_path: str, episode_key: str) -> torch.Tensor:
        """分块模式：等待 episode 就绪后获取 latent。

        使用轮转编码后，所有 episode 均匀推进，训练时遇到的 episode 大概率已部分完成。
        episode_key 格式为 "{scene}_{episode_id}"，确保跨 scene 的 episode 唯一性。
        等待逻辑：
        1. 如果 episode 部分完成（有 .ready marker），只等待当前 batch 的图像
        2. 如果 episode 还未开始，根据 encoder 进度估算等待时间
        3. 心跳检测防止 encoder 卡死
        """
        from src.train.latent_cache import build_episode_ready_path

        poll_interval = 0.2
        waited = 0.0
        min_timeout = 15.0  # 最少等待 15 秒
        max_hard_timeout = 300.0  # 硬超时上限

        while True:
            # 检查 episode 是否就绪
            if self._episode_ready.get(episode_key, False):
                latents = self._episode_latent_cache.get(episode_key, {})
                if image_path in latents:
                    return latents[image_path].detach().cpu()

            # 检查 episode ready marker
            if self._cache_dir is not None:
                ready_path = build_episode_ready_path(self._cache_dir, episode_key)
                if ready_path.exists():
                    # episode 标记为就绪但 latents 尚未加载，尝试同步加载
                    self._reload_chunked_cache()
                    latents = self._episode_latent_cache.get(episode_key, {})
                    if image_path in latents:
                        return latents[image_path].detach().cpu()

            time.sleep(poll_interval)
            waited += poll_interval

            # 动态超时估算
            stall_timeout = self._estimate_stall_timeout(episode_key)
            hard_timeout = max(max_hard_timeout, waited + 10.0)

            # 心跳检测
            heartbeat_age = None
            if self._encoder_heartbeat_file is not None and self._encoder_heartbeat_file.exists():
                try:
                    heartbeat_age = max(0.0, time.time() - self._encoder_heartbeat_file.stat().st_mtime)
                except OSError:
                    heartbeat_age = None

            if waited >= stall_timeout and (heartbeat_age is None or heartbeat_age > stall_timeout):
                raise RuntimeError(
                    f"检测到 encoder server 疑似卡死（心跳停滞）。"
                    f" episode_key={episode_key}, waited={waited:.1f}s, heartbeat_age={heartbeat_age}"
                )
            if waited >= hard_timeout:
                raise RuntimeError(
                    f"encoder 返回超时（非心跳停滞，但 episode {episode_key} 等待超过硬阈值）。"
                    f" waited={waited:.1f}s, image={image_path}"
                )

    def _estimate_stall_timeout(self, episode_key: str) -> float:
        """基于 encoder 进度估算合理的 stall timeout。

        使用轮转编码策略后，encoder 会均匀推进所有 episode。
        我们根据 encoder_state.json 中的进度来估算当前 episode 何时应该完成。
        episode_key 格式为 "{scene}_{episode_id}"。

        Returns:
            估算的 stall_timeout（秒）
        """
        # 从 episode_key 解析 episode_id
        try:
            _, ep_id_str = episode_key.rsplit("_", 1)
            episode_id = int(ep_id_str)
        except (ValueError, IndexError):
            episode_id = -1

        if self._cache_dir is None:
            return 60.0  # 默认 60 秒

        encoder_state_path = self._cache_dir / "encoder_state.json"
        if not encoder_state_path.exists():
            return 60.0

        try:
            import json
            state = json.loads(encoder_state_path.read_text(encoding="utf-8"))
            current_ep = state.get("current_episode", -1)
            episode_progress = state.get("episode_progress", 0)
            episode_total = state.get("episode_total", 1)

            # 轮转编码时，encoder 每轮处理每个 episode 一个 batch
            # 估算当前 episode 的进度
            if current_ep == episode_id:
                # encoder 正在编码这个 episode
                return 60.0  # 正常等待
            elif current_ep > episode_id:
                # encoder 已经超过这个 episode
                # 估算 episode 完成率
                progress_ratio = episode_progress / max(1, episode_total)
                remaining = (1.0 - progress_ratio) * 30.0  # 假设每 episode 最多 30 秒
                return max(15.0, remaining)
            else:
                # encoder 还在更早的 episode
                # 估算需要多少轮才能到当前 episode
                episodes_behind = episode_id - current_ep
                batch_per_ep = episode_total / 64  # 估算每 episode 需要多少 batch
                estimated_delay = episodes_behind * batch_per_ep * 0.5  # 每 batch 约 0.5 秒
                return max(30.0, min(estimated_delay, 120.0))
        except Exception:
            return 60.0

    def disable_encoder_after_warmup(self) -> None:
        """预编码完成后关闭 encoder，便于 DataLoader 多进程并行读取缓存。"""
        self.image_encoder = None

    def _warmup_latent_cache(self) -> None:
        cache_path = self.latent_cache_path
        # 分块目录模式不通过单文件 warmup 加载，由后台线程按需补充
        if cache_path is not None and cache_path.is_dir():
            logger.info("latent cache 为分块目录，跳过单文件 warmup，依赖后台线程按需加载。")
            return
        # 当 latent_cache_path 存在但 image_encoder=None 时，直接从文件加载缓存（用于评估模式）
        if cache_path is not None and cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu")
            latents = payload.get("latents", {}) if isinstance(payload, dict) else {}
            cached_latent_dim = payload.get("latent_dim") if isinstance(payload, dict) else None
            if cached_latent_dim is not None and int(cached_latent_dim) != int(self.latent_dim):
                logger.warning(
                    "检测到 latent 缓存维度不匹配，忽略旧缓存: cache_dim=%s, expected_dim=%s, cache=%s",
                    str(cached_latent_dim),
                    str(self.latent_dim),
                    str(cache_path),
                )
            elif isinstance(latents, dict):
                for key, value in latents.items():
                    if not (isinstance(key, str) and isinstance(value, torch.Tensor)):
                        continue
                    matches_flat_dim = int(value.numel()) == int(self.latent_dim)
                    matches_patch_layout = (
                        self.expected_num_patches > 0
                        and self.expected_token_dim > 0
                        and value.dim() == 2
                        and int(value.size(0)) == int(self.expected_num_patches)
                        and int(value.size(1)) == int(self.expected_token_dim)
                    )
                    if matches_flat_dim or matches_patch_layout:
                        self._latent_cache[key] = value.detach().cpu()
            unique_paths = {str(sample["image_path"]) for sample in self.samples if "image_path" in sample}
            missing_paths = [path for path in unique_paths if path not in self._latent_cache]
            if not missing_paths:
                logger.info("latent 缓存已命中: %d 张图像，无需预编码。", len(self._latent_cache))
                if self.on_latent_progress is not None:
                    self.on_latent_progress(0, 0)
                return
            # 如果 image_encoder=None（评估模式）但有缓存缺失，发出警告
            if self.image_encoder is None:
                logger.warning(
                    "latent 缓存部分缺失（缺失=%d/%d），在评估模式下应使用完整缓存。",
                    len(missing_paths),
                    len(unique_paths),
                )
            else:
                # 有 encoder，正常预编码
                logger.info(
                    "开始预编码 latent: 总图像=%d, 待编码=%d, workers=%d, batch_size=%d",
                    len(unique_paths),
                    len(missing_paths),
                    self.encoder_num_workers,
                    self.encoder_batch_size,
                )
                flush_every = 500
                for start in range(0, len(missing_paths), self.encoder_batch_size):
                    batch_paths = missing_paths[start : start + self.encoder_batch_size]
                    batch_outputs = self.image_encoder.encode_image_paths(batch_paths)
                    for image_path, output in zip(batch_paths, batch_outputs, strict=True):
                        self._latent_cache[image_path] = output.z.detach().cpu()
                    done = start + len(batch_paths)
                    if self.on_latent_progress is not None:
                        self.on_latent_progress(done, len(missing_paths))
                    if done % flush_every == 0:
                        logger.info("latent 预编码进度: %d/%d", done, len(missing_paths))
                        self._save_latent_cache(cache_path)
                self._save_latent_cache(cache_path)
                logger.info("latent 预编码完成: 新增=%d, 缓存总量=%d", len(missing_paths), len(self._latent_cache))

    def _save_latent_cache(self, cache_path: Path | None) -> None:
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "latent_dim": self.latent_dim,
                "latents": self._latent_cache,
            },
            cache_path,
        )

    def _build_action_vec(self, sample: dict[str, Any]) -> torch.Tensor:
        move = float(sample.get("move_ahead_distance", 0.0))
        yaw = float(sample.get("delta_yaw", 0.0))
        pitch = float(sample.get("delta_pitch", 0.0))
        action = torch.tensor([move, yaw, pitch], dtype=torch.float32)
        if self.action_dim <= 3:
            return action[: self.action_dim]
        padded = torch.zeros(self.action_dim, dtype=torch.float32)
        padded[:3] = action
        return padded

    def __getitem__(self, idx: int) -> dict[str, Any]:
        index_data = self._training_indices[idx]
        history_samples = [self.samples[i] for i in index_data["history_indices"]]
        future_samples = [self.samples[i] for i in index_data["future_indices"]]
        action_source_samples = [self.samples[i] for i in index_data["action_source_indices"]]
        z_history = torch.stack([self._encode_latent(sample) for sample in history_samples], dim=0)
        action_history = torch.stack([self._build_action_vec(sample) for sample in history_samples], dim=0)
        z_future = torch.stack([self._encode_latent(sample) for sample in future_samples], dim=0)
        gt_action_future = torch.stack([self._build_action_vec(sample) for sample in action_source_samples], dim=0)
        z_next = z_future[0]
        gt_action = gt_action_future[0]
        env_context = build_env_context(history_samples[-1].get("metadata", {}))
        return {
            "z_history": z_history,
            "action_history": action_history,
            "z_next": z_next,
            "gt_action": gt_action,
            "z_future": z_future,
            "gt_action_future": gt_action_future,
            "env_context": env_context,
            "temporal_stride": torch.tensor(index_data["temporal_stride"], dtype=torch.int64),
        }

