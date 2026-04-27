"""Encoder cache 协议层：锁、状态写入与 cache 追加。"""

from __future__ import annotations

import fcntl
import json
import time
from pathlib import Path
from typing import TextIO

import torch


def acquire_lock(lock_path: Path, *, timeout_sec: float = 30.0, poll_sec: float = 0.2) -> TextIO:
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
                raise TimeoutError(
                    f"获取缓存锁超时: lock={lock_path}, waited={waited:.1f}s, timeout={timeout_sec:.1f}s"
                )
            time.sleep(poll_sec)


def release_lock(fd: TextIO) -> None:
    fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
    fd.close()


def append_single_file_latents(cache_path: Path, new_latents: dict[str, torch.Tensor], latent_dim: int) -> None:
    lock_path = cache_path.with_suffix(".lock")
    lock_fd: TextIO | None = None
    try:
        try:
            lock_fd = acquire_lock(lock_path, timeout_sec=30.0, poll_sec=0.2)
        except TimeoutError:
            lock_fd = None
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
            release_lock(lock_fd)


def append_episode_latents(
    *,
    episode_path: Path,
    episode_key: str,
    new_latents: dict[str, torch.Tensor],
    latent_dim: int,
) -> None:
    lock_path = episode_path.with_suffix(".lock")
    lock_fd: TextIO | None = None
    try:
        try:
            lock_fd = acquire_lock(lock_path, timeout_sec=30.0, poll_sec=0.2)
        except TimeoutError:
            lock_fd = None
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
            release_lock(lock_fd)


def write_json_state(path: Path, payload: dict) -> None:
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass
