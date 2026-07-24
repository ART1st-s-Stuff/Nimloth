"""Trajectory source 协议和离线 JSONL 实现。"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Protocol

from nimloth.rollout.schema import RolloutTrajectory
from nimloth.rollout.storage import load_trajectories


class RolloutCollector(Protocol):
    def collect(
        self,
        *,
        num_episodes: int,
        max_steps_per_episode: int = 20,
        output_dir: Path | None = None,
    ) -> list[RolloutTrajectory]:
        ...


class JSONLRolloutCollector:
    """以确定性顺序循环读取预先收集的 trajectory 文件。"""

    def __init__(self, sources: list[Path] | None = None, loop: bool = True) -> None:
        self._sources = list(sources) if sources else []
        self._loop = loop
        self._all_trajectories: list[RolloutTrajectory] | None = None
        self._cursor = 0
        self._call_count = 0

    def _load_all(self) -> list[RolloutTrajectory]:
        trajectories: list[RolloutTrajectory] = []
        files = self._expand_sources()
        if not files:
            raise FileNotFoundError(
                "JSONLRolloutCollector 未找到任何 JSONL 文件："
                f"sources={self._sources}"
            )
        for path in files:
            try:
                trajectories.extend(load_trajectories(path))
            except Exception as error:
                print(
                    json.dumps(
                        {"jsonl_load_warning": str(path), "error": str(error)}
                    ),
                    flush=True,
                )
        if not trajectories:
            raise ValueError(
                f"JSONLRolloutCollector read no trajectories from {len(files)} files"
            )
        random.Random(42).shuffle(trajectories)
        return trajectories

    def _expand_sources(self) -> list[Path]:
        files: list[Path] = []
        for source in self._sources:
            if source.is_dir():
                for pattern in ("**/*.jsonl", "**/*.jsonl.gz"):
                    files.extend(sorted(source.glob(pattern)))
            elif source.exists():
                files.append(source)
        return files

    @property
    def total_trajectories(self) -> int:
        if self._all_trajectories is None:
            self._all_trajectories = self._load_all()
        return len(self._all_trajectories)

    def collect(
        self,
        *,
        num_episodes: int,
        max_steps_per_episode: int = 20,
        output_dir: Path | None = None,
    ) -> list[RolloutTrajectory]:
        del max_steps_per_episode, output_dir
        self._call_count += 1
        if self._all_trajectories is None:
            self._all_trajectories = self._load_all()
        total = len(self._all_trajectories)
        if total == 0:
            return []

        result: list[RolloutTrajectory] = []
        needed = num_episodes
        while needed > 0:
            remaining = total - self._cursor
            take = min(needed, remaining)
            if take:
                result.extend(
                    self._all_trajectories[self._cursor : self._cursor + take]
                )
                self._cursor += take
                needed -= take
            if self._cursor >= total:
                if not self._loop:
                    break
                self._cursor = 0
        return result
