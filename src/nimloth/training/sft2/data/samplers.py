"""SFT2 固定长度轨迹窗口的 DataLoader batch sampler。"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler

from nimloth.rollout.transitions import TransitionSample


class TrajectoryWindowBatchSampler(Sampler[list[int]]):
    """把 ``B`` 个长度为 ``H`` 的连续 transition 窗口展平成 DataLoader 行。

    每个窗口对应真实状态 ``s_0 ... s_H`` 和动作 ``a_0 ... a_{H-1}``。
    DataLoader 仍读取逐 transition 数据，但输出索引严格按 window-major、
    time-minor 排列，因此 SFT2 assembler 可以无歧义地恢复 ``(B,H)``。
    """

    def __init__(
        self,
        samples: Sequence[TransitionSample],
        *,
        history_size: int,
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        shuffle_windows: bool = False,
        seed: int = 0,
        drop_last: bool = False,
        max_images_per_batch: int | None = None,
        max_transition_rows_per_batch: int | None = None,
        pad_to_equal_batches: bool = True,
    ) -> None:
        if history_size < 1:
            raise ValueError("history_size must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if num_replicas < 1:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError(
                f"rank {rank} out of range for num_replicas={num_replicas}"
            )
        if max_images_per_batch is not None and max_images_per_batch < 1:
            raise ValueError("max_images_per_batch must be positive")
        if shuffle_windows and max_images_per_batch is not None:
            raise ValueError(
                "shuffle_windows cannot be combined with variable image-budget packing"
            )
        if (
            max_transition_rows_per_batch is not None
            and max_transition_rows_per_batch < 1
        ):
            raise ValueError("max_transition_rows_per_batch must be positive")

        self.samples = samples
        self.history_size = int(history_size)
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.shuffle_windows = bool(shuffle_windows)
        self.seed = int(seed)
        self.pad_to_equal_batches = bool(pad_to_equal_batches)
        self.epoch = 0

        windows = self._build_windows(samples, self.history_size)
        self._windows = tuple(windows)
        self._pack_options = {
            "samples": samples,
            "history_size": self.history_size,
            "batch_size": self.batch_size,
            "drop_last": drop_last,
            "max_images_per_batch": max_images_per_batch,
            "max_transition_rows_per_batch": max_transition_rows_per_batch,
        }
        self._base_batches = self._pack_windows(
            windows,
            **self._pack_options,
        )
        if self.pad_to_equal_batches:
            self.num_batches = (
                math.ceil(len(self._base_batches) / self.num_replicas)
                if self._base_batches
                else 0
            )
            self.total_size = self.num_batches * self.num_replicas
        else:
            remaining = len(self._base_batches) - self.rank
            self.num_batches = max(
                0,
                (remaining + self.num_replicas - 1) // self.num_replicas,
            )
            self.total_size = len(self._base_batches)

    @staticmethod
    def _build_windows(
        samples: Sequence[TransitionSample],
        history_size: int,
    ) -> list[tuple[int, ...]]:
        """枚举所有拥有 ``H`` 个动作和 ``H+1`` 个真实状态的滑动窗口。"""

        by_record: dict[str, list[int]] = defaultdict(list)
        for index, sample in enumerate(samples):
            by_record[sample.record_id].append(index)

        windows: list[tuple[int, ...]] = []
        for indices in by_record.values():
            indices.sort(key=lambda index: samples[index].step_index)
            for start in range(0, len(indices) - history_size + 1):
                window = tuple(indices[start : start + history_size])
                steps = [samples[index].step_index for index in window]
                if any(
                    right != left + 1
                    for left, right in zip(steps, steps[1:])
                ):
                    continue
                # 每个动作都必须有真实 next observation；缺失 target 的 legacy
                # transition 不能伪装成 LeWM 序列。
                if not all(
                    samples[index].next_prefix_messages is not None
                    and samples[index].next_prefix_image_paths is not None
                    for index in window
                ):
                    continue
                windows.append(window)
        return windows

    @classmethod
    def _pack_windows(
        cls,
        windows: Sequence[tuple[int, ...]],
        *,
        samples: Sequence[TransitionSample],
        history_size: int,
        batch_size: int,
        drop_last: bool,
        max_images_per_batch: int | None,
        max_transition_rows_per_batch: int | None,
    ) -> list[list[int]]:
        """按窗口数及可选图片预算打包，单个超预算窗口仍保留。"""

        batches: list[list[int]] = []
        current_windows: list[tuple[int, ...]] = []
        current_images = 0

        def flush() -> None:
            nonlocal current_windows, current_images
            if current_windows:
                batches.append(
                    [index for window in current_windows for index in window]
                )
            current_windows = []
            current_images = 0

        row_limit = max_transition_rows_per_batch
        for window in windows:
            image_cost = cls._window_image_cost(window, samples)
            next_window_count = len(current_windows) + 1
            exceeds_window_count = next_window_count > batch_size
            exceeds_rows = (
                row_limit is not None
                and next_window_count * history_size > row_limit
            )
            exceeds_images = (
                max_images_per_batch is not None
                and current_images + image_cost > max_images_per_batch
            )
            if current_windows and (
                exceeds_window_count or exceeds_rows or exceeds_images
            ):
                flush()
            current_windows.append(window)
            current_images += image_cost
        if current_windows and not (
            drop_last
            and max_images_per_batch is None
            and len(current_windows) < batch_size
        ):
            flush()
        return batches

    @staticmethod
    def _window_image_cost(
        window: tuple[int, ...],
        samples: Sequence[TransitionSample],
    ) -> int:
        """估算三条顺序执行的 forward 中单次最大图片引用数。"""

        current = sum(len(samples[index].prefix_image_paths) for index in window)
        targets = sum(
            len(samples[index].next_prefix_image_paths or ())
            for index in window
        )
        online_tail = len(samples[window[-1]].next_prefix_image_paths or ())
        return max(current, targets, online_tail)

    @property
    def window_count(self) -> int:
        """返回进入本 sampler 的有效窗口总数，便于启动时校验数据。"""

        return sum(
            len(batch) // self.history_size
            for batch in self._base_batches
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle and self.shuffle_windows:
            windows = list(self._windows)
            rng.shuffle(windows)
            batches = self._pack_windows(windows, **self._pack_options)
        else:
            batches = list(self._base_batches)
            if self.shuffle:
                rng.shuffle(batches)

        if self.pad_to_equal_batches and self.total_size > len(batches):
            if not batches:
                return iter(())
            repeats = math.ceil(self.total_size / len(batches))
            batches = (batches * repeats)[: self.total_size]
        elif self.pad_to_equal_batches:
            batches = batches[: self.total_size]

        return iter(batches[self.rank : self.total_size : self.num_replicas])

    def __len__(self) -> int:
        return self.num_batches


__all__ = ["TrajectoryWindowBatchSampler"]
