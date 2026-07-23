"""SFT2 单步 ownership 与真实短历史的 DataLoader batch sampler。"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler

from nimloth.rollout.transitions import TransitionContextIndex, TransitionSample


class TrajectoryWindowBatchSampler(Sampler[list[TransitionContextIndex]]):
    """把 ``B`` 个最长为 ``H`` 的单步上下文展平成 DataLoader 行。

    每个 transition 恰好作为一次当前 step。episode 开头使用真实的短上下文，
    后续 step 使用最多 ``H`` 个 state/action；不伪造 padding，也不在重叠窗口中
    重复计算旧 step 的 loss。
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
        backbone_rows_per_forward: int | None = None,
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
            "backbone_rows_per_forward": backbone_rows_per_forward,
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
        """为每个拥有真实 next state 的 step 构造一次最长 ``H`` 的上下文。"""

        by_record: dict[str, list[int]] = defaultdict(list)
        for index, sample in enumerate(samples):
            by_record[sample.record_id].append(index)

        windows: list[tuple[int, ...]] = []
        for indices in by_record.values():
            indices.sort(key=lambda index: samples[index].step_index)
            consecutive: list[int] = []
            for index in indices:
                if consecutive and (
                    samples[index].step_index
                    != samples[consecutive[-1]].step_index + 1
                ):
                    consecutive = []
                consecutive.append(index)
                sample = samples[index]
                if (
                    sample.next_prefix_messages is None
                    or sample.next_prefix_image_paths is None
                ):
                    continue
                windows.append(tuple(consecutive[-history_size:]))
        # 同一 microbatch 只能共享一个真实 context length；按长度稳定分组后再打包，
        # 避免每条 trajectory 开头的短上下文都退化成 singleton batch。
        windows.sort(key=len)
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
        backbone_rows_per_forward: int | None,
    ) -> list[list[TransitionContextIndex]]:
        """按窗口数及可选图片预算打包，单个超预算窗口仍保留。"""

        batches: list[list[TransitionContextIndex]] = []
        current_windows: list[tuple[int, ...]] = []
        current_images = 0

        def flush() -> None:
            nonlocal current_windows, current_images
            if current_windows:
                rows: list[TransitionContextIndex] = []
                for window in current_windows:
                    for position, index in enumerate(window):
                        rows.append(
                            TransitionContextIndex(
                                sample_index=index,
                                context_length=len(window),
                                is_current_step=position == len(window) - 1,
                            )
                        )
                batches.append(rows)
            current_windows = []
            current_images = 0

        row_limit = max_transition_rows_per_batch
        for window in windows:
            image_cost = cls._window_image_cost(
                window,
                samples,
                sequential_rows=backbone_rows_per_forward == 1,
            )
            next_window_count = len(current_windows) + 1
            changes_context_length = bool(
                current_windows and len(current_windows[0]) != len(window)
            )
            exceeds_window_count = next_window_count > batch_size
            exceeds_rows = (
                row_limit is not None
                and next_window_count * len(window) > row_limit
            )
            exceeds_images = (
                backbone_rows_per_forward != 1
                and max_images_per_batch is not None
                and current_images + image_cost > max_images_per_batch
            )
            if current_windows and (
                changes_context_length
                or exceeds_window_count
                or exceeds_rows
                or exceeds_images
            ):
                flush()
            current_windows.append(window)
            if backbone_rows_per_forward == 1:
                current_images = max(current_images, image_cost)
            else:
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
        *,
        sequential_rows: bool,
    ) -> int:
        """按实际逐 row forward 估算单次最大图片引用数。"""

        current_images = [len(samples[index].prefix_image_paths) for index in window]
        next_state = len(samples[window[-1]].next_prefix_image_paths or ())
        if sequential_rows:
            return max((*current_images, next_state))
        return max(sum(current_images), next_state)

    @property
    def window_count(self) -> int:
        """返回进入本 sampler 的有效窗口总数，便于启动时校验数据。"""

        return sum(
            sum(1 for row in batch if row.is_current_step)
            for batch in self._base_batches
        )

    @property
    def current_steps_per_batch(self) -> tuple[int, ...]:
        """返回每个全局 microbatch 实际拥有的 current step 数。"""

        return tuple(
            sum(1 for row in batch if row.is_current_step)
            for batch in self._base_batches
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[TransitionContextIndex]]:
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle and self.shuffle_windows:
            windows = []
            for context_length in range(1, self.history_size + 1):
                same_length = [
                    window
                    for window in self._windows
                    if len(window) == context_length
                ]
                rng.shuffle(same_length)
                windows.extend(same_length)
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
