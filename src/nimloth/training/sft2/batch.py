"""SFT2 transition 元数据到公共 Backbone batch 的装配。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import torch

from nimloth.backbone import BackboneBatch, BackboneInputBuilder
from nimloth.rollout import TransitionBatch
from nimloth.rollout.transitions import (
    ContextualTransitionSample,
    RolloutTransitionSample,
    TransitionSample,
    transition_training_item,
)
from nimloth.training.sft2.history_cache import StateKey


@dataclass(frozen=True)
class CachedNextBatch:
    """DataLoader worker 已按 prompt 去重的下一状态输入。"""

    keys: tuple[str, ...]
    batch: BackboneBatch


@dataclass(frozen=True)
class SFT2Batch:
    """SFT2 的 ``B`` 个 current step 及其相同长度真实 context。"""

    transitions: TransitionBatch
    online_tail: BackboneBatch
    history_size: int
    sample_weights: torch.Tensor
    next_image_paths: tuple[str, ...] = ()
    dino_grid_target: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.history_size < 1:
            raise ValueError("SFT2 history_size must be positive")
        row_count = len(self.transitions.trajectory_steps)
        if row_count == 0 or row_count % self.history_size != 0:
            raise ValueError(
                "SFT2 transition rows must contain complete history windows: "
                f"rows={row_count}, history_size={self.history_size}"
            )
        for start in range(0, row_count, self.history_size):
            window = self.transitions.trajectory_steps[
                start : start + self.history_size
            ]
            record_ids = {record_id for record_id, _step in window}
            step_indices = [step for _record_id, step in window]
            if len(record_ids) != 1 or any(
                right != left + 1
                for left, right in zip(step_indices, step_indices[1:])
            ):
                raise ValueError(
                    "SFT2 history window must contain consecutive steps from "
                    f"one trajectory, got {window}"
                )
        if self.sample_weights.shape != (self.batch_size,):
            raise ValueError(
                "SFT2 sample weights must have shape (B,), "
                f"got {tuple(self.sample_weights.shape)} for B={self.batch_size}"
            )
        unique_weights = set(float(value) for value in self.sample_weights.tolist())
        if not unique_weights.issubset({0.0, 1.0}) or len(unique_weights) != 1:
            raise ValueError(
                "SFT2 batches must be entirely real (weight=1) or padding "
                f"(weight=0), got {sorted(unique_weights)}"
            )
        if self.next_image_paths and len(self.next_image_paths) != self.batch_size:
            raise ValueError(
                "SFT2 next-image paths must align with current steps: "
                f"paths={len(self.next_image_paths)}, B={self.batch_size}"
            )
        if self.dino_grid_target is not None and (
            self.dino_grid_target.ndim < 1
            or self.dino_grid_target.shape[0] != self.batch_size
        ):
            raise ValueError(
                "SFT2 DINO-grid target must start with the current batch size "
                f"B={self.batch_size}, got {tuple(self.dino_grid_target.shape)}"
            )

    @property
    def batch_size(self) -> int:
        return len(self.transitions.trajectory_steps) // self.history_size

    @property
    def current(self) -> BackboneBatch:
        return self.transitions.current

    @property
    def next(self) -> BackboneBatch:
        return self.transitions.next

    @property
    def action_indices(self) -> torch.Tensor:
        return self.transitions.action_indices.reshape(
            self.batch_size,
            self.history_size,
        )

    @property
    def value_targets(self) -> torch.Tensor:
        return self.transitions.value_targets.reshape(
            self.batch_size,
            self.history_size,
        )

    @property
    def next_indices(self) -> torch.Tensor:
        return self.transitions.next_indices.reshape(
            self.batch_size,
            self.history_size,
        )

    @property
    def current_action_indices(self) -> torch.Tensor:
        return self.action_indices[:, -1]

    @property
    def current_value_targets(self) -> torch.Tensor:
        return self.value_targets[:, -1]

    @property
    def current_next_indices(self) -> torch.Tensor:
        return self.next_indices[:, -1]

    @property
    def history_keys(self) -> tuple[tuple[StateKey, ...], ...]:
        rows = self.transitions.trajectory_steps
        return tuple(
            tuple(rows[start : start + self.history_size - 1])
            for start in range(0, len(rows), self.history_size)
        )

    @property
    def current_keys(self) -> tuple[StateKey, ...]:
        rows = self.transitions.trajectory_steps
        return tuple(
            rows[end - 1]
            for end in range(self.history_size, len(rows) + 1, self.history_size)
        )

    @property
    def is_padding(self) -> bool:
        return bool(torch.count_nonzero(self.sample_weights).item() == 0)


@dataclass(frozen=True)
class SFT2RolloutBatch:
    """``B`` fixed-length future rollouts starting from one real state each."""

    transitions: TransitionBatch
    online_tail: BackboneBatch
    prediction_horizon: int
    sample_weights: torch.Tensor
    next_image_paths: tuple[str, ...] = ()
    dino_grid_target: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.prediction_horizon < 1:
            raise ValueError("SFT2 prediction_horizon must be positive")
        row_count = len(self.transitions.trajectory_steps)
        if row_count == 0 or row_count % self.prediction_horizon != 0:
            raise ValueError(
                "SFT2 rollout rows must contain complete future windows: "
                f"rows={row_count}, T={self.prediction_horizon}"
            )
        for start in range(0, row_count, self.prediction_horizon):
            window = self.transitions.trajectory_steps[
                start : start + self.prediction_horizon
            ]
            record_ids = {record_id for record_id, _step in window}
            steps = [step for _record_id, step in window]
            if len(record_ids) != 1 or any(
                right != left + 1
                for left, right in zip(steps, steps[1:])
            ):
                raise ValueError(
                    "SFT2 rollout window must contain consecutive recorded "
                    f"actions from one trajectory, got {window}"
                )
        if self.sample_weights.shape != (self.batch_size,):
            raise ValueError(
                "SFT2 rollout sample weights must have shape (B,), "
                f"got {tuple(self.sample_weights.shape)} for B={self.batch_size}"
            )
        unique_weights = set(float(value) for value in self.sample_weights.tolist())
        if not unique_weights.issubset({0.0, 1.0}) or len(unique_weights) != 1:
            raise ValueError(
                "SFT2 rollout batches must be entirely real or padding, "
                f"got {sorted(unique_weights)}"
            )
        expected_targets = self.batch_size * self.prediction_horizon
        if self.next_image_paths and len(self.next_image_paths) != expected_targets:
            raise ValueError(
                "SFT2 rollout next-image paths must align with every target: "
                f"paths={len(self.next_image_paths)}, expected={expected_targets}"
            )
        if self.dino_grid_target is not None and (
            self.dino_grid_target.ndim < 2
            or tuple(self.dino_grid_target.shape[:2])
            != (self.batch_size, self.prediction_horizon)
        ):
            raise ValueError(
                "SFT2 rollout DINO target must start with (B,T), got "
                f"{tuple(self.dino_grid_target.shape)}"
            )

    @property
    def batch_size(self) -> int:
        return len(self.transitions.trajectory_steps) // self.prediction_horizon

    @property
    def current(self) -> BackboneBatch:
        return self.transitions.current

    @property
    def next(self) -> BackboneBatch:
        return self.transitions.next

    @property
    def action_sequences(self) -> torch.Tensor:
        return self.transitions.action_indices.reshape(
            self.batch_size,
            self.prediction_horizon,
        )

    @property
    def value_target_sequences(self) -> torch.Tensor:
        return self.transitions.value_targets.reshape(
            self.batch_size,
            self.prediction_horizon,
        )

    @property
    def next_indices(self) -> torch.Tensor:
        return self.transitions.next_indices.reshape(
            self.batch_size,
            self.prediction_horizon,
        )

    @property
    def current_keys(self) -> tuple[StateKey, ...]:
        rows = self.transitions.trajectory_steps
        return tuple(
            rows[start]
            for start in range(0, len(rows), self.prediction_horizon)
        )

    @property
    def is_padding(self) -> bool:
        return bool(torch.count_nonzero(self.sample_weights).item() == 0)


class SFT2BatchBuilder(Protocol):
    """DataLoader 输出到 SFT2 连续窗口 batch 的阶段契约。"""

    processor: Any

    def collate_transition_samples(self, batch: list[Any]) -> Any: ...

    def prepare(self, raw_batch: Any) -> SFT2Batch | SFT2RolloutBatch: ...


class SFT2BatchAssembler:
    """负责 SFT2 target 对齐；不包含任何 Qwen 模型或训练算法。"""

    def __init__(
        self,
        *,
        input_builder: BackboneInputBuilder,
        device: torch.device,
        history_size: int,
        prediction_horizon: int = 1,
    ) -> None:
        self.input_builder = input_builder
        self.device = device
        self.history_size = int(history_size)
        self.prediction_horizon = int(prediction_horizon)
        if self.history_size < 1:
            raise ValueError("SFT2 history_size must be positive")
        if self.prediction_horizon < 1:
            raise ValueError("SFT2 prediction_horizon must be positive")
        if self.prediction_horizon > 1 and self.history_size != 1:
            raise ValueError(
                "multi-step SFT2 rollout currently requires history_size=1, "
                f"got H={self.history_size}, T={self.prediction_horizon}"
            )

    @property
    def processor(self) -> Any:
        """预处理 cache 仍需访问 backend 的 processor。"""

        return self.input_builder.processor

    def collate_transition_samples(
        self,
        batch: list[
            TransitionSample | ContextualTransitionSample | RolloutTransitionSample
        ],
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for row in batch:
            if isinstance(row, ContextualTransitionSample):
                item = transition_training_item(row.sample)
                item["context_length"] = row.context_length
                item["is_current_step"] = row.is_current_step
                item["loss_weight"] = row.loss_weight
            elif isinstance(row, RolloutTransitionSample):
                item = transition_training_item(row.sample)
                item["prediction_horizon"] = row.prediction_horizon
                item["rollout_position"] = row.rollout_position
                item["is_current_step"] = row.rollout_position == 0
                item["needs_next_state"] = True
                item["loss_weight"] = row.loss_weight
            else:
                item = transition_training_item(row)
            items.append(item)
        return items

    def prepare(self, raw_batch: Any) -> SFT2Batch | SFT2RolloutBatch:
        """构造 current/next 模型输入和对齐后的 transition target。"""

        if isinstance(raw_batch, (SFT2Batch, SFT2RolloutBatch)):
            return raw_batch
        if isinstance(raw_batch, dict) and "current_enc_rows" in raw_batch:
            # compact cache 只在 worker 内恢复 mmap row，统一的输入 builder
            # 负责最后的 backend collate。
            items = [self._metadata(item) for item in raw_batch["items"]]
            current = self.input_builder.collate_encoded(
                raw_batch["current_enc_rows"],
                include_labels=True,
            )
            if self._is_future_rollout(items):
                online_tail = self._collate_rollout_online_tail(
                    items,
                    raw_batch["next_enc_rows"],
                )
            else:
                online_tail = self._collate_online_tail(
                    items,
                    raw_batch["next_enc_rows"],
                )
            cached_next = self._collate_next(
                items,
                raw_batch["next_enc_rows"],
            )
        else:
            items = [self._metadata(item) for item in raw_batch]
            current = self.input_builder.build(
                [item["messages"] for item in items if item["is_current_step"]],
                [() for item in items if item["is_current_step"]],
                include_labels=True,
            )
            online_tail = None
            cached_next = None

        if self._is_future_rollout(items):
            return self._prepare_future_rollout(
                items,
                current=current,
                online_tail=online_tail,
                cached_next=cached_next,
            )

        context_length = self._validate_window_items(items)
        if online_tail is None:
            tail_messages = [
                items[end - 1].get("next_messages")
                for end in range(context_length, len(items) + 1, context_length)
            ]
            if any(messages is None for messages in tail_messages):
                raise ValueError(
                    "SFT2 history window requires a real state after its final action"
                )
            online_tail = self.input_builder.build(
                [messages for messages in tail_messages if messages is not None],
                [() for _ in tail_messages],
                include_labels=False,
            )

        unique_keys, key_to_row = self._next_prompt_index(items)
        next_batch = self._next_batch(
            items,
            unique_keys,
            key_to_row,
            cached_next,
        )

        non_terminal = torch.ones(
            len(items),
            dtype=torch.bool,
            device=self.device,
        )
        next_indices = torch.tensor(
            [
                key_to_row[self._prompt_key(item["next_messages"])]
                for item in items
                if item["is_current_step"]
            ],
            dtype=torch.long,
            device=self.device,
        ).repeat_interleave(context_length)
        return SFT2Batch(
            transitions=TransitionBatch(
                current=current,
                next=next_batch,
                action_indices=torch.tensor(
                    [item["action_index"] for item in items],
                    dtype=torch.long,
                    device=self.device,
                ),
                value_targets=torch.tensor(
                    [item["action_value_target"] for item in items],
                    dtype=torch.float32,
                    device=self.device,
                ),
                next_indices=next_indices,
                non_terminal_mask=non_terminal,
                trajectory_steps=tuple(
                    self._trajectory_step(item) for item in items
                ),
            ),
            online_tail=online_tail,
            history_size=context_length,
            sample_weights=torch.tensor(
                [
                    item["loss_weight"]
                    for item in items
                    if item["is_current_step"]
                ],
                dtype=torch.float32,
                device=self.device,
            ),
            next_image_paths=tuple(
                str(item["next_image_path"])
                for item in items
                if item["is_current_step"]
            ),
        )

    def _prepare_future_rollout(
        self,
        items: Sequence[dict[str, Any]],
        *,
        current: BackboneBatch,
        online_tail: BackboneBatch | None,
        cached_next: CachedNextBatch | None,
    ) -> SFT2RolloutBatch:
        horizon = self._validate_rollout_items(items)
        if online_tail is None:
            first_next_messages = [
                items[start].get("next_messages")
                for start in range(0, len(items), horizon)
            ]
            if any(messages is None for messages in first_next_messages):
                raise ValueError("SFT2 rollout requires a real first next state")
            online_tail = self.input_builder.build(
                [messages for messages in first_next_messages if messages is not None],
                [() for _ in first_next_messages],
                include_labels=False,
            )

        unique_keys, key_to_row = self._next_prompt_index(items)
        next_batch = self._next_batch(
            items,
            unique_keys,
            key_to_row,
            cached_next,
        )
        next_indices = torch.tensor(
            [
                key_to_row[self._prompt_key(item["next_messages"])]
                for item in items
            ],
            dtype=torch.long,
            device=self.device,
        )
        return SFT2RolloutBatch(
            transitions=TransitionBatch(
                current=current,
                next=next_batch,
                action_indices=torch.tensor(
                    [item["action_index"] for item in items],
                    dtype=torch.long,
                    device=self.device,
                ),
                value_targets=torch.tensor(
                    [item["action_value_target"] for item in items],
                    dtype=torch.float32,
                    device=self.device,
                ),
                next_indices=next_indices,
                non_terminal_mask=torch.ones(
                    len(items),
                    dtype=torch.bool,
                    device=self.device,
                ),
                trajectory_steps=tuple(
                    self._trajectory_step(item) for item in items
                ),
            ),
            online_tail=online_tail,
            prediction_horizon=horizon,
            sample_weights=torch.tensor(
                [
                    item["loss_weight"]
                    for item in items
                    if item["rollout_position"] == 0
                ],
                dtype=torch.float32,
                device=self.device,
            ),
            next_image_paths=tuple(str(item["next_image_path"]) for item in items),
        )

    def _collate_online_tail(
        self,
        items: Sequence[dict[str, Any]],
        rows: Sequence[dict[str, torch.Tensor] | None],
    ) -> BackboneBatch:
        """只合并每个窗口的最后一个真实 next state，供在线 SIGReg 编码。"""

        context_length = self._validate_window_items(items)
        tail_rows = [
            rows[end - 1]
            for end in range(context_length, len(rows) + 1, context_length)
        ]
        if any(row is None for row in tail_rows):
            raise ValueError(
                "SFT2 cached history window is missing its final next-state encoding"
            )
        return self.input_builder.collate_encoded(
            [row for row in tail_rows if row is not None],
            include_labels=False,
        )

    def _collate_rollout_online_tail(
        self,
        items: Sequence[dict[str, Any]],
        rows: Sequence[dict[str, torch.Tensor] | None],
    ) -> BackboneBatch:
        """Collate the real ``s_{t+1}`` used by the unchanged SIGReg stage."""

        horizon = self._validate_rollout_items(items)
        first_rows = [rows[start] for start in range(0, len(rows), horizon)]
        if any(row is None for row in first_rows):
            raise ValueError("SFT2 cached rollout is missing its first next state")
        return self.input_builder.collate_encoded(
            [row for row in first_rows if row is not None],
            include_labels=False,
        )

    def _validate_rollout_items(
        self,
        items: Sequence[dict[str, Any]],
    ) -> int:
        if not items:
            raise ValueError("SFT2 rollout rows must not be empty")
        horizons = {int(item["prediction_horizon"]) for item in items}
        if horizons != {self.prediction_horizon}:
            raise ValueError(
                "SFT2 rollout horizon does not match assembler: "
                f"batch={sorted(horizons)}, configured={self.prediction_horizon}"
            )
        horizon = horizons.pop()
        if len(items) % horizon != 0:
            raise ValueError(
                f"SFT2 rows do not contain complete T={horizon} rollout windows"
            )
        for start in range(0, len(items), horizon):
            window = items[start : start + horizon]
            record_ids = {item["record_id"] for item in window}
            steps = [item["step_index"] for item in window]
            positions = [item["rollout_position"] for item in window]
            coordinates = [
                (item["record_id"], item["step_index"], item["rollout_position"])
                for item in window
            ]
            if (
                len(record_ids) != 1
                or any(
                    right != left + 1
                    for left, right in zip(steps, steps[1:])
                )
                or positions != list(range(horizon))
            ):
                raise ValueError(
                    "SFT2 rollout must use consecutive recorded transitions in "
                    f"order, got {coordinates}"
                )
            if [item["is_current_step"] for item in window] != [
                True,
                *([False] * (horizon - 1)),
            ]:
                raise ValueError("SFT2 rollout must mark only its first row current")
            if any(item.get("next_messages") is None for item in window):
                raise ValueError("SFT2 rollout requires all T real next states")
            if not all(item["needs_next_state"] for item in window):
                raise ValueError("SFT2 rollout must encode every next-state target")
        return horizon

    @staticmethod
    def _is_future_rollout(items: Sequence[dict[str, Any]]) -> bool:
        return bool(items and items[0].get("prediction_horizon") is not None)

    def _validate_window_items(
        self,
        items: Sequence[dict[str, Any]],
    ) -> int:
        """在 processor 调用前检查扁平行能否还原为完整连续窗口。"""

        if not items:
            raise ValueError("SFT2 rows must not be empty")
        context_lengths = {int(item["context_length"]) for item in items}
        if len(context_lengths) != 1:
            raise ValueError("SFT2 batch must contain one context length")
        context_length = context_lengths.pop()
        if not 1 <= context_length <= self.history_size:
            raise ValueError(
                "SFT2 context length must be in [1, history_size], "
                f"got {context_length} for history_size={self.history_size}"
            )
        if len(items) % context_length != 0:
            raise ValueError(
                "SFT2 rows must contain complete context windows: "
                f"rows={len(items)}, context_length={context_length}"
            )
        for start in range(0, len(items), context_length):
            window = items[start : start + context_length]
            record_ids = {item["record_id"] for item in window}
            steps = [item["step_index"] for item in window]
            if len(record_ids) != 1 or any(
                right != left + 1
                for left, right in zip(steps, steps[1:])
            ):
                raise ValueError(
                    "SFT2 history window must contain consecutive steps from one "
                    "trajectory, got "
                    f"{[(item['record_id'], item['step_index']) for item in window]}"
                )
            if window[-1].get("next_messages") is None:
                raise ValueError(
                    "SFT2 current step requires a real next state"
                )
            if [item["is_current_step"] for item in window] != [
                *([False] * (context_length - 1)),
                True,
            ]:
                raise ValueError("SFT2 context must mark only its final row current")
        return context_length

    def _collate_next(
        self,
        items: Sequence[dict[str, Any]],
        rows: Sequence[dict[str, torch.Tensor] | None],
    ) -> CachedNextBatch | None:
        unique_rows: list[dict[str, torch.Tensor]] = []
        unique_keys: list[str] = []
        seen: set[str] = set()
        for item, row in zip(items, rows, strict=True):
            if not item["needs_next_state"]:
                continue
            messages = item.get("next_messages")
            if messages is None or row is None:
                continue
            key = self._prompt_key(messages)
            if key in seen:
                continue
            seen.add(key)
            unique_keys.append(key)
            unique_rows.append(row)
        if not unique_rows:
            return None
        return CachedNextBatch(
            keys=tuple(unique_keys),
            batch=self.input_builder.collate_encoded(
                unique_rows,
                include_labels=False,
            ),
        )

    def _next_batch(
        self,
        items: Sequence[dict[str, Any]],
        unique_keys: Sequence[str],
        key_to_row: dict[str, int],
        cached: CachedNextBatch | dict[str, Any] | None,
    ) -> BackboneBatch:
        if cached is not None:
            if isinstance(cached, dict):
                keys = tuple(cached.get("keys", ()))
                batch = cached.get("batch")
                if not isinstance(batch, BackboneBatch):
                    encoding = cached.get("encoding", cached.get("enc"))
                    if not isinstance(encoding, dict):
                        raise ValueError("cached next-state batch is missing encoding")
                    batch = BackboneBatch(encoding)
                cached = CachedNextBatch(keys=keys, batch=batch)
            if cached.keys != tuple(unique_keys):
                raise ValueError(
                    "cached next-state order does not match transition prompts"
                )
            return cached.batch

        unique_messages: list[list[dict[str, Any]] | None] = [None] * len(unique_keys)
        for item in items:
            if not item["needs_next_state"]:
                continue
            messages = item.get("next_messages")
            if messages is not None:
                unique_messages[key_to_row[self._prompt_key(messages)]] = messages
        return self.input_builder.build(
            [messages for messages in unique_messages if messages is not None],
            [() for _ in unique_keys],
            include_labels=False,
        )

    def _prompt_key(self, messages: Sequence[dict[str, Any]]) -> str:
        return self.input_builder.cache_key(messages, ())

    def _next_prompt_index(
        self,
        items: Sequence[dict[str, Any]],
    ) -> tuple[list[str], dict[str, int]]:
        unique_keys: list[str] = []
        key_to_row: dict[str, int] = {}
        for item in items:
            if not item["needs_next_state"]:
                continue
            messages = item.get("next_messages")
            if messages is None:
                continue
            key = self._prompt_key(messages)
            if key not in key_to_row:
                key_to_row[key] = len(unique_keys)
                unique_keys.append(key)
        return unique_keys, key_to_row

    def _metadata(self, item: dict[str, Any]) -> dict[str, Any]:
        current_messages = item.get("messages")
        if not isinstance(current_messages, list):
            raise ValueError("transition is missing current Agent messages")
        next_messages = item.get("next_messages")
        if next_messages is not None and not isinstance(next_messages, list):
            raise ValueError("next_messages must be a list or None")
        return {
            "id": str(item.get("id", "")),
            "record_id": str(item.get("record_id", "")),
            "step_index": int(item.get("step_index", 0)),
            "action_index": int(item["action_index"]),
            "action_value_target": float(item["action_value_target"]),
            "success": bool(item.get("success", False)),
            "messages": current_messages,
            "next_messages": next_messages,
            "context_length": int(
                self.history_size
                if item.get("context_length") is None
                else item["context_length"]
            ),
            "is_current_step": bool(
                True
                if item.get("is_current_step") is None
                else item["is_current_step"]
            ),
            "prediction_horizon": (
                None
                if item.get("prediction_horizon") is None
                else int(item["prediction_horizon"])
            ),
            "rollout_position": (
                None
                if item.get("rollout_position") is None
                else int(item["rollout_position"])
            ),
            "needs_next_state": bool(
                item.get(
                    "needs_next_state",
                    True
                    if item.get("is_current_step") is None
                    else item["is_current_step"],
                )
            ),
            "loss_weight": float(item.get("loss_weight", 1.0)),
            "next_image_path": str(item.get("next_image_path", "")),
        }

    @staticmethod
    def _trajectory_step(item: dict[str, Any]) -> tuple[str, int]:
        record_id = item["record_id"]
        if not record_id and ":" in item["id"]:
            record_id = item["id"].split(":", 1)[0]
        return record_id, item["step_index"]


__all__ = [
    "CachedNextBatch",
    "SFT2Batch",
    "SFT2BatchAssembler",
    "SFT2BatchBuilder",
    "SFT2RolloutBatch",
]
