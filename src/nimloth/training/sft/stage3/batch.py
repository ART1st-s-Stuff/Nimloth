"""Trajectory-native Stage3 model inputs and window supervision indices."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from nimloth.backbone import BackboneBatch, BackboneInputBuilder
from nimloth.training.sft.stage3.data.trajectory import EncodedTrajectory


@dataclass(frozen=True)
class Stage3TrajectoryBatch:
    inputs: BackboneBatch
    state_keys: tuple[tuple[str, int], ...]
    trajectory_ids: tuple[str, ...]
    state_offsets: tuple[int, ...]
    window_offsets: tuple[int, ...]
    current_indices: torch.Tensor
    next_indices: torch.Tensor
    action_sequences: torch.Tensor
    value_targets: torch.Tensor
    sample_weights: torch.Tensor
    lm_weights: torch.Tensor
    outcome_targets: torch.Tensor
    outcome_mask: torch.Tensor
    current_image_paths: tuple[str, ...] = ()
    next_image_paths: tuple[str, ...] = ()
    dino_grid_target: torch.Tensor | None = None
    current_dino_target: torch.Tensor | None = None

    def __post_init__(self):
        windows = self.current_indices.numel()
        if self.action_sequences.ndim != 2 or self.action_sequences.shape[0] != windows:
            raise ValueError("trajectory actions must have shape (windows,horizon)")
        if windows < 1 or self.prediction_horizon < 1:
            raise ValueError("trajectory batch must contain complete prediction windows")
        if self.inputs.tensors["input_ids"].shape[0] != self.trajectory_count:
            raise ValueError("one full input row is required per trajectory")
        positions = self.inputs.tensors["state_positions"]
        if positions.ndim != 3 or positions.shape[0] != len(self.state_keys) or positions.shape[-1] != 2:
            raise ValueError("trajectory Query positions must align with every state")
        labels = self.inputs.tensors["lm_labels"]
        if labels.shape != (windows, self.inputs.tensors["input_ids"].shape[1]):
            raise ValueError("trajectory LM labels must align with all eligible windows")
        expected = self.action_sequences.shape
        for value in (self.next_indices, self.value_targets, self.outcome_targets, self.outcome_mask):
            if value.shape != expected:
                raise ValueError("trajectory supervision arrays disagree with window shape")
        for value in (self.sample_weights, self.lm_weights):
            if value.shape != (windows,) or not torch.all((value == 0) | (value == 1)):
                raise ValueError("trajectory window weights must be binary vectors")
        if torch.any(self.lm_weights > self.sample_weights):
            raise ValueError("padding windows cannot carry language supervision")
        if self.outcome_mask.dtype != torch.bool or torch.any(self.outcome_mask & ~self.sample_weights.bool()[:, None]):
            raise ValueError("invalid trajectory outcome mask")
        if not torch.all((self.outcome_targets == 0) | (self.outcome_targets == 1)):
            raise ValueError("trajectory outcomes must be binary")
        if len(self.state_offsets) != self.trajectory_count + 1 or len(self.window_offsets) != self.trajectory_count + 1:
            raise ValueError("trajectory offsets must contain one boundary per record")
        if self.state_offsets[0] != 0 or self.state_offsets[-1] != len(self.state_keys):
            raise ValueError("trajectory state offsets do not cover all states")
        if self.window_offsets[0] != 0 or self.window_offsets[-1] != windows:
            raise ValueError("trajectory window offsets do not cover all windows")
        for record, left, right, start, end in zip(self.trajectory_ids, self.state_offsets[:-1],
                self.state_offsets[1:], self.window_offsets[:-1], self.window_offsets[1:], strict=True):
            if right <= left or end <= start:
                raise ValueError("empty trajectory in model batch")
            if tuple(self.state_keys[left:right]) != tuple((record, step) for step in range(right - left)):
                raise ValueError("trajectory states are not contiguous within one record")
            expected_current = torch.arange(left, right - self.prediction_horizon,
                                            device=self.current_indices.device)
            if not torch.equal(self.current_indices[start:end], expected_current):
                raise ValueError("trajectory windows do not cover each eligible start exactly once")
            expected_next = expected_current[:, None] + torch.arange(1, self.prediction_horizon + 1,
                                                                      device=self.next_indices.device)
            if not torch.equal(self.next_indices[start:end], expected_next):
                raise ValueError("trajectory successor indices cross a record or skip a step")

    @property
    def batch_size(self):
        """Number of supervised windows; input batch size is trajectory_count."""
        return self.current_indices.numel()

    @property
    def trajectory_count(self):
        return len(self.trajectory_ids)

    @property
    def prediction_horizon(self):
        return self.action_sequences.shape[1]

    @property
    def current_keys(self):
        return tuple(self.state_keys[index] for index in self.current_indices.tolist())

    @property
    def is_padding(self):
        return not bool(self.sample_weights.any())

    @property
    def window_trajectory_indices(self):
        counts = torch.tensor([right - left for left, right in zip(self.window_offsets[:-1],
                              self.window_offsets[1:], strict=True)], device=self.current_indices.device)
        return torch.arange(self.trajectory_count, device=self.current_indices.device).repeat_interleave(counts)


class SFT2BatchBuilder(Protocol):
    """Shared loader/loop interface; Stage3 now prepares complete trajectories."""
    input_builder: BackboneInputBuilder
    device: torch.device
    @property
    def processor(self): ...
    def supervision_counts(self, raw_batch) -> tuple[int, int]: ...
    def outcome_count(self, raw_batch) -> int: ...
    def prepare(self, raw_batch) -> Stage3TrajectoryBatch: ...


class Stage3BatchAssembler:
    def __init__(self, *, input_builder: BackboneInputBuilder, device: torch.device,
                 prediction_horizon: int):
        if prediction_horizon < 1:
            raise ValueError("prediction_horizon must be positive")
        self.input_builder = input_builder
        self.device = device
        self.prediction_horizon = prediction_horizon

    @property
    def processor(self):
        return self.input_builder.processor

    def supervision_counts(self, raw_batch: list[EncodedTrajectory]):
        all_count = lm_count = 0
        for item in raw_batch:
            count = len(item.lm_labels) if item.loss_weight else 0
            all_count += count
            lm_count += count if item.samples[0].success else 0
        return all_count, lm_count

    def outcome_count(self, raw_batch: list[EncodedTrajectory]):
        return sum(sample.action_success is not None for item in raw_batch if item.loss_weight
                   for start in range(len(item.lm_labels))
                   for sample in item.samples[start:start + self.prediction_horizon])

    def prepare(self, raw_batch: list[EncodedTrajectory]) -> Stage3TrajectoryBatch:
        if not raw_batch:
            raise ValueError("trajectory batch must not be empty")
        packed = self.input_builder.collate_encoded([item.encoding for item in raw_batch], include_labels=False)
        sequence_length = packed.tensors["input_ids"].shape[1]
        state_keys, records, state_offsets, window_offsets = [], [], [0], [0]
        state_positions, current_indices, next_indices, actions, returns = [], [], [], [], []
        weights, lm_weights, outcomes, outcome_masks = [], [], [], []
        labels, sources, current_images, next_images = [], [], [], []
        for row, item in enumerate(raw_batch):
            samples = item.samples
            record = samples[0].record_id
            records.append(record)
            base = len(state_keys)
            state_keys.extend((record, step) for step in range(len(samples) + 1))
            state_positions.extend([[row, index] for index in block] for block in item.state_positions)
            state_offsets.append(len(state_keys))
            for start, label in enumerate(item.lm_labels):
                window = samples[start:start + self.prediction_horizon]
                if len(window) != self.prediction_horizon:
                    raise ValueError("incomplete trajectory prediction window")
                current_indices.append(base + start)
                next_indices.append(list(range(base + start + 1, base + start + 1 + self.prediction_horizon)))
                actions.append([sample.action_index for sample in window])
                returns.append([sample.action_value_target for sample in window])
                outcomes.append([float(sample.action_success) if sample.action_success is not None else 0.0 for sample in window])
                outcome_masks.append([sample.action_success is not None and bool(item.loss_weight) for sample in window])
                weights.append(item.loss_weight)
                lm_weights.append(item.loss_weight * float(samples[0].success))
                padded = torch.full((sequence_length,), -100, dtype=torch.long)
                padded[:len(label)] = label
                labels.append(padded)
                sources.append(row)
                current_images.append(window[0].current_image_path)
                next_images.extend(sample.next_image_path for sample in window)
            window_offsets.append(len(current_indices))
        inputs = dict(packed.tensors)
        inputs.update(state_positions=torch.tensor(state_positions, dtype=torch.long),
                      lm_labels=torch.stack(labels), lm_source_rows=torch.tensor(sources),
                      lm_row_weights=torch.tensor(lm_weights, dtype=torch.float32))
        def tensor(values, dtype):
            return torch.tensor(values, dtype=dtype, device=self.device)
        return Stage3TrajectoryBatch(BackboneBatch(inputs), tuple(state_keys), tuple(records),
            tuple(state_offsets), tuple(window_offsets), tensor(current_indices, torch.long),
            tensor(next_indices, torch.long), tensor(actions, torch.long), tensor(returns, torch.float32),
            tensor(weights, torch.float32), tensor(lm_weights, torch.float32),
            tensor(outcomes, torch.float32), tensor(outcome_masks, torch.bool),
            tuple(current_images), tuple(next_images))
