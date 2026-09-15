"""Complete-trajectory data items, reusing immutable Qwen preprocess artifacts."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Sequence

import torch
from torch.utils.data import Dataset

from nimloth.backbone.qwen25vl.batch import encode_qwen_item
from nimloth.agent import bind_image_placeholders
from nimloth.latent import find_last_latent_state_block, latent_state_tokens
from nimloth.rollout.transitions import TransitionSample, bind_transition_prompt
from nimloth.util.cache import CachedTransitionDataset, CompactCachedTransitionCollator


@dataclass(frozen=True)
class TrajectoryIndex:
    index: int
    loss_weight: float = 1.0


@dataclass(frozen=True)
class TrajectoryItem:
    samples: tuple[TransitionSample, ...]
    encodings: tuple[dict[str, torch.Tensor], ...] | None
    loss_weight: float


@dataclass(frozen=True)
class EncodedTrajectory:
    samples: tuple[TransitionSample, ...]
    encoding: dict[str, torch.Tensor]
    state_positions: tuple[tuple[int, ...], ...]
    lm_labels: tuple[torch.Tensor, ...]
    loss_weight: float

    def pin_memory(self):
        return replace(self,
            encoding={key: value.pin_memory() for key, value in self.encoding.items()},
            lm_labels=tuple(value.pin_memory() for value in self.lm_labels))


class TrajectoryDataset(Dataset):
    """Each item owns every eligible T-step window of one unsplit trajectory."""
    def __init__(self, samples: Sequence[TransitionSample], *, prediction_horizon: int,
                 cached: CachedTransitionDataset | None = None):
        if prediction_horizon < 1:
            raise ValueError("prediction_horizon must be positive")
        groups: dict[str, list[int]] = OrderedDict()
        for index, sample in enumerate(samples):
            groups.setdefault(sample.record_id, []).append(index)
        self.samples = samples
        self.cached = cached
        self.prediction_horizon = prediction_horizon
        self.groups = []
        self.omitted_short_trajectories = 0
        for record_id, indices in groups.items():
            indices.sort(key=lambda index: samples[index].step_index)
            record = [samples[index] for index in indices]
            if [sample.step_index for sample in record] != list(range(len(record))):
                raise ValueError(f"trajectory {record_id!r} has missing or duplicate steps")
            if any(not isinstance(sample.success, bool) for sample in record):
                raise ValueError(f"trajectory {record_id!r} success must be boolean")
            if len({sample.success for sample in record}) != 1:
                raise ValueError(f"trajectory {record_id!r} has inconsistent success")
            if any(sample.next_prefix_messages is None or sample.next_prefix_image_paths is None for sample in record):
                raise ValueError(f"trajectory {record_id!r} is missing a real next-state prefix")
            if len(record) < prediction_horizon:
                self.omitted_short_trajectories += 1
                continue
            self.groups.append(tuple(indices))
        self.window_counts = tuple(len(group) - prediction_horizon + 1 for group in self.groups)
        self.window_count = sum(self.window_counts)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index: int | TrajectoryIndex) -> TrajectoryItem:
        annotated = index if isinstance(index, TrajectoryIndex) else TrajectoryIndex(index)
        if annotated.loss_weight not in (0.0, 1.0):
            raise ValueError("trajectory padding weight must be zero or one")
        indices = self.groups[annotated.index]
        samples = tuple(self.samples[index] for index in indices)
        encodings = None
        if self.cached is not None:
            entries = [self.cached[index] for index in indices]
            terminal = entries[-1]["next_enc"]
            if terminal is None:
                raise ValueError("trajectory preprocess cache is missing the terminal prefix")
            encodings = tuple(entry["current_enc"] for entry in entries) + (terminal,)
        return TrajectoryItem(samples, encodings, annotated.loss_weight)


def _require_prefix(prefix: dict, full: dict) -> None:
    for field in ("input_ids", "attention_mask", "image_grid_thw", "image_indices", "pixel_values"):
        left, right = prefix.get(field), full.get(field)
        if left is None:
            continue
        if (right is None or left.shape[1:] != right.shape[1:] or len(left) > len(right)
                or not torch.equal(left, right[:len(left)])):
            raise ValueError(f"trajectory cached {field} is not an exact full-input prefix")


class TrajectoryCollator:
    """Verify original prefix labels, but materialize each complete image history once."""
    def __init__(self, input_builder, *, prediction_horizon: int,
                 cache_collator: CompactCachedTransitionCollator | None = None):
        self.input_builder = input_builder
        self.prediction_horizon = prediction_horizon
        self.cache_collator = cache_collator

    def __call__(self, items: list[TrajectoryItem]) -> list[EncodedTrajectory]:
        result = []
        builder = self.input_builder
        token_map = {token: builder.processor.tokenizer.convert_tokens_to_ids(token)
                     for token in latent_state_tokens(builder.latent_token_count)}
        for item in items:
            encodings = item.encodings
            if encodings is None:
                messages = [bind_transition_prompt(sample) for sample in item.samples]
                messages.append(bind_image_placeholders(item.samples[-1].next_prefix_messages,
                                                       item.samples[-1].next_prefix_image_paths))
                encodings = tuple(encode_qwen_item(message, builder.processor, builder.max_length,
                    include_labels=(index < len(item.samples)), latent_token_count=builder.latent_token_count,
                    mask_latent_query_labels=builder.mask_latent_query_labels)
                    for index, message in enumerate(messages))
            full = encodings[-1]
            positions = []
            for prefix in encodings:
                _require_prefix(prefix, full)
                positions.append(tuple(find_last_latent_state_block(prefix["input_ids"], token_map,
                                                       latent_token_count=builder.latent_token_count)))
            window_count = len(item.samples) - self.prediction_horizon + 1
            labels = []
            for prefix in encodings[:window_count]:
                label = prefix.get("labels")
                if label is None or label.shape != prefix["input_ids"].shape:
                    raise ValueError("trajectory current prefix lacks original LM labels")
                if not torch.any(label[1:] != -100):
                    raise ValueError("trajectory LM answer has no supervised tokens")
                valid = label != -100
                if not torch.equal(label[valid], prefix["input_ids"][valid]):
                    raise ValueError("trajectory LM labels differ from original input tokens")
                labels.append(label)
            if self.cache_collator is not None:
                full = self.cache_collator.materialize_encoding(full)
            elif "image_indices" in full:
                raise ValueError("compact trajectory images require a cache materializer")
            full = {key: value for key, value in full.items() if key != "labels"}
            result.append(EncodedTrajectory(item.samples, full, tuple(positions), tuple(labels), item.loss_weight))
        return result
