"""CPU tests for trajectory_once label/span helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from nimloth.training.sft2.trajectory_once import labels_for_trajectory_steps, supervised_token_count

_spec = importlib.util.spec_from_file_location(
    "test_preprocess_cache",
    Path(__file__).with_name("test_preprocess_cache.py"),
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
FakeProcessor = _mod.FakeProcessor


def test_labels_for_trajectory_steps_marks_each_assistant_span() -> None:
    processor = FakeProcessor()
    steps_messages = [
        [{"role": "user", "content": "u0"}, {"role": "assistant", "content": "a0"}],
        [
            {"role": "user", "content": "u0"},
            {"role": "assistant", "content": "a0"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ],
    ]
    from nimloth.wm.dataset import TransitionSample

    steps = [
        TransitionSample(
            record_id="r",
            step_index=0,
            prefix_messages=steps_messages[0],
            prefix_image_paths=[],
            action_index=0,
            current_image_path="a.png",
            next_image_path="b.png",
        ),
        TransitionSample(
            record_id="r",
            step_index=1,
            prefix_messages=steps_messages[1],
            prefix_image_paths=[],
            action_index=1,
            current_image_path="b.png",
            next_image_path="c.png",
        ),
    ]
    cache = processor.apply_chat_template(steps_messages[1], tokenize=False, add_generation_prompt=False)
    input_ids = torch.tensor([ord(c) for c in cache], dtype=torch.long)
    labels = labels_for_trajectory_steps(input_ids, cache, steps, processor, max_length=512)
    assert supervised_token_count(labels) > 0
    assert int((labels != -100).sum()) >= 2
