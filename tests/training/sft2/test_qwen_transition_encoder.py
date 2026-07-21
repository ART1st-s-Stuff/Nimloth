from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from nimloth.backbone.qwen25vl.batch import message_cache_key
from nimloth.backbone.qwen25vl.transition import (
    CachedQwenNextBatch,
    QwenTransitionEncoder,
    QwenTransitionMessages,
)


def _encoder() -> QwenTransitionEncoder:
    return QwenTransitionEncoder(
        processor=MagicMock(),
        token_id_map={"latent_state": 1},
        device=torch.device("cpu"),
        max_length=32,
        pad_token_id=0,
    )


def test_encode_next_deduplicates_identical_prompts() -> None:
    shared_next = [{"role": "user", "content": "shared"}]
    transitions = [
        QwenTransitionMessages(current=[], next=shared_next),
        QwenTransitionMessages(current=[], next=shared_next),
    ]
    captured_batch_sizes: list[int] = []

    def fake_build_qwen_batch(next_items, _processor, _max_length, **_kwargs):
        captured_batch_sizes.append(len(next_items))
        batch_size = len(next_items)
        return {
            "input_ids": torch.zeros((batch_size, 4), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 4), dtype=torch.long),
        }

    def fake_extract(_model, encoding, _token_id_map, _device, **_kwargs):
        batch_size = encoding["input_ids"].shape[0]
        return torch.arange(batch_size, dtype=torch.float32).view(batch_size, 1), None

    with (
        patch(
            "nimloth.backbone.qwen25vl.transition.build_qwen_batch",
            side_effect=fake_build_qwen_batch,
        ),
        patch(
            "nimloth.backbone.qwen25vl.transition.extract_qwen_latents",
            side_effect=fake_extract,
        ),
    ):
        next_latent = _encoder().encode_next(
            MagicMock(),
            transitions,
            [0, 1],
            cached=None,
            use_vision_ema=False,
        )

    assert captured_batch_sizes == [1]
    assert next_latent.shape == (2, 1)
    assert torch.allclose(next_latent[0], next_latent[1])


def test_encode_next_uses_worker_prebatched_cache() -> None:
    next_messages = [{"role": "user", "content": "next"}]
    cached = CachedQwenNextBatch(
        keys=(message_cache_key(next_messages),),
        encoding={
            "input_ids": torch.ones((1, 3), dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        },
    )

    def fake_extract(_model, encoding, _token_id_map, _device, **_kwargs):
        assert encoding["input_ids"].shape == (1, 3)
        return torch.tensor([[7.0]]), None

    with (
        patch("nimloth.backbone.qwen25vl.transition.build_qwen_batch") as build,
        patch(
            "nimloth.backbone.qwen25vl.transition.extract_qwen_latents",
            side_effect=fake_extract,
        ),
    ):
        next_latent = _encoder().encode_next(
            MagicMock(),
            [QwenTransitionMessages(current=[], next=next_messages)],
            [0],
            cached=cached,
            use_vision_ema=False,
        )

    build.assert_not_called()
    assert torch.equal(next_latent, torch.tensor([[7.0]]))
