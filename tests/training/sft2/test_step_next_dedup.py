from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from nimloth.training.sft2.step import _forward_next_latents, wm_eligible_indices


def test_wm_eligible_indices_skips_terminal_steps() -> None:
    items = [
        {"next_messages": [{"role": "user", "content": "a"}]},
        {"next_messages": None},
    ]
    assert wm_eligible_indices(items) == [0]


def test_forward_next_latents_dedups_identical_next_prefixes() -> None:
    shared_next = [{"role": "user", "content": "shared"}]
    items = [
        {"next_messages": shared_next},
        {"next_messages": shared_next},
    ]
    indices = wm_eligible_indices(items)
    model = MagicMock()
    processor = MagicMock()
    token_id_map = {"latent_state": 1}
    device = torch.device("cpu")

    captured_batch_sizes: list[int] = []

    def fake_build_qwen_batch(next_items, _processor, _max_length):
        captured_batch_sizes.append(len(next_items))
        batch_size = len(next_items)
        return {
            "input_ids": torch.zeros((batch_size, 4), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 4), dtype=torch.long),
        }

    def fake_extract(_model, enc, _token_id_map, _device):
        batch_size = enc["input_ids"].shape[0]
        return torch.arange(batch_size, dtype=torch.float32).view(batch_size, 1), None

    with (
        patch("nimloth.training.sft2.step.build_qwen_batch", side_effect=fake_build_qwen_batch),
        patch("nimloth.training.sft2.step.extract_qwen_latents", side_effect=fake_extract),
    ):
        next_latent = _forward_next_latents(
            model,
            items,
            indices,
            processor,
            token_id_map,
            device,
            max_length=32,
            vision_ema=None,
            next_enc_rows=None,
            pad_token_id=None,
        )

    assert captured_batch_sizes == [1]
    assert next_latent.shape == (2, 1)
    assert torch.allclose(next_latent[0], next_latent[1])
