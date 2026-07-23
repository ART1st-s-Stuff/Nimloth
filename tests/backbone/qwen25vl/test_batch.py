from __future__ import annotations

import torch

from nimloth.backbone.qwen25vl.batch import assistant_char_spans, split_qwen_batch_rows


class FakeProcessor:
    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
        assert tokenize is False
        rendered = ""
        for msg in messages:
            if msg["role"] == "assistant":
                rendered += "<assistant>" + msg["content"]
            else:
                rendered += f"<{msg['role']}>" + msg["content"]
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


def test_assistant_char_spans_only_returns_final_assistant_span() -> None:
    messages = [
        {"role": "user", "content": "obs0"},
        {"role": "assistant", "content": "act0"},
        {"role": "user", "content": "obs1"},
        {"role": "assistant", "content": "act1"},
    ]

    spans = assistant_char_spans(messages, FakeProcessor())

    assert len(spans) == 1
    start, end = spans[0]
    rendered = FakeProcessor().apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    assert rendered[start:end] == "act1"


def test_assistant_char_spans_empty_without_assistant() -> None:
    assert assistant_char_spans([{"role": "user", "content": "obs"}], FakeProcessor()) == []


def test_split_qwen_batch_rows_keeps_each_rows_images_and_pixels() -> None:
    # merge_size=2: grids produce 2, 1, and 3 text image tokens respectively.
    grids = torch.tensor([[1, 2, 4], [1, 2, 2], [1, 3, 4]])
    pixels = torch.arange(24).unsqueeze(1)
    batch = {
        "input_ids": torch.tensor(
            [
                [10, 99, 99, 0],
                [20, 99, 0, 0],
                [30, 99, 99, 99],
            ]
        ),
        "attention_mask": torch.ones(3, 4),
        "labels": torch.tensor([[-100, 1, 2, -100], [-100, 3, -100, -100], [-100, 4, 5, 6]]),
        "image_grid_thw": grids,
        "pixel_values": pixels,
    }

    chunks = split_qwen_batch_rows(
        batch,
        max_rows=1,
        image_token_id=99,
        spatial_merge_size=2,
    )

    assert [chunk["input_ids"][0, 0].item() for chunk in chunks] == [10, 20, 30]
    assert [chunk["image_grid_thw"].shape[0] for chunk in chunks] == [1, 1, 1]
    assert [chunk["pixel_values"].shape[0] for chunk in chunks] == [8, 4, 12]
    assert torch.equal(torch.cat([chunk["pixel_values"] for chunk in chunks]), pixels)
