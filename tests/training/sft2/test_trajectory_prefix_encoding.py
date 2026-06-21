"""CPU tests for trajectory prefix encoding invariants."""

from __future__ import annotations

from nimloth.training.sft2.trajectory_once import (
    encode_full_trajectory,
    find_step_latent_indices,
    verify_prefix_tokenization,
)
from nimloth.wm.dataset import TransitionSample


class FakeTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        text,
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_offsets_mapping: bool,
        add_special_tokens: bool,
    ):
        del padding, truncation, max_length, add_special_tokens
        offsets = []
        pos = 0
        for ch in text:
            offsets.append((pos, pos + 1))
            pos += 1
        return {"offset_mapping": offsets}


class FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()

    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
        del add_generation_prompt
        assert tokenize is False
        rendered = ""
        for msg in messages:
            if msg["role"] == "assistant":
                rendered += "<assistant>" + str(msg["content"])
            else:
                rendered += f"<{msg['role']}>" + str(msg["content"])
        return rendered

    def __call__(self, *, text, images, padding, truncation, max_length, return_tensors):
        del images, truncation, max_length
        import torch

        batch = len(text)
        max_len = max(len(t) for t in text)
        input_ids = torch.zeros((batch, max_len), dtype=torch.long)
        attention_mask = torch.zeros((batch, max_len), dtype=torch.long)
        for row, t in enumerate(text):
            for col, ch in enumerate(t):
                input_ids[row, col] = ord(ch)
                attention_mask[row, col] = 1
        if padding:
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _two_step_samples() -> list[TransitionSample]:
    messages_step0 = [
        {"role": "user", "content": "u0"},
        {"role": "assistant", "content": "a0\x01"},
    ]
    messages_step1 = messages_step0 + [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1\x01"},
    ]
    return [
        TransitionSample(
            record_id="r",
            step_index=0,
            prefix_messages=messages_step0,
            prefix_image_paths=[],
            action_index=0,
            current_image_path="a.png",
            next_image_path="b.png",
        ),
        TransitionSample(
            record_id="r",
            step_index=1,
            prefix_messages=messages_step1,
            prefix_image_paths=[],
            action_index=1,
            current_image_path="b.png",
            next_image_path="c.png",
        ),
    ]


def test_two_step_prefix_tokenization_is_stable() -> None:
    processor = FakeProcessor()
    steps = _two_step_samples()
    full_enc, full_text = encode_full_trajectory(steps, processor, max_length=512)
    verify_prefix_tokenization(
        steps, full_enc, processor, max_length=512, full_text=full_text, token_id_map=token_id_map
    )
    token_id_map = {"<|latent_state|>": ord("\x01")}
    indices = find_step_latent_indices(steps, full_enc, processor, token_id_map, max_length=512)
    assert indices[0] == full_text.index("\x01")
    assert indices[1] == full_text.rindex("\x01")
