from __future__ import annotations

import torch

from nimloth.training.common.qwen_batch import build_qwen_batch, encode_qwen_item
from nimloth.training.sft2.preprocess_cache import (
    cache_fingerprint,
    collate_cached_transition_batch,
    encode_transition_item,
)


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
        assert tokenize is False
        rendered = ""
        for msg in messages:
            if msg["role"] == "assistant":
                rendered += "<assistant>" + str(msg["content"])
            else:
                rendered += f"<{msg['role']}>" + str(msg["content"])
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    def __call__(self, *, text, images, padding, truncation, max_length, return_tensors):
        del images, truncation, max_length
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


def test_encode_qwen_item_matches_build_qwen_batch_single_item() -> None:
    processor = FakeProcessor()
    messages = [
        {"role": "user", "content": "obs"},
        {"role": "assistant", "content": "act"},
    ]
    online = build_qwen_batch([{"messages": messages}], processor, max_length=64)
    cached = encode_qwen_item(messages, processor, max_length=64)
    assert torch.equal(online["input_ids"][0], cached["input_ids"])
    assert torch.equal(online["attention_mask"][0], cached["attention_mask"])
    assert torch.equal(online["labels"][0], cached["labels"])


def test_encode_transition_item_roundtrip_collate() -> None:
    processor = FakeProcessor()
    item = {
        "id": "rec:0",
        "messages": [
            {"role": "user", "content": "obs0"},
            {"role": "assistant", "content": "act0"},
        ],
        "action_index": 2,
        "action_value_target": 0.5,
        "success": True,
        "next_messages": [
            {"role": "user", "content": "obs0"},
            {"role": "assistant", "content": "act0"},
            {"role": "user", "content": "obs1"},
            {"role": "assistant", "content": "act1"},
        ],
    }
    encoded = encode_transition_item(item, processor, max_length=128)
    encoded["next_messages"] = item.get("next_messages")
    batch = collate_cached_transition_batch([encoded], pad_token_id=0)
    online_current = build_qwen_batch([{"messages": item["messages"]}], processor, max_length=128)
    assert torch.equal(batch["current_enc"]["input_ids"][0], online_current["input_ids"][0])
    assert torch.equal(batch["current_enc"]["labels"][0], online_current["labels"][0])


def test_cache_fingerprint_changes_when_jsonl_changes(tmp_path) -> None:
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text('{"id":"a"}\n', encoding="utf-8")
    fp1 = cache_fingerprint(
        jsonl,
        max_length=100,
        max_pixels=1000,
        min_pixels=100,
        vocab_size=50000,
    )
    jsonl.write_text('{"id":"a"}\n{"id":"b"}\n', encoding="utf-8")
    fp2 = cache_fingerprint(
        jsonl,
        max_length=100,
        max_pixels=1000,
        min_pixels=100,
        vocab_size=50000,
    )
    assert fp1 != fp2
