from __future__ import annotations

import torch

from nimloth.backbone.qwen25vl.batch import build_qwen_batch, encode_qwen_item
from nimloth.training.sft2.data.cache import (
    COMPACT_CACHE_FORMAT,
    CachedTransitionDataset,
    CompactCachedTransitionCollator,
    cache_fingerprint,
    encode_transition_item,
)
from nimloth.training.sft2.data.batch import collate_cached_transition_batch
from nimloth.training.sft2.data.cache.encoding import _expand_qwen_image_tokens
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


def test_cache_fingerprint_changes_with_latent_token_count(tmp_path) -> None:
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text('{"id":"a"}\n', encoding="utf-8")

    fp1 = cache_fingerprint(
        jsonl,
        max_length=100,
        max_pixels=1000,
        min_pixels=100,
        vocab_size=50000,
        latent_token_count=1,
    )
    fp2 = cache_fingerprint(
        jsonl,
        max_length=100,
        max_pixels=1000,
        min_pixels=100,
        vocab_size=50000,
        latent_token_count=3,
    )

    assert fp1 != fp2


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


def test_cache_fingerprint_changes_with_compact_image_dtype(tmp_path) -> None:
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text('{"id":"a"}\n', encoding="utf-8")
    fp_bf16 = cache_fingerprint(
        jsonl,
        max_length=100,
        max_pixels=1000,
        min_pixels=100,
        vocab_size=50000,
        cache_format=COMPACT_CACHE_FORMAT,
        image_dtype="bfloat16",
    )
    fp_f32 = cache_fingerprint(
        jsonl,
        max_length=100,
        max_pixels=1000,
        min_pixels=100,
        vocab_size=50000,
        cache_format=COMPACT_CACHE_FORMAT,
        image_dtype="float32",
    )
    assert fp_bf16 != fp_f32


def test_expand_qwen_image_tokens_uses_cached_grids() -> None:
    class ImageProcessor:
        merge_size = 2

    class Processor:
        image_token = "<image_pad>"
        image_processor = ImageProcessor()

    grids = torch.tensor([[1, 4, 4], [1, 2, 2]])
    expanded = _expand_qwen_image_tokens("a<image_pad>b<image_pad>c", grids, Processor())
    assert expanded == "a" + "<image_pad>" * 4 + "b<image_pad>c"


def test_compact_cache_mmap_collator_reuses_next_row(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    (cache_dir / "images").mkdir(parents=True)
    (cache_dir / "transitions").mkdir(parents=True)
    (cache_dir / "manifest.json").write_text(
        '{"format":"dedup_sharded_v1","count":2,"transition_shard_size":2}',
        encoding="utf-8",
    )
    (cache_dir / "image_index.json").write_text(
        '{"format":"dedup_sharded_v1","images":['
        '{"path":"im0","shard":0,"index":0,"grid_thw":[1,1,2]},'
        '{"path":"im1","shard":0,"index":1,"grid_thw":[1,1,3]}]}',
        encoding="utf-8",
    )
    pixels = torch.arange(10, dtype=torch.float32).reshape(5, 2).to(torch.bfloat16)
    torch.save(
        {
            "pixel_values": pixels,
            "offsets": torch.tensor([0, 2, 5]),
            "image_grid_thw": torch.tensor([[1, 1, 2], [1, 1, 3]]),
        },
        cache_dir / "images" / "shard_00000.pt",
    )

    def enc(ids: list[int], image_indices: list[int], grids: list[list[int]]) -> dict:
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
            "labels": torch.tensor(ids),
            "image_grid_thw": torch.tensor(grids),
            "image_indices": torch.tensor(image_indices, dtype=torch.int32),
        }

    torch.save(
        {
            "entries": [
                {
                    "id": "rec:0",
                    "record_id": "rec",
                    "step_index": 0,
                    "action_index": 0,
                    "action_value_target": 1.0,
                    "success": True,
                    "current_enc": enc([1, 2], [0], [[1, 1, 2]]),
                },
                {
                    "id": "rec:1",
                    "record_id": "rec",
                    "step_index": 1,
                    "action_index": 1,
                    "action_value_target": 1.0,
                    "success": True,
                    "current_enc": enc([3, 4, 5], [0, 1], [[1, 1, 2], [1, 1, 3]]),
                },
            ]
        },
        cache_dir / "transitions" / "shard_00000.pt",
    )
    samples = [
        TransitionSample(
            record_id="rec",
            step_index=0,
            prefix_messages=[{"role": "assistant", "content": "a <image>"}],
            prefix_image_paths=["im0"],
            action_index=0,
            current_image_path="im0",
            next_image_path="im1",
            next_prefix_messages=[
                {"role": "assistant", "content": "b <image> <image>"}
            ],
            next_prefix_image_paths=["im0", "im1"],
        ),
        TransitionSample(
            record_id="rec",
            step_index=1,
            prefix_messages=[
                {"role": "assistant", "content": "b <image> <image>"}
            ],
            prefix_image_paths=["im0", "im1"],
            action_index=1,
            current_image_path="im1",
            next_image_path="im1",
        ),
    ]
    dataset = CachedTransitionDataset(cache_dir, samples, max_open_shards=1)
    collator = CompactCachedTransitionCollator(cache_dir, pad_token_id=0, max_open_shards=1)
    batch = collator([dataset[0], dataset[1]])

    current_pixels = batch["current_enc"]["pixel_values"]
    assert current_pixels.dtype == torch.bfloat16
    assert torch.equal(current_pixels, torch.cat([pixels[:2], pixels[:2], pixels[2:]], dim=0))
    next_bundle = batch["next_enc_bundle"]
    assert next_bundle is not None
    assert len(next_bundle["keys"]) == 1
    assert torch.equal(next_bundle["enc"]["pixel_values"], torch.cat([pixels[:2], pixels[2:]], dim=0))
