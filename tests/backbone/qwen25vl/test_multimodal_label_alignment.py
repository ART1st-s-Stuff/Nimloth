"""Real Qwen processor/tokenizer image expansion and final-answer label alignment."""
from __future__ import annotations

import copy

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import Qwen2TokenizerFast, Qwen2VLImageProcessor, Qwen2_5_VLProcessor

from nimloth.backbone.qwen25vl.batch import build_qwen_batch, encode_qwen_item
from nimloth.backbone.qwen25vl.input import Qwen25VLInputBuilder
from nimloth.util.cache.encoding import encode_qwen_item_from_image_grids


TEMPLATE = """{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' }}{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for part in message['content'] %}{% if part['type'] == 'image' %}{{ '<|vision_start|><|image_pad|><|vision_end|>' }}{% else %}{{ part['text'] }}{% endif %}{% endfor %}{% endif %}{{ '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""


def _processor(padding_side):
    specials = ["[UNK]", "<pad>", "<|im_start|>", "<|im_end|>", "<|vision_start|>",
                "<|image_pad|>", "<|vision_end|>", "<|video_pad|>", "<|latent_state|>",
                "<|action_start|>", "<|action_end|>", "<|action_forward|>"]
    words = specials + ["user", "assistant", "old", "reply", "new", "reason", "observe", "next"]
    backend = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = Qwen2TokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="<pad>",
                                  additional_special_tokens=specials[2:], padding_side=padding_side)
    return Qwen2_5_VLProcessor(tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(min_pixels=56*56, max_pixels=56*56),
        chat_template=TEMPLATE)


def _messages(path, *, two_images):
    image = {"type": "image", "image": str(path)}
    result = [{"role": "user", "content": [image, {"type": "text", "text": "observe"}]}]
    if two_images:
        result += [{"role": "assistant", "content": "old reply"},
                   {"role": "user", "content": [image, {"type": "text", "text": "next"}]}]
    result.append({"role": "assistant", "content":
                   "new reason <|action_start|><|action_forward|><|action_end|><|latent_state|>"})
    return result


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "observation.png"
    Image.new("RGB", (56, 56), (12, 33, 61)).save(path)
    return path


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_multimodal_online_cached_and_batch_labels_equal_final_answer(image_path, padding_side):
    processor = _processor(padding_side)
    messages = [_messages(image_path, two_images=False), _messages(image_path, two_images=True)]
    batch = build_qwen_batch([{"messages": item} for item in messages], processor, 512)
    query = processor.tokenizer.convert_tokens_to_ids("<|latent_state|>")
    image = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    expected = processor.tokenizer(messages[0][-1]["content"] + "<|im_end|>\n",
                                   add_special_tokens=False)["input_ids"]
    expected = [token for token in expected if token != query]
    assert len(expected) == 6
    for row, item in enumerate(messages):
        online = encode_qwen_item(item, processor, 512)
        cached = encode_qwen_item_from_image_grids(item, online["image_grid_thw"], processor, 512)
        for key in ("input_ids", "labels", "attention_mask", "image_grid_thw"):
            torch.testing.assert_close(online[key], cached[key], rtol=0, atol=0)
        active = batch["attention_mask"][row].bool()
        torch.testing.assert_close(batch["input_ids"][row, active], online["input_ids"])
        torch.testing.assert_close(batch["labels"][row, active], online["labels"])
        supervised = online["labels"][online["labels"] != -100].tolist()
        assert supervised == expected
        assert image not in supervised and query not in supervised
        assert torch.all(batch["labels"][row, ~active] == -100)
        assert int((online["input_ids"] == image).sum()) == 4 * (row + 1)


@pytest.mark.parametrize("include_labels", [False, True])
def test_complete_prefix_over_limit_fails_without_truncating(image_path, include_labels):
    processor = _processor("right")
    messages = _messages(image_path, two_images=True)
    untouched = copy.deepcopy(messages)
    full = encode_qwen_item(messages, processor, 512, include_labels=include_labels)
    cap = len(full["input_ids"]) - 1
    with pytest.raises(ValueError, match="truncation is not allowed"):
        encode_qwen_item(messages, processor, cap, include_labels=include_labels)
    with pytest.raises(ValueError, match="truncation is not allowed"):
        encode_qwen_item_from_image_grids(messages, full["image_grid_thw"], processor, cap,
                                         include_labels=include_labels)
    with pytest.raises(ValueError, match="truncation is not allowed"):
        build_qwen_batch([{"messages": messages}], processor, cap)
    assert messages == untouched
    again = encode_qwen_item(messages, processor, 512, include_labels=include_labels)
    torch.testing.assert_close(full["input_ids"], again["input_ids"])


def test_cached_incomplete_image_tokens_rejected_before_collation(image_path):
    processor = _processor("right")
    row = encode_qwen_item(_messages(image_path, two_images=True), processor, 512)
    image_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    index = (row["input_ids"] == image_id).nonzero(as_tuple=True)[0][0]
    row["input_ids"][index] = processor.tokenizer.pad_token_id
    with pytest.raises(ValueError, match="image token/grid mismatch"):
        Qwen25VLInputBuilder(processor, 512).collate_encoded([row], include_labels=True)


def test_old_label_cache_version_rejected_by_production_loader(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from nimloth.util.cache import schema
    from nimloth.training.sft.stage3.data.factory import _verify_cache_manifest

    processor = _processor("right")
    data = tmp_path / "records.jsonl"
    data.write_text('{}\n')
    cache = tmp_path / "cache"
    cache.mkdir()
    config = SimpleNamespace(require_prebuilt_cache=True, max_length=512, max_pixels=3136,
        value_gamma=1., latent_token_count=1, mask_latent_query_labels=True,
        preprocess_cache_image_dtype="bfloat16", preprocess_cache_processor_source=None, model=tmp_path)
    kwargs = dict(max_length=512, max_pixels=3136, min_pixels=3136,
        vocab_size=len(processor.tokenizer), value_gamma=1., latent_token_count=1,
        mask_latent_query_labels=True, cache_format=schema.COMPACT_CACHE_FORMAT,
        image_dtype="bfloat16", processor_source=str(tmp_path.resolve()))
    current = schema.cache_fingerprint(data, **kwargs)
    with monkeypatch.context() as old_version:
        old_version.setattr(schema, "CE_MASK_VERSION", "last_assistant_span_v1")
        old = schema.cache_fingerprint(data, **kwargs)
    assert old != current
    manifest = dict(format=schema.COMPACT_CACHE_FORMAT, base_fingerprint=current, count=1)
    (cache / "manifest.json").write_text(json.dumps(manifest))
    verify = dict(cache_dir=cache, jsonl_path=data, expected_count=1, allow_prefix_subset=False,
                  config=config, processor=processor)
    _verify_cache_manifest(**verify)
    manifest["base_fingerprint"] = old
    (cache / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="fingerprint/count mismatch"):
        _verify_cache_manifest(**verify)
