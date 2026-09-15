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


def test_old_label_cache_version_rejected_by_production_loader(tmp_path):
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
    old = schema.cache_fingerprint(data, **kwargs, ce_mask_version="last_assistant_span_v1")
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


def test_image_reuse_builds_fresh_transition_labels_without_touching_source(tmp_path, image_path, monkeypatch):
    import json
    from concurrent.futures import Future
    from types import SimpleNamespace
    from nimloth.rollout.transitions import TransitionSample
    from nimloth.util.cache import build, schema
    from nimloth.util.cache.image_reuse import file_sha256

    processor = _processor("right")
    messages = _messages(image_path, two_images=False)
    paths = [str(image_path.resolve())]
    unbound = [{"role": "user", "content": "<image>observe"}, messages[-1]]
    sample = TransitionSample("record", 0, unbound, paths, 0, paths[0], paths[0])
    data = tmp_path / "data.jsonl"
    data.write_text('{}\n')
    source, destination = tmp_path / "old", tmp_path / "new"
    (source / "images").mkdir(parents=True)
    (source / "transitions").mkdir()
    with Image.open(image_path) as picture:
        pixels = processor.image_processor(images=[picture.convert("RGB")], return_tensors="pt")
    grids = pixels["image_grid_thw"].long()
    image_file = source / "images" / "shard_00000.pt"
    torch.save(dict(pixel_values=pixels["pixel_values"].bfloat16(), image_grid_thw=grids,
                    offsets=torch.tensor([0, pixels["pixel_values"].shape[0]])), image_file)
    old_text = source / "transitions" / "shard_00000.pt"
    torch.save({"old_label_sentinel": True}, old_text)
    index = dict(format=schema.COMPACT_CACHE_FORMAT,
                 images=[dict(path=paths[0], shard=0, index=0, grid_thw=grids[0].tolist())])
    (source / "image_index.json").write_text(json.dumps(index))
    old_base = schema.cache_fingerprint(data, max_length=128, max_pixels=3136,
        min_pixels=3136, vocab_size=len(processor.tokenizer), image_dtype="bfloat16",
        processor_source=str(tmp_path.resolve()), ce_mask_version="last_assistant_span_v1")
    manifest = dict(format=schema.COMPACT_CACHE_FORMAT, base_fingerprint=old_base,
        image_source_fingerprint=build._compact_image_source_fingerprint(paths), image_dtype="bfloat16",
        max_pixels=3136, min_pixels=3136, max_length=128, image_shard_size=128,
        image_shards=1, unique_images=1, ce_mask_version="last_assistant_span_v1",
        transition_expansion_version=schema.TRANSITION_EXPANSION_VERSION)
    (source / "manifest.json").write_text(json.dumps(manifest))
    before = {path: file_sha256(path) for path in source.rglob("*") if path.is_file()}
    monkeypatch.setattr(build, "TransitionJsonlDataset", lambda *args, **kwargs: SimpleNamespace(samples=[sample]))

    def initialize(_model, _min, _max, max_length, latent_count, mask_queries):
        monkeypatch.setattr(build, "_CACHE_PROCESSOR", processor)
        monkeypatch.setattr(build, "_CACHE_MAX_LENGTH", max_length)
        monkeypatch.setattr(build, "_CACHE_LATENT_TOKEN_COUNT", latent_count)
        monkeypatch.setattr(build, "_CACHE_MASK_LATENT_QUERY_LABELS", mask_queries)

    class DirectExecutor:
        def __init__(self, *, initializer, initargs, **kwargs):
            initializer(*initargs)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def submit(self, function, task):
            future = Future()
            future.set_result(function(task))
            return future

    def no_image_reencode(*args):
        raise AssertionError("verified image shard must be reused")

    monkeypatch.setattr(build, "_init_cache_worker", initialize)
    monkeypatch.setattr(build, "ProcessPoolExecutor", DirectExecutor)
    monkeypatch.setattr(build, "_cache_one_image_shard", no_image_reencode)
    build.build_compact_transition_preprocess_cache(jsonl_path=data, cache_dir=destination,
        model_path=tmp_path, processor=processor, max_length=512, max_pixels=3136,
        min_pixels=3136, reuse_image_cache=source)
    assert (destination / "images" / image_file.name).samefile(image_file)
    generated = destination / "transitions" / old_text.name
    assert not generated.samefile(old_text)
    fresh = torch.load(generated, weights_only=True)["entries"][0]["current_enc"]
    expected = encode_qwen_item(messages, processor, 512)
    torch.testing.assert_close(fresh["labels"], expected["labels"])
    assert int((fresh["labels"] != -100).sum()) == 6
    assert all(file_sha256(path) == digest for path, digest in before.items())
    result = json.loads((destination / "manifest.json").read_text())
    assert result["ce_mask_version"] == schema.CE_MASK_VERSION and result["max_length"] == 512
    assert not (destination / "build_state.json").exists()
