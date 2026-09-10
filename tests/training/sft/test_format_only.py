"""Production encoding/identity tests with an isolated text processor, no GPU claim."""

import copy
import json

import pytest
import torch

from nimloth.latent import all_special_tokens_for_latent_count, latent_state_block
from nimloth.training.sft.stage1.cli import parse_args
from nimloth.training.sft.stage1.data import (
    build_preprocess_cache,
    cache_fingerprint,
    collate_fn,
    encode_sample_with_labels,
    render_stage_text,
)
from nimloth.training.sft.stage1.trainer import nimloth_format_correct


class TextProcessor:
    """Simple tokenizer for mask tests only; no HF or image-encoder claims."""

    def __init__(self):
        from nimloth.latent import latent_state_tokens

        self.tokenizer = self
        self.vocab = {
            token: i + 1
            for i, token in enumerate((*latent_state_tokens(4), "<|image_pad|>"))
        }
        self.unk_token_id = -1
        self.pad_token_id = 0

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        result = ""
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                content = "".join(
                    "<|image_pad|>" if part["type"] == "image" else part["text"]
                    for part in content
                )
            result += "[" + message["role"] + "]" + content + "[end]"
        return result + ("[assistant]" if add_generation_prompt else "")

    def __call__(self, *, text, max_length, **kwargs):
        import re

        rows = []
        for item in text:
            tokens = re.findall(r"<\|[^>]+\|>|.", item, re.DOTALL)
            ids = []
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab) + 1
                ids.append(self.vocab[token])
            rows.append(ids[:max_length] if kwargs.get("truncation", True) else ids)
        count = max(map(len, rows))
        return {
            "input_ids": torch.tensor([r + [0] * (count - len(r)) for r in rows]),
            "attention_mask": torch.tensor(
                [[1] * len(r) + [0] * (count - len(r)) for r in rows]
            ),
        }


def messages():
    return [
        {
            "role": "system",
            "content": "Answer <think>...</think>"
            + latent_state_block(16)
            + "<|action_start|>action<|action_end|>",
        },
        {"role": "user", "content": [{"type": "text", "text": "Real observation"}]},
        {
            "role": "assistant",
            "content": "<think>Real recorded reasoning</think>"
            + latent_state_block(16)
            + "<|action_start|><|action_(2)|><|action_end|>",
        },
    ]


def decode(processor, ids):
    vocab = {v: k for k, v in processor.vocab.items()}
    return "".join(vocab[int(i)] for i in ids if int(i) > 0)


def test_format_online_and_cache_preserve_real_answer_without_any_queries():
    processor = TextProcessor()
    original = messages()
    before = copy.deepcopy(original)
    online = collate_fn([{"messages": original}], processor, 1000)
    cached = encode_sample_with_labels(original, processor, 1000)
    torch.testing.assert_close(online["input_ids"][0], cached["input_ids"])
    torch.testing.assert_close(online["labels"][0], cached["labels"])
    text = decode(processor, cached["input_ids"])
    answer = decode(processor, cached["labels"])
    assert "latent_state" not in text
    assert "latent_state" not in answer
    assert "Real recorded reasoning" in answer
    assert "<|action_(2)|>" in answer
    assert "Real observation" not in answer
    assert "Answer " not in answer
    assert original == before
    assert not any(
        "latent_state" in t
        for t in all_special_tokens_for_latent_count(latent_token_count=None)
    )


def test_explicit_query_path_keeps_actual_slots_masked():
    processor = TextProcessor()
    batch = collate_fn(
        [{"messages": messages()}], processor, 1000, latent_token_count=4
    )
    text = decode(processor, batch["input_ids"][0])
    answer = decode(processor, batch["labels"][0])
    assert latent_state_block(4) in text
    assert "latent_state" not in answer
    assert "Real recorded reasoning" in answer


def test_projection_removes_individual_markers_without_erasing_real_text():
    assert (
        render_stage_text("before<|latent_state_15|>middle<|latent_state|>after", None)
        == "beforemiddleafter"
    )


def cli_base():
    return [
        "--model",
        "model",
        "--train-jsonl",
        "train",
        "--val-jsonl",
        "val",
        "--output-dir",
        "out",
    ]


@pytest.mark.parametrize(
    "flag",
    [
        "--latent-token-count",
        "--latent-query-mode",
        "--mask-latent-query-labels",
        "--no-mask-latent-query-labels",
    ],
)
def test_format_cli_rejects_query_options(flag):
    with pytest.raises(SystemExit):
        parse_args(cli_base() + [flag, "1"])


def test_format_cli_has_no_query_semantics():
    args, query = parse_args(cli_base())
    assert query is None
    assert (
        args.latent_token_count
        is args.latent_query_mode
        is args.mask_latent_query_labels
        is None
    )


@pytest.mark.parametrize(
    "name", ["LATENT_TOKEN_COUNT", "LATENT_QUERY_MODE", "MASK_LATENT_QUERY_LABELS"]
)
def test_format_cli_rejects_query_environment(monkeypatch, name):
    monkeypatch.setenv(name, "1")
    with pytest.raises(ValueError, match="environment"):
        parse_args(cli_base())


def test_format_metric_requires_cot_and_action_without_latent_block():
    answer = "<think>Real reasoning</think><|action_start|><|action_(2)|><|action_end|>"
    assert nimloth_format_correct(answer)
    assert not nimloth_format_correct(
        answer.replace("</think>", "</think>" + latent_state_block(4))
    )
    assert not nimloth_format_correct("<|action_start|><|action_(2)|><|action_end|>")


def test_cache_fingerprint_separates_format_from_query(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text("{}")
    common = (path, 1000, 100, 10, 20, -1)
    assert cache_fingerprint(*common) != cache_fingerprint(
        *common, latent_token_count=16, latent_query_mode="generate"
    )


def test_cache_refuses_to_relabel_old_tensors(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"count": 1, "max_length": 1000, "latent_token_count": 16})
    )

    class Dataset:
        def __len__(self):
            return 1

    with pytest.raises(ValueError, match="incompatible preprocess cache"):
        build_preprocess_cache(Dataset(), None, tmp_path, 1000, tmp_path, 10, 100, 1)
