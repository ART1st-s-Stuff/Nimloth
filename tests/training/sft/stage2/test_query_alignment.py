"""CPU interface/gradient tests, with a tiny causal model as an isolated substitute."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from nimloth.training.sft.stage1.checkpoint import save_checkpoint
from nimloth.training.sft.stage1.trainer import build_optimizer
from nimloth.training.sft.stage2.config import QueryAlignmentConfig
from nimloth.training.sft.stage2.data import answer_examples
from nimloth.training.sft.stage2.model import QueryAlignmentModel
from nimloth.wm.grid import SharedSlotProjector, load_sft1_slot_projector


class TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 6)
        self.model = nn.Module()
        self.model.norm = nn.LayerNorm(6)
        self.lm_head = nn.Linear(6, 16)
        self.config = SimpleNamespace(nimloth_training_stage="query")

    def forward(self, input_ids, labels, **kwargs):
        hidden = self.model.norm(self.embed_tokens(input_ids).cumsum(1))
        logits = self.lm_head(hidden)
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 16), labels[:, 1:].reshape(-1)
        )
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def save_pretrained(self, directory, **kwargs):
        torch.save(self.state_dict(), directory / "tiny_lm.pt")


def make_model():
    objective = QueryAlignmentConfig(grid_size=2, projector_hidden_dim=7)
    return QueryAlignmentModel(
        TinyCausalLM(),
        SharedSlotProjector(6, 3, 7, grid_tokens=4),
        [6, 7, 8, 9],
        objective,
    )


def inputs():
    return {
        "input_ids": torch.tensor([[1, 2, 6, 7, 8, 9, 3, 4]]),
        "labels": torch.tensor([[-100, -100, -100, -100, -100, -100, 3, 4]]),
        "query_positions": torch.tensor([[2, 3, 4, 5]]),
        "dino_target": torch.randn(1, 4, 3, requires_grad=True),
    }


def test_combined_loss_reaches_backbone_queries_projector_and_lm_but_not_teacher():
    model = make_model()
    batch = inputs()
    optimizer = build_optimizer(model, 1e-3, 2e-3, 0)
    trainable = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert trainable == {id(p) for p in model.parameters() if p.requires_grad}
    before = model.projector.net[0].weight.detach().clone()
    model(**batch).loss.backward()
    assert batch["dino_target"].grad is None
    assert model.projector.net[0].weight.grad.abs().sum() > 0
    assert model.language_model.embed_tokens.weight.grad[6:10].abs().sum() > 0
    assert model.language_model.lm_head.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, model.projector.net[0].weight)


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda b: b.update(dino_target=torch.zeros(1, 1, 3)), "shape mismatch"),
        (
            lambda b: b.update(query_positions=torch.tensor([[3, 2, 4, 5]])),
            "ordered query",
        ),
        (
            lambda b: b.update(labels=torch.full_like(b["labels"], -100)),
            "target-answer",
        ),
    ],
)
def test_invalid_contracts_fail(change, match):
    batch = inputs()
    change(batch)
    with pytest.raises(ValueError, match=match):
        make_model()(**batch)


def test_checkpoint_restores_projector_and_preserves_stage_for_legacy_wm_loader(
    tmp_path,
):
    model = make_model()
    processor = SimpleNamespace(save_pretrained=lambda path: None)
    save_checkpoint(model, processor, tmp_path, "epoch1", latent_token_count=4)
    checkpoint = tmp_path / "epoch1"
    state = torch.load(checkpoint / "training_state.pt", weights_only=False)
    assert state["training_stage"] == "query"
    restored = make_model()
    restored.restore_projector(checkpoint)
    shared = load_sft1_slot_projector(
        checkpoint, qwen_hidden_dim=6, state_dim=3, grid_tokens=4
    )
    for key, value in model.projector.state_dict().items():
        torch.testing.assert_close(restored.projector.state_dict()[key], value)
        torch.testing.assert_close(shared.state_dict()[key], value)
    restored.objective = QueryAlignmentConfig(
        grid_size=2, projector_hidden_dim=7, weight_dino=2
    )
    with pytest.raises(ValueError, match="configuration mismatch"):
        restored.restore_projector(checkpoint)


def test_dialogue_expansion_keeps_real_turn_observations_and_cot():
    messages = [
        {"role": "user", "content": [{"type": "image", "image": "before.png"}]},
        {"role": "assistant", "content": "<think>turn one observation</think>answer"},
        {"role": "user", "content": [{"type": "image", "image": "after.png"}]},
        {"role": "assistant", "content": "<think>turn two observation</think>answer"},
    ]
    examples, paths = answer_examples([{"messages": messages}])
    assert paths == ["before.png", "after.png"]
    assert examples[0]["messages"] == messages[:2]
    assert examples[1]["messages"] == messages
    messages[-1]["content"] = "<think></think>answer"
    with pytest.raises(ValueError, match="recorded nonempty CoT"):
        answer_examples([{"messages": messages}])


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


def records(tmp_path):
    from PIL import Image

    from nimloth.latent import latent_state_block

    path = tmp_path / "image.png"
    Image.new("RGB", (2, 2)).save(path)
    return [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(path)},
                        {"type": "text", "text": "prompt"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": "<think>observed</think>"
                    + latent_state_block(4)
                    + "<|action_start|>ACTION<|action_end|>",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(path)},
                        {"type": "text", "text": "next"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": "<think>changed</think>"
                    + latent_state_block(4)
                    + "<|action_start|>OTHER<|action_end|>",
                },
            ]
        }
    ]


def test_real_collation_masks_prompt_padding_queries_and_repeated_history(tmp_path):
    from nimloth.backbone.dino_grid import DINOV2_LARGE_IDENTITY, CachedDINOGridTargets
    from nimloth.training.sft.stage1.data import (
        collate_cached_fn,
        collate_fn,
        encode_sample_with_labels,
    )
    from nimloth.training.sft.stage2.data import QueryAlignmentCollator

    processor = TextProcessor()
    batch = records(tmp_path)
    encoded = collate_fn(batch, processor, 1000, latent_token_count=4)
    cached = collate_cached_fn(
        [
            encode_sample_with_labels(
                batch[0]["messages"], processor, 1000, latent_token_count=4
            )
        ],
        0,
    )
    torch.testing.assert_close(encoded["labels"], cached["labels"])
    assert torch.all(encoded["labels"][encoded["input_ids"] <= 5] == -100)
    features = torch.randn(1, 4, 1024)
    targets = CachedDINOGridTargets(
        identity=DINOV2_LARGE_IDENTITY,
        grid_size=2,
        path_to_feature={str(tmp_path / "image.png"): (features, 0)},
        cache_fingerprint="test",
    )
    collator = QueryAlignmentCollator(processor, 1000, 4, targets)
    result = collator(batch)
    assert result["labels"].shape[0] == 2
    assert (result["labels"] != -100).sum() == (encoded["labels"] != -100).sum()
    assert torch.all(result["labels"][result["attention_mask"] == 0] == -100)
    assert torch.all(result["labels"].gather(1, result["query_positions"]) == -100)
    assert not result["dino_target"].requires_grad
    torch.testing.assert_close(result["dino_target"], features.expand(2, -1, -1))
    # A sampled answer prefix produces one row, without supervising history again.
    prefix_collator = QueryAlignmentCollator(
        processor, 1000, 4, targets, last_answer_only=True
    )
    prefixes, _ = answer_examples(batch)
    for index, prefix in enumerate(prefixes):
        single = prefix_collator([prefix])
        assert single["input_ids"].shape[0] == 1
        width = single["input_ids"].shape[1]
        for key in ("input_ids", "labels", "attention_mask"):
            torch.testing.assert_close(single[key][0], result[key][index, :width])
        torch.testing.assert_close(
            single["query_positions"][0], result["query_positions"][index]
        )
    logits = torch.randn(
        *result["labels"].shape, len(processor.vocab) + 1, requires_grad=True
    )
    nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        result["labels"][:, 1:].reshape(-1),
    ).backward()
    assert torch.all(logits.grad[:, :-1][result["labels"][:, 1:] == -100] == 0)
    with pytest.raises(ValueError, match="truncation is forbidden"):
        QueryAlignmentCollator(processor, 30, 4, targets)(batch)
    # Even complete query slots do not make a partially truncated action valid.
    tail_limit = int(result["query_positions"][0, -1]) + 1
    with pytest.raises(ValueError, match="truncation is forbidden"):
        QueryAlignmentCollator(processor, tail_limit, 4, targets)(prefixes[:1])


def test_answer_dataset_indexes_every_answer_once_and_preserves_history(tmp_path):
    import json

    from nimloth.training.sft.stage1.data import NimlothVLSFTDataset
    from nimloth.training.sft.stage2.data import AnswerPrefixDataset

    messages = [
        {"role": "system", "content": "navigate"},
        {"role": "user", "content": "<image>first"},
        {"role": "assistant", "content": "<think>first</think>answer1"},
        {"role": "user", "content": "<image>second"},
        {"role": "assistant", "content": "<think>second</think>answer2"},
        {"role": "user", "content": "<image>terminal"},
    ]
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "episode",
                "messages": messages,
                "image_paths": ["first.png", "second.png", "terminal.png"],
            }
        )
        + "\n"
    )
    trajectories = NimlothVLSFTDataset(path, None)
    dataset = AnswerPrefixDataset(trajectories)
    assert len(dataset) == 2
    full = trajectories.get_messages(0)
    assert dataset[0]["messages"] == full[:3]
    assert dataset[1]["messages"] == full[:5]
    for index, expected in enumerate(("first.png", "second.png")):
        examples, paths = answer_examples([dataset[index]], last_answer_only=True)
        assert len(examples) == 1
        assert paths == [expected]
    with pytest.raises(ValueError, match="end at its target"):
        answer_examples([{"messages": full}], last_answer_only=True)


def test_query_state_is_before_action_and_uses_same_teacher_forced_forward():
    model = make_model()
    captured = []
    handle = model.projector.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone())
    )
    batch = inputs()
    model(**batch)
    batch["input_ids"][0, -2:] = torch.tensor([10, 11])
    model(**batch)
    handle.remove()
    assert len(captured) == 2
    torch.testing.assert_close(captured[0], captured[1])


def test_cli_selects_real_stage_and_validates_grid_before_model_load(tmp_path):
    from nimloth.training.sft.stage1.cli import parse_args

    base = [
        "--model",
        "model",
        "--train-jsonl",
        "train",
        "--val-jsonl",
        "val",
        "--output-dir",
        "out",
    ]
    args, objective = parse_args(base)
    assert objective is None and not args.no_cache
    args, objective = parse_args(
        base + ["--dino-cache-root", "cache", "--weight-dino", "2"], stage="query"
    )
    assert args.no_cache and objective.grid_tokens == args.latent_token_count == 16
    assert objective.weight_dino == 2
    with pytest.raises(ValueError, match="token count"):
        parse_args(
            base + ["--dino-cache-root", "cache", "--latent-token-count", "1"],
            stage="query",
        )
    config = tmp_path / "config.yaml"
    config.write_text(
        "latent:\n  token_count: 4\nquery_alignment:\n  grid_size: 2\n  dino_cache_root: cached_targets\n  weight_dino: 3\n"
    )
    args, objective = parse_args(base + ["--config", str(config)], stage="query")
    assert objective.grid_tokens == 4 and objective.weight_dino == 3
    assert args.dino_cache_root.name == "cached_targets"


def test_resume_stage_distinguishes_legacy_format_from_wm_and_query(tmp_path):
    from nimloth.training.sft.stage1.checkpoint import validate_resume_stage

    legacy = {"step": 3, "epoch": 1, "best_val": 0.5, "lora": True}
    validate_resume_stage(legacy, tmp_path, "format")
    with pytest.raises(ValueError, match="stage mismatch"):
        validate_resume_stage(legacy, tmp_path, "query")
    with pytest.raises(ValueError, match="WM/value"):
        validate_resume_stage({**legacy, "best_val_wm_mse": 0.2}, tmp_path, "format")
    with pytest.raises(ValueError, match="identity"):
        validate_resume_stage({}, tmp_path, "format")


def test_merge_export_requires_and_preserves_exact_query_artifacts(tmp_path):
    from nimloth.training.sft.stage1.checkpoint_export import copy_query_artifacts

    source, destination = tmp_path / "source", tmp_path / "merged"
    source.mkdir()
    with pytest.raises(FileNotFoundError, match="slot_projector"):
        copy_query_artifacts(source, destination, stage="query")
    (source / "slot_projector.pt").write_bytes(b"exact projector checkpoint bytes")
    (source / "grid_state_config.json").write_text('{"training_stage":"query"}')
    copy_query_artifacts(source, destination, stage="query")
    for name in ("slot_projector.pt", "grid_state_config.json"):
        assert (source / name).read_bytes() == (destination / name).read_bytes()


def test_real_tiny_qwen_multimodal_forward_and_backward():
    """Exercise the real HF image/CoT/query/CE path with random CPU unit weights."""
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    from nimloth.training.sft.stage1.trainer import (
        resize_token_embeddings_and_sync_vocab,
    )

    config = Qwen2_5_VLConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "rope_scaling": {"type": "mrope", "mrope_section": [1, 1, 2]},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 16,
        },
        image_token_id=28,
        video_token_id=29,
        vision_start_token_id=30,
        vision_end_token_id=31,
    )
    language_model = Qwen2_5_VLForConditionalGeneration(config)
    resize_token_embeddings_and_sync_vocab(language_model, 32)
    model = QueryAlignmentModel(
        language_model,
        SharedSlotProjector(16, 3, 8, grid_tokens=4),
        [6, 7, 8, 9],
        QueryAlignmentConfig(grid_size=2, projector_hidden_dim=8),
    )
    ids = torch.tensor([[1, 30, 28, 31, 2, 6, 7, 8, 9, 3, 4]])
    labels = torch.full_like(ids, -100)
    labels[0, -2:] = ids[0, -2:]
    loss = model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        labels=labels,
        query_positions=torch.tensor([[5, 6, 7, 8]]),
        dino_target=torch.ones(1, 4, 3),
        pixel_values=torch.randn(4, 3 * 2 * 14 * 14),
        image_grid_thw=torch.tensor([[1, 2, 2]]),
    ).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert model.projector.net[0].weight.grad.abs().sum() > 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in language_model.model.visual.parameters()
    )


def test_resume_query_vocabulary_preserves_trained_rows_even_if_base_tokenizer_added_slots():
    from nimloth.latent import latent_state_tokens
    from nimloth.training.sft.stage1.trainer import prepare_query_vocabulary

    class VocabularyModel(nn.Module):
        def __init__(self, size):
            super().__init__()
            self.embedding = nn.Embedding(size, 3)
            self.config = SimpleNamespace(vocab_size=size)

        def get_input_embeddings(self):
            return self.embedding

        def resize_token_embeddings(self, size):
            previous = self.embedding.weight.detach().clone()
            self.embedding = nn.Embedding(size, 3)
            with torch.no_grad():
                self.embedding.weight[: len(previous)].copy_(previous)

    ids = dict(zip(latent_state_tokens(4), [6, 7, 8, 9], strict=True))
    resumed = VocabularyModel(10)
    before = resumed.embedding.weight.detach().clone()
    prepare_query_vocabulary(resumed, 10, ids, added_tokens=3, latent_token_count=4)
    torch.testing.assert_close(resumed.embedding.weight, before)
    initial = VocabularyModel(7)
    slot_zero = initial.embedding.weight[6].detach().clone()
    prepare_query_vocabulary(initial, 10, ids, added_tokens=3, latent_token_count=4)
    torch.testing.assert_close(initial.embedding.weight[7:10], slot_zero.expand(3, -1))
