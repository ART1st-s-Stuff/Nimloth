"""精确验证动作 CE、真实训练入口分支及恢复目标身份。"""

from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from nimloth.training.sft.stage1.checkpoint import RESUME_SCHEMA, validate_resume_state
from nimloth.training.sft.stage1.cli import parse_args
from nimloth.training.sft.stage1.loss import (
    resolve_action_token_ids,
    training_loss,
    validate_action_weight,
    weighted_answer_loss,
)

IDS = tuple(range(2, 12))


@pytest.mark.parametrize("weight", [1.0, 10.0])
def test_exact_shift_mask_eos_and_gradient(weight):
    torch.manual_seed(5)
    logits = torch.randn(2, 6, 13, requires_grad=True)
    labels = torch.tensor([[-100, -100, 2, 4, 1, -100], [-100, 7, 0, 11, 1, -100]])
    target = labels[:, 1:]
    valid = target != -100
    scores = logits[:, :-1][valid]
    targets = target[valid]
    weights = torch.where(torch.isin(targets, torch.tensor(IDS)), weight, 1.0)
    reference = (
        F.cross_entropy(scores, targets, reduction="none") * weights
    ).sum() / weights.sum()
    (expected_grad,) = torch.autograd.grad(reference, logits)
    actual = weighted_answer_loss(logits, labels, IDS, weight, chunk_size=2)
    actual.backward()
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(logits.grad, expected_grad)
    assert logits.grad[:, -1].count_nonzero() == 0
    assert logits.grad[0, 0].count_nonzero() == 0
    assert logits.grad[0, 3].count_nonzero() > 0  # EOS label=1 retains supervision.


def test_training_dispatch_uses_logits_only_when_weighted():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scores = torch.nn.Parameter(
                torch.randn(1, 3, 13, dtype=torch.bfloat16)
            )
            self.got_labels = None

        def forward(self, input_ids, labels=None):
            self.got_labels = labels
            return SimpleNamespace(logits=self.scores, loss=self.scores.float().sum())

    model = Model()
    batch = {
        "input_ids": torch.tensor([[0, 2, 1]]),
        "labels": torch.tensor([[-100, 2, 1]]),
    }
    legacy = training_loss(model, batch)
    assert model.got_labels is batch["labels"]
    torch.testing.assert_close(legacy, model.scores.float().sum())
    loss = training_loss(model, batch, action_token_ids=IDS, action_weight=10)
    assert model.got_labels is None
    loss.backward()
    assert model.scores.dtype == torch.bfloat16
    assert model.scores.grad.dtype == torch.bfloat16


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), 0, -1])
def test_invalid_weight(weight):
    with pytest.raises(ValueError):
        validate_action_weight(weight)


@pytest.mark.parametrize("ids", [IDS[:-1], (2,) * 10, tuple(range(4, 14))])
def test_invalid_ids(ids):
    with pytest.raises(ValueError):
        weighted_answer_loss(torch.zeros(1, 2, 13), torch.tensor([[-100, 1]]), ids, 2)


def test_atomic_tokens():
    class Tokenizer:
        unk_token_id = 0

        def convert_tokens_to_ids(self, token):
            tokens = ["<|action_start|>", "<|action_end|>"] + [
                f"<|action_({i})|>" for i in range(8)
            ]
            return tokens.index(token) + 2

        def encode(self, token, **kwargs):
            return [self.convert_tokens_to_ids(token)]

    tokenizer = Tokenizer()
    assert resolve_action_token_ids(tokenizer) == IDS
    tokenizer.encode = lambda *args, **kwargs: [1, 2]
    with pytest.raises(ValueError, match="atomic"):
        resolve_action_token_ids(tokenizer)


def test_resume_missing_weight_means_one_and_change_rejected():
    state = {
        "resume_schema": RESUME_SCHEMA,
        "identity": {"stage": "format"},
        "world_size": 1,
        "micro_accum": 0,
        "rank_rng_states": [{}],
        "epoch": 1,
        "next_micro_batch": 0,
    }
    validate_resume_state(
        state,
        expected_identity={"stage": "format", "action_token_loss_weight": 1.0},
        rank=0,
        world=1,
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_resume_state(
            state,
            expected_identity={"stage": "format", "action_token_loss_weight": 10.0},
            rank=0,
            world=1,
        )


def test_cli_stage_boundary():
    flags = [
        "--model",
        "/tmp/model",
        "--train-jsonl",
        "/tmp/train",
        "--val-jsonl",
        "/tmp/val",
        "--output-dir",
        "/tmp/out",
    ]
    args, _ = parse_args(flags + ["--action-token-loss-weight", "10"])
    assert args.action_token_loss_weight == 10
    with pytest.raises(ValueError, match="only for format"):
        parse_args(
            flags
            + ["--dino-cache-root", "/tmp/dino", "--action-token-loss-weight", "10"],
            stage="query",
        )


def test_real_tokenizer_registered_protocol():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from nimloth.latent import add_special_tokens

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    add_special_tokens(tokenizer, latent_token_count=None)
    ids = resolve_action_token_ids(tokenizer)
    assert len(ids) == 10
    assert tokenizer.convert_ids_to_tokens(ids[2]) == "<|action_(0)|>"
    assert tokenizer.convert_ids_to_tokens(ids[-1]) == "<|action_(7)|>"


def test_yaml_weight_reaches_cli(tmp_path):
    config = tmp_path / "sft.yaml"
    config.write_text("train:\n  action_token_loss_weight: 7\n")
    args, _ = parse_args(
        [
            "--config",
            str(config),
            "--model",
            "/tmp/model",
            "--train-jsonl",
            "/tmp/train",
            "--val-jsonl",
            "/tmp/val",
            "--output-dir",
            "/tmp/out",
        ]
    )
    assert args.action_token_loss_weight == 7
