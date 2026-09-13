from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nimloth.training.sft.stage1.cli import parse_args
from nimloth.training.sft.stage1.trainer import build_optimizer
from nimloth.training.sft.stage2.full_tuning import prepare_full_language


class DenseLanguage(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(12, 4)
        self.lm_head = nn.Linear(4, 12, bias=False)
        self.layers = nn.Sequential(nn.Linear(4, 4))
        self.visual = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        self.config = SimpleNamespace()

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def test_dense_scope_optimizer_update_and_restore():
    model = nn.Module()
    model.language_model = DenseLanguage().bfloat16()
    model.projector = nn.Linear(4, 3).bfloat16()
    prepare_full_language(model)
    frozen = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
    assert set(frozen) == {"language_model.visual.0.weight", "language_model.visual.0.bias",
                           "language_model.visual.1.weight", "language_model.visual.1.bias"}
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
    optimizer = build_optimizer(model, 2e-5, 2e-5, 0.01, 2e-5)
    members = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(members) == len(set(map(id, members)))
    assert set(map(id, members)) == {id(p) for p in model.parameters() if p.requires_grad}
    before = {n: p.clone() for n, p in model.named_parameters()}
    sum(p.sum() for p in members).backward()
    optimizer.step()
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name]) == (not parameter.requires_grad)
    restored = build_optimizer(model, 2e-5, 2e-5, 0.01, 2e-5)
    restored.load_state_dict(optimizer.state_dict())
    assert all(g["lr"] == 2e-5 for g in restored.param_groups)
    assert len(restored.state) == len(members)


def test_full_mode_cli_and_default():
    base = ["--model", "/m", "--train-jsonl", "/t", "--val-jsonl", "/v",
            "--output-dir", "/o", "--dino-cache-root", "/d"]
    default, _ = parse_args(base, stage="query")
    assert default.lora and default.tuning_mode == "selected_lora"
    full, _ = parse_args(base + ["--tuning-mode", "full_language"], stage="query")
    assert not full.lora and full.query_token_lr is None and full.protocol_token_lr is None
    with pytest.raises(ValueError, match="incompatible"):
        parse_args(base + ["--tuning-mode", "full_language", "--lora"], stage="query")
    with pytest.raises(ValueError, match="FSDP"):
        parse_args(base + ["--tuning-mode", "full_language", "--distributed-strategy", "ddp"], stage="query")
