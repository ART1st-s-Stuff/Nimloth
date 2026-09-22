"""CPU policy/math checks: these do not exercise CUDA FSDP flattening."""
import pytest
import torch
from torch import nn

from nimloth.training.sft.stage3 import fsdp


class TinyQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Sequential(nn.Linear(3, 3), nn.LayerNorm(3))
        self.embedding = nn.Embedding(8, 3).to(torch.bfloat16)
        self.embedding.weight.requires_grad_(False)
        self.embedding.register_parameter('selected_rows', nn.Parameter(torch.ones(2, 3)))
        self.decoder = nn.Linear(3, 3)


def test_policy_separates_frozen_table_and_keeps_visual_rotary_float():
    model = TinyQwen()
    policy, ignored, precision = fsdp.qwen_wrap_policy(model)
    assert ignored == {model.embedding.weight}
    assert precision.param_dtype == torch.bfloat16
    assert precision.reduce_dtype == torch.float32
    assert precision.buffer_dtype is None
    assert not precision.keep_low_precision_grads
    # CustomPolicy delegates exact per-module options through its public run.
    policies = policy._run_policy(model, ignored_modules=set(), root_kwargs={})
    assert model.visual in policies
    assert policies[model.visual]['mixed_precision'].cast_forward_inputs is False
    assert policies[model.visual[0]]['mixed_precision'].param_dtype == torch.bfloat16
    assert policies[model.embedding]['mixed_precision'].cast_forward_inputs is True
    model.decoder.to(torch.bfloat16)
    with pytest.raises(ValueError, match='FP32'):
        fsdp.qwen_wrap_policy(model)


def test_combined_norm_reduces_shards_but_not_replicated_branch(monkeypatch):
    qwen = nn.Linear(1, 1, bias=False)
    wm = nn.Linear(1, 1, bias=False)
    qwen.weight.grad = torch.tensor([[3.]])
    wm.weight.grad = torch.tensor([[4.]])
    monkeypatch.setattr(fsdp, 'is_fsdp', lambda _: True)
    calls = []
    def reduce(tensor, op):
        calls.append(float(tensor))
        tensor.add_(16.)  # Other rank's Qwen shard norm squared; not its WM.
    monkeypatch.setattr(fsdp.dist, 'all_reduce', reduce)
    norm = fsdp.clip_mixed_grad_norm(qwen, (wm, wm), 1.)
    assert calls == [9.]
    assert norm.item() == pytest.approx(41**.5)
    assert wm.weight.grad.item() == pytest.approx(4/41**.5)


def test_fsdp_rejects_cpu_before_constructing_wrapper():
    with pytest.raises(ValueError, match='multi-rank CUDA'):
        fsdp.wrap_qwen_fsdp(TinyQwen(), torch.device('cpu'))


class Qwen2_5_VLDecoderLayer(nn.Sequential):
    pass


class Qwen2_5_VLVisionBlock(nn.Sequential):
    pass


def test_block_policy_groups_only_block_internals_and_preserves_vocab_owners():
    model = TinyQwen()
    model.decoder = Qwen2_5_VLDecoderLayer(nn.Linear(3, 3), nn.LayerNorm(3), nn.Linear(3, 3))
    model.visual = nn.Sequential(Qwen2_5_VLVisionBlock(nn.Linear(3, 3), nn.LayerNorm(3)), nn.Linear(3, 3))
    model.lm_head = nn.Linear(3, 8)
    linear, ignored_linear, _ = fsdp.qwen_wrap_policy(model)
    block, ignored_block, _ = fsdp.qwen_wrap_policy(model, "block")
    linear = linear._run_policy(model, ignored_modules=set(), root_kwargs={})
    block = block._run_policy(model, ignored_modules=set(), root_kwargs={})
    assert ignored_linear == ignored_block == {model.embedding.weight}
    assert set(block) == {model.visual, model.visual[0], model.visual[1],
                          model.decoder, model.embedding, model.lm_head}
    assert set(linear) - set(block) == {model.decoder[0], model.decoder[2], model.visual[0][0]}
    assert block[model.visual[0]]["mixed_precision"].cast_forward_inputs is False
    # Every trainable parameter has exactly one nearest target/root owner.
    owners = {}
    for name, module in model.named_modules():
        for parameter in module.parameters(recurse=False):
            if parameter.requires_grad:
                candidates = [(prefix, ancestor) for prefix, ancestor in model.named_modules()
                              if ancestor in block and (name == prefix or name.startswith(prefix + "."))]
                owner = max(candidates, key=lambda item: len(item[0]))[1] if candidates else model
                owners[id(parameter)] = owner
    assert len(owners) == sum(p.requires_grad for p in model.parameters())
    assert owners[id(model.visual[0][0].weight)] is model.visual[0]
    assert owners[id(model.embedding.selected_rows)] is model.embedding


def test_unknown_wrap_granularity_rejected():
    with pytest.raises(ValueError, match="granularity"):
        fsdp.qwen_wrap_policy(TinyQwen(), "automatic")
