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
