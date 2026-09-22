"""Chunk projection and CE recompute exactly without retaining vocabulary scores."""
from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from nimloth.backbone.qwen25vl import supervised_lm
from nimloth.backbone.selected_token_rows import _install_leaf


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("weights", [[1., 1.], [1., 0.], [0., 1.], [0., 0.]])
@pytest.mark.parametrize("hidden_grad", [False, True])
def test_joint_checkpoint_chunks_equal_full_projection(dtype, weights, hidden_grad, monkeypatch):
    torch.manual_seed(62)
    head = nn.Linear(16, 257, bias=False, dtype=dtype)
    _install_leaf(head, (200, 201), (220, 221, 222))
    reference = copy.deepcopy(head)
    hidden = torch.randn(2, 303, 16, dtype=dtype, requires_grad=hidden_grad)
    reference_hidden = hidden.detach().clone().requires_grad_(hidden_grad)
    labels = torch.randint(0, 257, (2, 303))
    labels[:, 0] = -100
    labels[0, 151] = -100  # Interior hole, not a suffix truncation.
    labels[1, 134:] = -100
    weights = torch.tensor(weights)
    shapes = []
    chunk_calls = []
    original = supervised_lm._token_loss
    def token_loss(module, chunk, targets):
        chunk_calls.append(chunk.shape[0])
        return original(module, chunk, targets)
    monkeypatch.setattr(supervised_lm, "_token_loss", token_loss)
    def pack(tensor):
        if tensor.untyped_storage().data_ptr() != head.weight.untyped_storage().data_ptr():
            shapes.append(tuple(tensor.shape))
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        with supervised_lm.window_lm_projection(head, labels, weights):
            loss = head(hidden)
    assert loss.ndim == 0
    calls_after_forward = len(chunk_calls)
    assert calls_after_forward == int(weights[0]) * 3 + int(weights[1]) * 2
    if calls_after_forward:
        assert max(chunk_calls) == 128
    # Neither full-prefix nor per-chunk vocabulary scores survive for backward.
    assert not any(len(shape) >= 2 and shape[-1] == 257 and shape[-2] > 1 for shape in shapes)
    # Independent non-checkpointed chunk reference preserves GEMM ordering;
    # the full projection separately verifies the same window loss definition.
    with torch.no_grad():
        full_scores = reference(reference_hidden)
    expected_rows = []
    dense_rows = []
    for row in range(2):
        positions = (labels[row, 1:] != -100).nonzero(as_tuple=True)[0]
        chunk_losses = []
        dense_losses = []
        for chunk in positions.split(128):
            scores = F.linear(reference_hidden[row, chunk], reference.weight)
            for ids, selected_rows in ((reference.nimloth_query_ids, reference.nimloth_query_rows),
                                       (reference.nimloth_protocol_ids, reference.nimloth_protocol_rows)):
                selected = F.linear(reference_hidden[row, chunk], selected_rows.to(dtype))
                scores = scores.index_copy(-1, ids, selected.to(scores.dtype))
            chunk_losses.append(F.cross_entropy(scores.float(), labels[row, chunk + 1], reduction="sum"))
            dense_losses.append(F.cross_entropy(full_scores[row, chunk].float(), labels[row, chunk + 1], reduction="sum"))
        expected_rows.append(sum(chunk_losses) / positions.numel())
        dense_rows.append(sum(dense_losses) / positions.numel())
    expected = (torch.stack(expected_rows) * weights).sum() / weights.sum().clamp_min(1)
    dense_loss = (torch.stack(dense_rows) * weights).sum() / weights.sum().clamp_min(1)
    torch.testing.assert_close(loss, dense_loss, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    loss.backward()
    expected.backward()
    assert len(chunk_calls) == 2 * calls_after_forward
    if hidden_grad:
        torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=0, atol=0)
    for (name, parameter), (_, other) in zip(head.named_parameters(), reference.named_parameters()):
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0, msg=name)
    assert not head._forward_pre_hooks
    assert len(head._forward_hooks) == 1  # Original selected-row hook remains.


def test_projection_context_removes_temporary_hooks_on_exception():
    head = nn.Linear(4, 12, bias=False)
    with pytest.raises(RuntimeError, match="consumer failed"):
        with supervised_lm.window_lm_projection(head, torch.tensor([[-100, 2]]), torch.ones(1)):
            raise RuntimeError("consumer failed")
    assert not head._forward_pre_hooks and not head._forward_hooks


def test_unknown_head_forward_cannot_be_replaced_silently():
    class CustomHead(nn.Linear):
        def forward(self, value):
            return super().forward(value) * 2
    with pytest.raises(TypeError, match="unbiased nn.Linear"):
        with supervised_lm.window_lm_projection(CustomHead(4, 12, bias=False),
                                                torch.tensor([[-100, 2]]), torch.ones(1)):
            pass


@pytest.mark.parametrize("weights", [[1., 0.], [0., 0.]])
@pytest.mark.parametrize("grad_enabled", [True, False])
def test_dense_head_skips_zero_rows_before_ce(weights, grad_enabled, monkeypatch):
    torch.manual_seed(19)
    head = nn.Linear(8, 31, bias=False)
    reference = copy.deepcopy(head)
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    other_hidden = hidden.detach().clone().requires_grad_(True)
    labels = torch.tensor([[-100, 2, 3, -100, 4], [-100, 5, 6, 7, 8]])
    weights = torch.tensor(weights)
    target_calls = []
    original = supervised_lm._token_loss
    def record(module, chunk, targets):
        target_calls.append(targets.detach().tolist())
        return original(module, chunk, targets)
    monkeypatch.setattr(supervised_lm, "_token_loss", record)
    with torch.set_grad_enabled(grad_enabled):
        with supervised_lm.window_lm_projection(head, labels, weights):
            loss = head(hidden)
        scores = reference(other_hidden)[:, :-1]
        per_token = F.cross_entropy(scores.reshape(-1, 31), labels[:, 1:].reshape(-1),
                                    ignore_index=-100, reduction="none").reshape(2, -1)
        per_row = per_token.sum(-1) / (labels[:, 1:] != -100).sum(-1)
        expected = (per_row * weights).sum() / weights.sum().clamp_min(1)
    assert target_calls == ([[2, 3, 4]] if weights[0] else [])
    torch.testing.assert_close(loss, expected)
    if grad_enabled:
        loss.backward()
        expected.backward()
        torch.testing.assert_close(hidden.grad, other_hidden.grad)
        torch.testing.assert_close(head.weight.grad, reference.weight.grad)
        assert hidden.grad[1].count_nonzero() == 0


@pytest.mark.parametrize("weight", [0., 1.])
def test_empty_supervision_still_rejected_on_zero_weight_rows(weight):
    head = nn.Linear(4, 12, bias=False)
    with pytest.raises(ValueError, match="no supervised answer tokens"):
        with supervised_lm.window_lm_projection(
                head, torch.full((1, 3), -100), torch.tensor([weight])):
            pass
    assert not head._forward_pre_hooks and not head._forward_hooks
