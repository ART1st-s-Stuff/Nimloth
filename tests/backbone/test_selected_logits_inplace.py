"""Exact old/new selected-logit values, gradients, and storage ownership."""
from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from nimloth.backbone.selected_token_rows import _install_leaf


def _old_projection(head, hidden):
    output = F.linear(hidden, head.weight)
    for ids, rows in ((head.nimloth_query_ids, head.nimloth_query_rows),
                      (head.nimloth_protocol_ids, head.nimloth_protocol_rows)):
        selected = F.linear(hidden, rows.to(hidden.dtype))
        output = output.index_copy(-1, ids.to(output.device), selected.to(output.dtype))
    return output


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("hidden_grad", [False, True])
@pytest.mark.parametrize("dense_grad", [False, True])
def test_inplace_logits_match_old_projection_and_every_gradient(dtype, hidden_grad, dense_grad):
    torch.manual_seed(812)
    head = nn.Linear(7, 19, bias=False, dtype=dtype)
    _install_leaf(head, (3, 17), (1, 5, 8))
    # Production freezes this table; additionally check dense gradient masking.
    head.weight.requires_grad_(dense_grad)
    with torch.no_grad():
        head.nimloth_query_rows.add_(0.37)
        head.nimloth_protocol_rows.sub_(0.29)
    reference = copy.deepcopy(head)
    hidden = torch.randn(2, 5, 7, dtype=dtype, requires_grad=hidden_grad)
    reference_hidden = hidden.detach().clone().requires_grad_(hidden_grad)
    pointers = []
    handle = head.register_forward_hook(
        lambda module, args, output: pointers.append(output.data_ptr()), prepend=True)
    try:
        # Two distinct forward graphs must remain valid until joint backward.
        actual = [head(hidden), head(hidden * 0.7)]
    finally:
        handle.remove()
    expected = [_old_projection(reference, reference_hidden),
                _old_projection(reference, reference_hidden * 0.7)]
    for out, old, pointer in zip(actual, expected, pointers):
        assert out.data_ptr() == pointer  # Same fresh Linear vocabulary buffer.
        assert torch.equal(out, old)
    def loss(outputs):
        return sum(out.float().square().mean() + out.float().sin().sum() for out in outputs)
    with torch.autograd.detect_anomaly():
        loss(actual).backward()
        loss(expected).backward()
    assert (hidden.grad is None) == (reference_hidden.grad is None)
    if hidden_grad:
        assert torch.equal(hidden.grad, reference_hidden.grad)
    for (name, parameter), (other_name, other) in zip(head.named_parameters(), reference.named_parameters()):
        assert name == other_name
        assert (parameter.grad is None) == (other.grad is None)
        if parameter.grad is not None:
            assert torch.equal(parameter.grad, other.grad), name
    if dense_grad:
        assert head.weight.grad[[1, 3, 5, 8, 17]].count_nonzero() == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_inplace_logits_no_grad_matches_old_projection(dtype):
    head = nn.Linear(7, 19, bias=False, dtype=dtype)
    _install_leaf(head, (3, 17), (1, 5, 8))
    hidden = torch.randn(2, 5, 7, dtype=dtype)
    with torch.no_grad():
        actual = head(hidden)
        expected = _old_projection(head, hidden)
    assert torch.equal(actual, expected)
    assert not actual.requires_grad
