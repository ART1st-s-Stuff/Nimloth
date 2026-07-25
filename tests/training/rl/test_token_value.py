from __future__ import annotations

import torch

from nimloth.training.rl.token_value import TokenValueHead


def test_token_value_head_predicts_one_value_per_hidden_state() -> None:
    head = TokenValueHead(input_dim=4, hidden_dim=3)
    hidden = torch.randn(5, 4, requires_grad=True)

    values = head(hidden)
    values.sum().backward()

    assert values.shape == (5,)
    assert hidden.grad is not None
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_token_value_head_checkpoint_roundtrip(tmp_path) -> None:
    original = TokenValueHead(input_dim=4, hidden_dim=3)
    original.save_checkpoint(tmp_path)

    restored = TokenValueHead.load_checkpoint(tmp_path)

    assert restored.input_dim == 4
    assert restored.hidden_dim == 3
    for expected, actual in zip(
        original.state_dict().values(),
        restored.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(expected, actual)
