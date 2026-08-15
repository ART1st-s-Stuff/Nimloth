from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nimloth.backbone.qwen25vl.latent import (
    extract_qwen_action_boundary_hidden,
    extract_qwen_latents,
)
from nimloth.latent.extraction import LatentActionTokens, latent_state_tokens


class _FakeQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = SimpleNamespace(language_model=SimpleNamespace(norm=nn.LayerNorm(4)))
        self.output_hidden_states_seen: bool | None = None
        self.logits_to_keep_seen = None

    def forward(self, input_ids, output_hidden_states: bool, return_dict: bool, **kwargs):
        assert return_dict is True
        self.output_hidden_states_seen = output_hidden_states
        self.logits_to_keep_seen = kwargs.get("logits_to_keep")
        batch, seq_len = input_ids.shape
        hidden = torch.arange(batch * seq_len * 4, dtype=torch.float32).reshape(batch, seq_len, 4)
        final_hidden = self.model.language_model.norm(hidden)
        logits = final_hidden @ torch.ones(4, 8)
        loss = logits.sum() * 0.0
        return SimpleNamespace(logits=logits, loss=loss, hidden_states=None)


def test_extract_qwen_latents_uses_final_norm_hook_without_all_hidden_states() -> None:
    tokens = LatentActionTokens()
    token_id_map = {token: i + 10 for i, token in enumerate(tokens.all_special_tokens)}
    input_ids = torch.tensor([[1, token_id_map[tokens.latent_state], 2]])
    model = _FakeQwen()

    latent, loss = extract_qwen_latents(
        model,
        {"input_ids": input_ids},
        token_id_map,
        torch.device("cpu"),
    )

    assert model.output_hidden_states_seen is False
    assert model.logits_to_keep_seen == 1
    assert loss is not None
    expected = model.model.language_model.norm(
        torch.arange(1 * 3 * 4, dtype=torch.float32).reshape(1, 3, 4)
    )[0, 1]
    torch.testing.assert_close(latent, expected.unsqueeze(0))


def test_extract_qwen_latents_can_return_multi_token_block() -> None:
    tokens = LatentActionTokens()
    all_tokens = (*latent_state_tokens(3), tokens.action_start, tokens.action_end, *tokens.action_tokens)
    token_id_map = {token: i + 10 for i, token in enumerate(all_tokens)}
    input_ids = torch.tensor([
        [
            1,
            token_id_map[tokens.latent_state],
            token_id_map["<|latent_state_1|>"],
            token_id_map["<|latent_state_2|>"],
            2,
        ]
    ])
    model = _FakeQwen()

    latent, _loss = extract_qwen_latents(
        model,
        {"input_ids": input_ids},
        token_id_map,
        torch.device("cpu"),
        latent_token_count=3,
    )

    expected_hidden = model.model.language_model.norm(
        torch.arange(1 * 5 * 4, dtype=torch.float32).reshape(1, 5, 4)
    )[0, 1:4]
    assert latent.shape == (1, 3, 4)
    torch.testing.assert_close(latent[0], expected_hidden)


def test_extract_qwen_action_boundary_hidden_uses_last_boundary_per_row() -> None:
    tokens = LatentActionTokens()
    token_id_map = {token: i + 10 for i, token in enumerate(tokens.all_special_tokens)}
    action_start = token_id_map[tokens.action_start]
    input_ids = torch.tensor(
        [
            [action_start, 1, action_start, 2],
            [3, action_start, 4, 5],
        ]
    )
    model = _FakeQwen()

    boundary = extract_qwen_action_boundary_hidden(
        model,
        {"input_ids": input_ids},
        token_id_map,
        torch.device("cpu"),
    )

    expected_hidden = model.model.language_model.norm(
        torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4)
    )
    assert model.logits_to_keep_seen == 1
    torch.testing.assert_close(boundary[0], expected_hidden[0, 2])
    torch.testing.assert_close(boundary[1], expected_hidden[1, 1])


def test_extract_qwen_action_boundary_hidden_rejects_labels() -> None:
    tokens = LatentActionTokens()
    token_id_map = {token: i + 10 for i, token in enumerate(tokens.all_special_tokens)}
    input_ids = torch.tensor([[token_id_map[tokens.action_start]]])

    with pytest.raises(ValueError, match="must not include labels"):
        extract_qwen_action_boundary_hidden(
            _FakeQwen(),
            {"input_ids": input_ids, "labels": input_ids.clone()},
            token_id_map,
            torch.device("cpu"),
        )


def test_extract_qwen_latents_keeps_full_supervised_lm_loss() -> None:
    tokens = LatentActionTokens()
    token_id_map = {token: i + 10 for i, token in enumerate(tokens.all_special_tokens)}
    input_ids = torch.tensor([[1, token_id_map[tokens.latent_state], 2]])
    model = _FakeQwen()

    _latent, loss = extract_qwen_latents(
        model,
        {"input_ids": input_ids, "labels": input_ids.clone()},
        token_id_map,
        torch.device("cpu"),
    )

    assert model.logits_to_keep_seen is None
    assert loss is not None
