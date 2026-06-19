from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from nimloth.latent.extraction import LatentActionTokens
from nimloth.training.sft2.qwen_latent import extract_qwen_latents


class _FakeQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = SimpleNamespace(language_model=SimpleNamespace(norm=nn.LayerNorm(4)))
        self.output_hidden_states_seen: bool | None = None

    def forward(self, input_ids, output_hidden_states: bool, return_dict: bool, **kwargs):
        assert return_dict is True
        self.output_hidden_states_seen = output_hidden_states
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
    assert loss is not None
    expected = model.model.language_model.norm(
        torch.arange(1 * 3 * 4, dtype=torch.float32).reshape(1, 3, 4)
    )[0, 1]
    torch.testing.assert_close(latent, expected.unsqueeze(0))
