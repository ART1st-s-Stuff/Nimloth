from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from nimloth.latent.extraction import (
    LatentActionTokens,
    extract_action_prior,
    extract_latent_state,
    find_extraction_positions,
)


def _token_ids(tokens: LatentActionTokens) -> dict[str, int]:
    return {token: i + 10 for i, token in enumerate(tokens.all_special_tokens)}


def test_find_extraction_positions() -> None:
    tokens = LatentActionTokens()
    ids = _token_ids(tokens)
    input_ids = torch.tensor(
        [
            1,
            ids[tokens.latent_state],
            2,
            ids[tokens.action_start],
            ids[tokens.action_tokens[0]],
            ids[tokens.action_end],
        ]
    )

    positions = find_extraction_positions(input_ids=input_ids, token_ids=ids, tokens=tokens)

    assert positions.latent_state_index == 1
    assert positions.action_start_index == 3
    assert positions.first_action_index == 4


def test_find_extraction_positions_requires_single_latent_token() -> None:
    tokens = LatentActionTokens()
    ids = _token_ids(tokens)
    input_ids = torch.tensor([ids[tokens.latent_state], ids[tokens.latent_state]])

    with pytest.raises(ValueError, match="Expected exactly one"):
        find_extraction_positions(input_ids=input_ids, token_ids=ids, tokens=tokens)


def test_extract_latent_state_reads_latent_token_hidden_state() -> None:
    hidden = torch.arange(1 * 6 * 4, dtype=torch.float32).reshape(1, 6, 4)

    latent = extract_latent_state(hidden, latent_state_index=2)

    torch.testing.assert_close(latent, hidden[0, 2])


def test_extract_action_prior_uses_action_start_causal_logits() -> None:
    logits = torch.zeros(1, 6, 20)
    hidden = torch.arange(1 * 6 * 3, dtype=torch.float32).reshape(1, 6, 3)
    action_token_ids = torch.tensor([11, 12, 13])
    logits[0, 3, 11] = 1.0
    logits[0, 3, 12] = 3.0
    logits[0, 4, 13] = 100.0

    prior = extract_action_prior(
        logits=logits,
        hidden_states=hidden,
        action_start_index=3,
        first_action_index=4,
        action_token_ids=action_token_ids,
        action_tokens=("a", "b", "c"),
    )

    assert prior.probs.argmax().item() == 1
    assert set(prior.token_to_log_prob) == {"a", "b", "c"}


class _FakeTokenizer:
    unk_token_id = -1

    def __init__(self, token_ids: dict[str, int]) -> None:
        self.token_ids = token_ids

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.token_ids.get(token, self.unk_token_id)

    def encode(self, token: str, add_special_tokens: bool = False) -> list[int]:
        if token not in self.token_ids:
            return [self.unk_token_id]
        return [self.token_ids[token]]


class _FakeCausalLM:
    def __init__(self, best_action_token_id: int) -> None:
        self.best_action_token_id = best_action_token_id

    def __call__(self, input_ids, output_hidden_states: bool = False, return_dict: bool = False):
        assert output_hidden_states is True
        assert return_dict is True
        batch, seq_len = input_ids.shape
        hidden = torch.arange(batch * seq_len * 4, dtype=torch.float32).reshape(batch, seq_len, 4)
        logits = torch.zeros(batch, seq_len, 64, dtype=torch.float32)
        logits[:, 3, self.best_action_token_id] = 9.0
        return SimpleNamespace(hidden_states=(hidden - 1000.0, hidden), logits=logits)


def test_extractor_runs_model_and_extracts_latent_state_and_action_prior() -> None:
    from nimloth.latent.extraction import LatentActionExtractor

    tokens = LatentActionTokens()
    token_ids = _token_ids(tokens)
    extractor = LatentActionExtractor(_FakeTokenizer(token_ids))
    input_ids = torch.tensor(
        [
            [
                1,
                token_ids[tokens.latent_state],
                2,
                token_ids[tokens.action_start],
                token_ids[tokens.action_tokens[0]],
                token_ids[tokens.action_end],
            ]
        ]
    )

    latent_state, action_prior, positions = extractor.extract_from_model(
        _FakeCausalLM(best_action_token_id=token_ids[tokens.action_tokens[0]]),
        input_ids=input_ids,
    )

    assert positions.latent_state_index == 1
    assert positions.action_start_index == 3
    assert positions.first_action_index == 4
    torch.testing.assert_close(latent_state, torch.tensor([4.0, 5.0, 6.0, 7.0]))
    assert action_prior is not None
    assert action_prior.probs.argmax().item() == 0
