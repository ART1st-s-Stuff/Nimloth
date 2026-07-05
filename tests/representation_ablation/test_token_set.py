from pathlib import Path

import torch

from nimloth.representation_ablation.qwen_tokens import (
    expand_latent_markers_in_messages,
    extract_latent_token_set,
    last_latent_token_run_indices,
)
from nimloth.representation_ablation.token_set import (
    TokenSetPredictorConfig,
    TokenSetValueHead,
    TokenSetWMPredictor,
)


def test_expand_latent_markers_in_messages_string_and_parts() -> None:
    messages = [
        {"role": "assistant", "content": "<think>x</think><|latent_state|><|action_start|>"},
        {"role": "user", "content": [{"type": "text", "text": "a <|latent_state|> b"}]},
    ]
    out = expand_latent_markers_in_messages(messages, 3)
    assert out[0]["content"].count("<|latent_state|>") == 3
    assert out[1]["content"][0]["text"].count("<|latent_state|>") == 3
    assert messages[0]["content"].count("<|latent_state|>") == 1


def test_last_latent_token_run_indices_uses_last_contiguous_run() -> None:
    token_ids = {"<|latent_state|>": 9}
    ids = torch.tensor([9, 1, 9, 9, 2, 9, 9, 9, 3])
    assert last_latent_token_run_indices(ids, token_ids, num_tokens=2) == [6, 7]
    assert last_latent_token_run_indices(ids, token_ids, num_tokens=3) == [5, 6, 7]


def test_extract_latent_token_set_batched() -> None:
    token_ids = {"<|latent_state|>": 9}
    ids = torch.tensor([[1, 9, 9, 2], [9, 9, 3, 0]])
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    tokens = extract_latent_token_set(hidden, ids, token_ids, num_tokens=2)
    assert tokens.shape == (2, 2, 3)
    torch.testing.assert_close(tokens[0], hidden[0, 1:3])
    torch.testing.assert_close(tokens[1], hidden[1, 0:2])


def test_token_set_predictor_shapes_and_rollout() -> None:
    predictor = TokenSetWMPredictor(TokenSetPredictorConfig(num_tokens=4, emb_dim=16, hidden_dim=32, heads=4, depth=2))
    state = torch.randn(3, 4, 16)
    actions = torch.tensor([0, 1, 2])
    pred = predictor(state, actions)
    assert pred.shape == (3, 4, 16)
    rollout = predictor.rollout_states(state, torch.tensor([[0, 1], [2, 3], [4, 5]]))
    assert rollout.shape == (3, 2, 4, 16)


def test_token_set_value_head_shapes_and_checkpoint(tmp_path: Path) -> None:
    head = TokenSetValueHead(emb_dim=16, num_tokens=4, hidden_dim=8)
    state = torch.randn(5, 4, 16)
    values = head(state)
    assert values.shape == (5, 8)
    ckpt = tmp_path / "head"
    head.save_checkpoint(ckpt)
    loaded = TokenSetValueHead.load_checkpoint(ckpt)
    loaded_values = loaded(state)
    torch.testing.assert_close(values, loaded_values)


def test_token_set_predictor_checkpoint(tmp_path: Path) -> None:
    predictor = TokenSetWMPredictor(TokenSetPredictorConfig(num_tokens=2, emb_dim=8, hidden_dim=16, heads=4, depth=1))
    ckpt = tmp_path / "pred"
    predictor.save_checkpoint(ckpt)
    loaded = TokenSetWMPredictor.load_checkpoint(ckpt)
    assert loaded.config.num_tokens == 2
    assert loaded.config.emb_dim == 8
