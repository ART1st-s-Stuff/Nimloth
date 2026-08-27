from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nimloth.backbone.base import BackboneBatch
from nimloth.backbone.qwen25vl.state_training import (
    QwenStateTrainingBatch,
    exact_instruction_token_span,
    forward_qwen_state_training,
    require_archived_assistant_response,
)
from nimloth.latent import LatentActionTokens, latent_state_tokens


class _SameForwardQwen(nn.Module):
    def __init__(self, vocab_size: int = 64, hidden_dim: int = 4) -> None:
        super().__init__()
        self.model = SimpleNamespace(
            language_model=SimpleNamespace(norm=nn.LayerNorm(hidden_dim))
        )
        self.vocab_size = vocab_size
        self.calls = 0
        self.logits_to_keep_seen: list[int] | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        logits_to_keep: list[int],
        output_hidden_states: bool,
        return_dict: bool,
        **_kwargs,
    ) -> SimpleNamespace:
        assert output_hidden_states is False
        assert return_dict is True
        self.calls += 1
        self.logits_to_keep_seen = logits_to_keep
        batch, sequence = input_ids.shape
        hidden = torch.arange(
            batch * sequence * 4, dtype=torch.float32
        ).reshape(batch, sequence, 4)
        hidden = self.model.language_model.norm(hidden)
        full_logits = hidden @ torch.arange(
            4 * self.vocab_size, dtype=torch.float32
        ).reshape(4, self.vocab_size)
        return SimpleNamespace(logits=full_logits[:, logits_to_keep])


def _token_contract() -> tuple[LatentActionTokens, dict[str, int]]:
    tokens = LatentActionTokens()
    query_tokens = latent_state_tokens(16, tokens)
    token_id_map = {
        token: 4 + index for index, token in enumerate(query_tokens)
    }
    token_id_map[tokens.action_start] = 20
    token_id_map[tokens.action_end] = 21
    token_id_map.update(
        {token: 32 + index for index, token in enumerate(tokens.action_tokens)}
    )
    return tokens, token_id_map


def _state_training_batch(
    input_ids: torch.Tensor,
    *,
    source: str = "archived",
    response: str | None = None,
) -> QwenStateTrainingBatch:
    actual_response = response or (
        "<think>The doorway is open, so move forward.</think>"
        "<|latent_state|><|action_start|><|action_(0)|><|action_end|>"
    )
    return QwenStateTrainingBatch(
        backbone_batch=BackboneBatch({"input_ids": input_ids}),
        archived_assistant_responses=(actual_response,) * input_ids.shape[0],
        response_sources=(source,) * input_ids.shape[0],
    )


def test_same_forward_returns_row_major_k16_hidden_and_exact_boundary_actions() -> None:
    tokens, token_id_map = _token_contract()
    query_ids = [token_id_map[token] for token in latent_state_tokens(16, tokens)]
    action_start = token_id_map[tokens.action_start]
    row = [1, *query_ids, action_start, 2]
    input_ids = torch.tensor([row, row])
    model = _SameForwardQwen()

    output = forward_qwen_state_training(
        model,
        _state_training_batch(input_ids),
        token_id_map,
        torch.device("cpu"),
        latent_token_count=16,
    )

    assert model.calls == 1
    assert model.logits_to_keep_seen == [17]
    assert output.query_hidden.shape == (2, 16, 4)
    assert output.action_logits.shape == (2, 8)
    full_hidden = model.model.language_model.norm(
        torch.arange(2 * len(row) * 4, dtype=torch.float32).reshape(
            2, len(row), 4
        )
    )
    torch.testing.assert_close(output.query_hidden, full_hidden[:, 1:17])
    full_logits = full_hidden @ torch.arange(
        4 * model.vocab_size, dtype=torch.float32
    ).reshape(4, model.vocab_size)
    action_ids = [token_id_map[token] for token in tokens.action_tokens]
    torch.testing.assert_close(output.action_logits, full_logits[:, 17, action_ids])


def test_multiturn_same_forward_selects_final_current_k16_and_rejects_drift() -> None:
    tokens, token_id_map = _token_contract()
    query_ids = [token_id_map[token] for token in latent_state_tokens(16, tokens)]
    action_start = token_id_map[tokens.action_start]
    action_end = token_id_map[tokens.action_end]
    system_example = [61, *query_ids, action_start, 32, action_end]
    completed = [1, *query_ids, action_start, 32, action_end]
    current = [3, *query_ids, action_start, 2]
    row = [*system_example, *system_example, *completed, *completed, *current]
    input_ids = torch.tensor([row])
    model = _SameForwardQwen()

    output = forward_qwen_state_training(
        model,
        _state_training_batch(input_ids),
        token_id_map,
        torch.device("cpu"),
    )

    final_query_start = 2 * len(system_example) + 2 * len(completed) + 1
    final_boundary = final_query_start + 16
    assert model.calls == 1
    assert model.logits_to_keep_seen == [final_boundary]
    full_hidden = model.model.language_model.norm(
        torch.arange(len(row) * 4, dtype=torch.float32).reshape(1, len(row), 4)
    )
    torch.testing.assert_close(
        output.query_hidden,
        full_hidden[:, final_query_start:final_boundary],
    )

    count_drift_model = _SameForwardQwen()
    count_drifted = torch.tensor([[action_start, *row]])
    with pytest.raises(ValueError, match="structural pair count mismatch"):
        forward_qwen_state_training(
            count_drift_model,
            _state_training_batch(count_drifted),
            token_id_map,
            torch.device("cpu"),
        )
    assert count_drift_model.calls == 0

    malformed_block_model = _SameForwardQwen()
    malformed_block = torch.tensor([[query_ids[0], 63, *row]])
    with pytest.raises(ValueError, match="malformed latent block"):
        forward_qwen_state_training(
            malformed_block_model,
            _state_training_batch(malformed_block),
            token_id_map,
            torch.device("cpu"),
        )
    assert malformed_block_model.calls == 0

    adjacency_drift_model = _SameForwardQwen()
    drifted = torch.tensor([[
        *system_example,
        *system_example,
        *completed,
        *completed,
        3,
        *query_ids,
        63,
        action_start,
        2,
    ]])
    with pytest.raises(ValueError, match="not adjacent"):
        forward_qwen_state_training(
            adjacency_drift_model,
            _state_training_batch(drifted),
            token_id_map,
            torch.device("cpu"),
        )
    assert adjacency_drift_model.calls == 0


def test_state_training_requires_real_archived_cot_and_has_no_fixed_fallback() -> None:
    real = (
        "<think>The doorway is open, so move forward.</think>"
        "<|latent_state|><|action_start|><|action_(0)|><|action_end|>"
    )
    assert require_archived_assistant_response(real, source="archived") == real

    tokens, token_id_map = _token_contract()
    query_ids = [token_id_map[token] for token in latent_state_tokens(16, tokens)]
    input_ids = torch.tensor([[1, *query_ids, token_id_map[tokens.action_start], 2]])
    fixed_batch = _state_training_batch(
        input_ids,
        source="fixed",
        response="<think>What should I do next?</think><|action_start|>",
    )
    model = _SameForwardQwen()
    with pytest.raises(ValueError, match="fixed|archived"):
        forward_qwen_state_training(
            model,
            fixed_batch,
            token_id_map,
            torch.device("cpu"),
        )
    assert model.calls == 0

    with pytest.raises(ValueError, match="archived assistant response"):
        require_archived_assistant_response(None, source="archived")
    with pytest.raises(ValueError, match="archived assistant response"):
        require_archived_assistant_response("", source="archived")


def test_instruction_span_preserves_contextual_bpe_boundary_and_fails_closed() -> None:
    class _BoundaryMergingTokenizer:
        def __call__(
            self,
            text: str,
            *,
            add_special_tokens: bool,
            return_offsets_mapping: bool,
        ) -> dict[str, list[int] | list[tuple[int, int]]]:
            assert add_special_tokens is False
            assert return_offsets_mapping is True
            prefix = "Human Instruction: "
            instruction = "Find it."
            start = len(prefix)
            stop = start + len(instruction)
            assert text.startswith(prefix + instruction)
            # Token 12 overlaps the final period and the real suffix newline.
            return {
                "input_ids": [10, 11, 12, 13],
                "offset_mapping": [
                    (0, start),
                    (start, stop - 1),
                    (stop - 1, stop + 1),
                    (stop + 1, len(text)),
                ],
            }

    tokenizer = _BoundaryMergingTokenizer()
    span = exact_instruction_token_span(
        [99, 10, 11, 12, 13, 98],
        tokenizer=tokenizer,
        instruction="Find it.",
    )
    assert span == (2, 4)

    # Dropping the context-merged boundary token would be an approximate span.
    with pytest.raises(ValueError, match="exact bounded instruction span"):
        exact_instruction_token_span(
            [99, 10, 11, 13, 98],
            tokenizer=tokenizer,
            instruction="Find it.",
        )
