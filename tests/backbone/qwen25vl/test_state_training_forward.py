from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
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
    labels: torch.Tensor | None = None,
    source: str = "archived",
    response: str | None = None,
) -> QwenStateTrainingBatch:
    actual_response = response or (
        "<think>The doorway is open, so move forward.</think>"
        "<|latent_state|><|action_start|><|action_(0)|><|action_end|>"
    )
    tensors = {"input_ids": input_ids}
    if labels is not None:
        tensors["labels"] = labels
    return QwenStateTrainingBatch(
        backbone_batch=BackboneBatch(tensors),
        archived_assistant_responses=(actual_response,) * input_ids.shape[0],
        response_sources=(source,) * input_ids.shape[0],
    )


def test_diagnostic_features_are_optional_and_come_from_the_same_forward() -> None:
    tokens, token_id_map = _token_contract()
    query_ids = [token_id_map[token] for token in latent_state_tokens(16, tokens)]
    row = torch.tensor([[1, 2, 3, *query_ids, token_id_map[tokens.action_start], 63]])
    model = _SameForwardQwen()
    batch = _state_training_batch(row)
    diagnostic_batch = QwenStateTrainingBatch(
        backbone_batch=batch.backbone_batch,
        archived_assistant_responses=batch.archived_assistant_responses,
        response_sources=batch.response_sources,
        diagnostic_image_token_indices=((1, 2),),
        diagnostic_instruction_token_spans=((2, 4),),
    )

    output = forward_qwen_state_training(
        model,
        diagnostic_batch,
        token_id_map,
        torch.device("cpu"),
    )

    assert model.calls == 1
    assert output.fused_image_features is not None
    assert output.instruction_features is not None
    full_hidden = model.model.language_model.norm(
        torch.arange(row.numel() * 4, dtype=torch.float32).reshape(1, row.shape[1], 4)
    )
    torch.testing.assert_close(output.fused_image_features, full_hidden[:, [1, 2]].mean(dim=1))
    torch.testing.assert_close(output.instruction_features, full_hidden[:, 2:4].mean(dim=1))

    ordinary = forward_qwen_state_training(
        _SameForwardQwen(),
        batch,
        token_id_map,
        torch.device("cpu"),
    )
    assert ordinary.fused_image_features is None
    assert ordinary.instruction_features is None


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
    assert output.lm_loss_sum is None
    assert output.lm_valid_token_count == 0


def test_same_forward_computes_exact_selected_shifted_lm_ce_and_masks_queries() -> None:
    tokens, token_id_map = _token_contract()
    query_ids = [token_id_map[token] for token in latent_state_tokens(16, tokens)]
    row = torch.tensor([[
        1,
        5,
        *query_ids,
        token_id_map[tokens.action_start],
        token_id_map[tokens.action_tokens[0]],
        token_id_map[tokens.action_end],
    ]])
    labels = torch.full_like(row, -100)
    target_positions = torch.tensor([1, 18, 19, 20])
    labels[0, target_positions] = row[0, target_positions]
    model = _SameForwardQwen()

    output = forward_qwen_state_training(
        model,
        _state_training_batch(row, labels=labels),
        token_id_map,
        torch.device("cpu"),
    )

    assert model.calls == 1
    assert model.logits_to_keep_seen == [0, 17, 18, 19]
    assert output.lm_loss_sum is not None
    assert output.lm_valid_token_count == 4
    assert output.action_lm_loss_sum is not None
    assert output.action_lm_valid_token_count == 1
    full_hidden = model.model.language_model.norm(
        torch.arange(row.numel() * 4, dtype=torch.float32).reshape(1, row.shape[1], 4)
    )
    full_logits = full_hidden @ torch.arange(
        4 * model.vocab_size, dtype=torch.float32
    ).reshape(4, model.vocab_size)
    expected = F.cross_entropy(
        full_logits[0, torch.tensor([0, 17, 18, 19])].float(),
        row[0, target_positions],
        reduction="sum",
    )
    torch.testing.assert_close(output.lm_loss_sum, expected)
    torch.testing.assert_close(
        output.action_lm_loss_sum,
        F.cross_entropy(
            full_logits[0, 18].float().unsqueeze(0),
            row[0, 19].unsqueeze(0),
            reduction="sum",
        ),
    )
    output.lm_loss_sum.backward()
    assert model.model.language_model.norm.weight.grad is not None

    bad_query_labels = labels.clone()
    bad_query_labels[0, 2] = row[0, 2]
    rejected = _SameForwardQwen()
    with pytest.raises(ValueError, match="query.*masked"):
        forward_qwen_state_training(
            rejected,
            _state_training_batch(row, labels=bad_query_labels),
            token_id_map,
            torch.device("cpu"),
        )
    assert rejected.calls == 0

    bad_first_label = labels.clone()
    bad_first_label[0, 0] = row[0, 0]
    rejected_first = _SameForwardQwen()
    with pytest.raises(ValueError, match="first sequence position"):
        forward_qwen_state_training(
            rejected_first,
            _state_training_batch(row, labels=bad_first_label),
            token_id_map,
            torch.device("cpu"),
        )
    assert rejected_first.calls == 0

    mismatched_response = (
        "<think>The doorway is open, so move right.</think>"
        "<|latent_state|><|action_start|><|action_(2)|><|action_end|>"
    )
    rejected_action = _SameForwardQwen()
    with pytest.raises(ValueError, match="disagrees with archived"):
        forward_qwen_state_training(
            rejected_action,
            _state_training_batch(
                row,
                labels=labels,
                response=mismatched_response,
            ),
            token_id_map,
            torch.device("cpu"),
        )
    assert rejected_action.calls == 0


def test_multiturn_same_forward_selects_final_current_k16_and_rejects_drift() -> None:
    tokens, token_id_map = _token_contract()
    query_ids = [token_id_map[token] for token in latent_state_tokens(16, tokens)]
    action_start = token_id_map[tokens.action_start]
    action_end = token_id_map[tokens.action_end]
    system_example = [61, *query_ids, action_start, 32, action_end]
    observation_example = [62, *query_ids, action_start, 32, action_end]
    completed = [1, *query_ids, action_start, 32, action_end]
    current = [3, *query_ids, action_start, 2]
    row = [
        *system_example,
        *system_example,
        *observation_example,
        *completed,
        *completed,
        *current,
    ]
    input_ids = torch.tensor([row])
    model = _SameForwardQwen()

    output = forward_qwen_state_training(
        model,
        _state_training_batch(input_ids),
        token_id_map,
        torch.device("cpu"),
    )

    final_query_start = (
        2 * len(system_example)
        + len(observation_example)
        + 2 * len(completed)
        + 1
    )
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
        *observation_example,
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
