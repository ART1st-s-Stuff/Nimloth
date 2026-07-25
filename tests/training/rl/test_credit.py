from __future__ import annotations

from dataclasses import replace

import torch

from nimloth.agent import (
    AgentTranscript,
    NimlothPromptTemplate,
    PolicyReplayInput,
    PolicyTokenTrace,
)
from nimloth.training.rl.credit import expand_step_advantages, token_level_gae


def _sample(selected_reasoning_tokens: int) -> PolicyReplayInput:
    template = NimlothPromptTemplate(latent_token_count=1, action_count=8)
    prompt = template.build_response_policy_prompt(
        AgentTranscript(
            system_prompt="system",
            observation_texts=("observation <image>",),
            observation_images=("image.png",),
            action_indices=(),
        )
    )
    reasoning_ids = tuple(range(10, 10 + selected_reasoning_tokens))
    trace = PolicyTokenTrace(
        token_ids=(*reasoning_ids, 30, 31),
        old_log_probs=(
            *([-0.2] * selected_reasoning_tokens),
            -0.3,
            None,
        ),
        loss_mask=(
            *([True] * selected_reasoning_tokens),
            True,
            False,
        ),
        token_roles=(
            *(["reasoning"] * selected_reasoning_tokens),
            "action",
            "injected",
        ),
        action_token_ids=tuple(range(30, 38)),
        reasoning_text="reasoning",
        finish_reason="stop",
    )
    return PolicyReplayInput(
        prompt=prompt,
        action_index=0,
        sampling_temperature=0.7,
        sampling_top_p=0.95,
        latent_token_count=1,
        credit_assignment="turn",
        token_trace=trace,
        assistant_response=(
            "<think>reasoning</think><|latent_state|><|action_start|>"
            "<|action_(0)|><|action_end|>"
        ),
    )


def test_turn_credit_broadcasts_step_advantage_to_selected_response_tokens() -> None:
    actual = expand_step_advantages(
        torch.tensor([1.5, -2.0]),
        (_sample(1), _sample(2)),
        credit_assignment="turn",
    )

    assert actual.tolist() == [1.5, 1.5, -2.0, -2.0, -2.0]


def test_token_level_gae_uses_separate_value_for_each_selected_token() -> None:
    samples = (replace(_sample(1), credit_assignment="token"),)

    actual = token_level_gae(
        torch.tensor([2.0]),
        torch.tensor([0.5, 1.0]),
        samples,
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert torch.allclose(actual.returns, torch.tensor([2.0, 2.0]))
    assert torch.allclose(actual.advantages, torch.tensor([1.0, -1.0]))


def test_token_level_gae_resets_at_each_turn_boundary() -> None:
    samples = (
        replace(_sample(1), credit_assignment="token"),
        replace(_sample(1), credit_assignment="token"),
    )

    actual = token_level_gae(
        torch.tensor([2.0, -1.0]),
        torch.zeros(4),
        samples,
        gamma=0.5,
        gae_lambda=1.0,
    )

    assert torch.allclose(actual.returns, torch.tensor([1.0, 2.0, -0.5, -1.0]))


def test_token_level_gae_rejects_missing_token_values() -> None:
    sample = replace(_sample(2), credit_assignment="token")
    try:
        token_level_gae(
            torch.tensor([1.0]),
            torch.zeros(2),
            (sample,),
            gamma=1.0,
            gae_lambda=1.0,
        )
    except ValueError as error:
        assert "do not align" in str(error)
    else:  # pragma: no cover
        raise AssertionError("token/value alignment must be validated")
