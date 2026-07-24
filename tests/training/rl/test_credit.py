from __future__ import annotations

import torch

from nimloth.agent import (
    AgentTranscript,
    NimlothPromptTemplate,
    PolicyReplayInput,
    PolicyTokenTrace,
)
from nimloth.training.rl.credit import expand_step_advantages


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
