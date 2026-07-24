from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from nimloth.agent import (
    AgentPrompt,
    PolicyReplayInput,
    PolicyTokenTrace,
    PromptTemplateSpec,
)
from nimloth.backbone.qwen25vl.policy import (
    _logits_to_keep_positions,
    replay_policy_token_log_probs,
)
from nimloth.latent import LatentActionTokens


class _Processor:
    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            if text == (
                "reason</think><|latent_state|><|action_start|>"
                "<|action_(2)|><|action_end|>"
            ):
                return [5, 20, 25, 22]
            return [63]

    tokenizer = Tokenizer()

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["continue_final_message"] is True
        return "prompt"

    def __call__(self, **kwargs):
        assert kwargs["text"] == ["prompt"]
        return {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
        }


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.last_logits_to_keep = None
        self.last_input_ids = None

    def forward(self, *, input_ids, logits_to_keep, **kwargs):
        del kwargs
        self.last_input_ids = input_ids
        self.last_logits_to_keep = logits_to_keep
        logits = self.scale * torch.zeros(
            (1, logits_to_keep.numel(), 64),
            device=input_ids.device,
        )
        return SimpleNamespace(logits=logits)


def test_logits_to_keep_positions_stay_on_cpu_for_device_mapped_qwen() -> None:
    positions = _logits_to_keep_positions([3, 5])

    assert positions.device.type == "cpu"
    assert positions.dtype == torch.long
    assert positions.tolist() == [3, 5]


def test_token_replay_keeps_only_masked_positions_and_role_vocabularies() -> None:
    tokens = LatentActionTokens()
    token_id_map = {
        tokens.latent_state: 20,
        tokens.action_start: 21,
        tokens.action_end: 22,
        **{
            token: 23 + index
            for index, token in enumerate(tokens.action_tokens)
        },
    }
    prompt = AgentPrompt(
        messages=({"role": "assistant", "content": "<think>"},),
        images=(),
        template=PromptTemplateSpec("test", "v1"),
    )
    trace = PolicyTokenTrace(
        token_ids=(5, 20, 25, 22),
        old_log_probs=(-0.2, None, -0.3, None),
        loss_mask=(True, False, True, False),
        token_roles=("reasoning", "injected", "action", "injected"),
        action_token_ids=tuple(range(23, 31)),
        reasoning_text="reason",
        finish_reason="stop",
    )
    sample = PolicyReplayInput(
        prompt=prompt,
        action_index=2,
        sampling_temperature=1.0,
        sampling_top_p=1.0,
        latent_token_count=1,
        credit_assignment="turn",
        token_trace=trace,
        assistant_response=(
            "<think>reason</think><|latent_state|><|action_start|>"
            "<|action_(2)|><|action_end|>"
        ),
    )
    model = _Model()

    output = replay_policy_token_log_probs(
        samples=(sample,),
        model=model,
        processor=_Processor(),
        token_id_map=token_id_map,
        device=torch.device("cpu"),
    )

    assert model.last_input_ids.shape == (1, 8)
    assert model.last_logits_to_keep.device.type == "cpu"
    assert model.last_logits_to_keep.tolist() == [3, 5]
    assert torch.allclose(
        output.selected_log_probs,
        torch.tensor([-math.log(53.0), -math.log(8.0)]),
    )
    assert torch.allclose(
        output.entropies,
        torch.tensor([math.log(53.0), math.log(8.0)]),
    )


def test_token_replay_rejects_response_that_does_not_match_trace() -> None:
    tokens = LatentActionTokens()
    token_id_map = {
        tokens.latent_state: 20,
        tokens.action_start: 21,
        tokens.action_end: 22,
        **{
            token: 23 + index
            for index, token in enumerate(tokens.action_tokens)
        },
    }
    trace = PolicyTokenTrace(
        token_ids=(5, 20, 25, 22),
        old_log_probs=(-0.2, None, -0.3, None),
        loss_mask=(True, False, True, False),
        token_roles=("reasoning", "injected", "action", "injected"),
        action_token_ids=tuple(range(23, 31)),
        reasoning_text="reason",
        finish_reason="stop",
    )
    sample = PolicyReplayInput(
        prompt=AgentPrompt(
            messages=({"role": "assistant", "content": "<think>"},),
            images=(),
            template=PromptTemplateSpec("test", "v1"),
        ),
        action_index=2,
        sampling_temperature=1.0,
        sampling_top_p=1.0,
        latent_token_count=1,
        credit_assignment="turn",
        token_trace=trace,
        assistant_response=(
            "<think>reason</think><|latent_state|><|action_start|>"
            "<|action_(2)|><|action_end|>"
        ),
    )

    with pytest.raises(ValueError, match="does not tokenize"):
        replay_policy_token_log_probs(
            samples=(replace(sample, assistant_response="<think>different</think>"),),
            model=_Model(),
            processor=_Processor(),
            token_id_map=token_id_map,
            device=torch.device("cpu"),
        )
