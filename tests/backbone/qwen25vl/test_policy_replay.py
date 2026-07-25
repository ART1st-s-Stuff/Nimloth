from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from nimloth.agent import (
    AgentPrompt,
    PlannerPolicyTrace,
    PolicyReplayInput,
    PolicyTokenTrace,
    PromptTemplateSpec,
)
from nimloth.backbone.qwen25vl.policy import (
    _logits_to_keep_positions,
    replay_policy_token_log_probs,
)
from nimloth.latent import LatentActionTokens
from nimloth.training.rl.token_value import TokenValueHead


class _Processor:
    class Tokenizer:
        all_special_ids = (60, 61)
        added_tokens_decoder = {
            62: SimpleNamespace(special=True),
            58: SimpleNamespace(special=False),
        }

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            del text
            return [63]

        def decode(self, token_ids, **kwargs):
            assert kwargs == {
                "skip_special_tokens": False,
                "clean_up_tokenization_spaces": False,
                "spaces_between_special_tokens": False,
            }
            if list(token_ids) == [5, 20, 25, 22]:
                return (
                    "reason</think><|latent_state|><|action_start|>"
                    "<|action_(2)|><|action_end|>"
                )
            return "different"

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
            (1, len(logits_to_keep), 64),
            device=input_ids.device,
        )
        return SimpleNamespace(logits=logits)


class _TokenModel(_Model):
    def __init__(self) -> None:
        super().__init__()
        self.lm_head = torch.nn.Linear(4, 64, bias=False)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, *, input_ids, logits_to_keep, **kwargs):
        del kwargs
        self.last_input_ids = input_ids
        self.last_logits_to_keep = logits_to_keep
        hidden = self.scale * torch.ones((1, len(logits_to_keep), 4))
        return SimpleNamespace(logits=self.lm_head(hidden))


def test_logits_to_keep_positions_are_native_indices_for_device_mapped_qwen() -> None:
    positions = _logits_to_keep_positions([3, 5])

    assert positions == [3, 5]
    assert all(isinstance(position, int) for position in positions)
    hidden_states = torch.arange(24).reshape(1, 6, 4)
    assert torch.equal(
        hidden_states[:, positions, :],
        hidden_states[:, torch.tensor([3, 5]), :],
    )


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
        old_log_probs=(-math.log(50.0), None, -math.log(8.0), None),
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
    processor = _Processor()
    assert tuple(
        processor.tokenizer.encode(
            sample.assistant_response[len("<think>") :],
            add_special_tokens=False,
        )
    ) != trace.token_ids

    output = replay_policy_token_log_probs(
        samples=(sample,),
        model=model,
        processor=processor,
        token_id_map=token_id_map,
        device=torch.device("cpu"),
    )

    assert model.last_input_ids.shape == (1, 8)
    assert model.last_logits_to_keep == [3, 5]
    assert torch.allclose(
        output.selected_log_probs,
        torch.tensor([-math.log(50.0), -math.log(8.0)]),
    )
    assert torch.allclose(
        output.entropies,
        torch.tensor([math.log(50.0), math.log(8.0)]),
    )
    old_log_probs = torch.tensor(
        [value for value in trace.old_log_probs if value is not None]
    )
    assert torch.allclose(
        torch.exp(output.selected_log_probs - old_log_probs),
        torch.ones(2),
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

    with pytest.raises(ValueError, match="does not decode"):
        replay_policy_token_log_probs(
            samples=(replace(sample, assistant_response="<think>different</think>"),),
            model=_Model(),
            processor=_Processor(),
            token_id_map=token_id_map,
            device=torch.device("cpu"),
        )


def test_token_credit_replay_captures_only_selected_hidden_states() -> None:
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
        credit_assignment="token",
        token_trace=PolicyTokenTrace(
            token_ids=(5, 20, 25, 22),
            old_log_probs=(-0.2, None, -0.3, None),
            loss_mask=(True, False, True, False),
            token_roles=("reasoning", "injected", "action", "injected"),
            action_token_ids=tuple(range(23, 31)),
            reasoning_text="reason",
            finish_reason="stop",
        ),
        assistant_response=(
            "<think>reason</think><|latent_state|><|action_start|>"
            "<|action_(2)|><|action_end|>"
        ),
    )
    model = _TokenModel()
    token_value_head = TokenValueHead(input_dim=4, hidden_dim=3)

    output = replay_policy_token_log_probs(
        samples=(sample,),
        model=model,
        processor=_Processor(),
        token_id_map=token_id_map,
        device=torch.device("cpu"),
        token_value_head=token_value_head,
    )

    assert output.token_values.shape == (2,)
    output.token_values.sum().backward()
    assert model.scale.grad is None
    assert all(parameter.grad is not None for parameter in token_value_head.parameters())


def test_planner_replay_returns_raw_action_distribution_without_action_ppo() -> None:
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
    uniform = tuple([-math.log(8.0)] * 8)
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
        credit_assignment="token",
        token_trace=PolicyTokenTrace(
            token_ids=(5, 20, 25, 22),
            old_log_probs=(-0.2, None, None, None),
            loss_mask=(True, False, False, False),
            token_roles=("reasoning", "injected", "action", "injected"),
            action_token_ids=tuple(range(23, 31)),
            reasoning_text="reason",
            finish_reason="stop",
        ),
        assistant_response=(
            "<think>reason</think><|latent_state|><|action_start|>"
            "<|action_(2)|><|action_end|>"
        ),
        planner_trace=PlannerPolicyTrace(
            qwen_action_log_probs=uniform,
            candidate_sequences=((2,),),
            candidate_scores=(0.0,),
            root_action_scores=(
                float("-inf"),
                float("-inf"),
                0.0,
                float("-inf"),
                float("-inf"),
                float("-inf"),
                float("-inf"),
                float("-inf"),
            ),
            teacher_action_log_probs=tuple(
                0.0 if index == 2 else float("-inf") for index in range(8)
            ),
            behavior_action_log_probs=tuple(
                0.0 if index == 2 else float("-inf") for index in range(8)
            ),
            horizon=1,
            search_mode="greedy",
        ),
    )
    model = _TokenModel()
    model.lm_head.weight.data.zero_()

    output = replay_policy_token_log_probs(
        samples=(sample,),
        model=model,
        processor=_Processor(),
        token_id_map=token_id_map,
        device=torch.device("cpu"),
        token_value_head=TokenValueHead(input_dim=4, hidden_dim=3),
    )

    assert model.last_logits_to_keep == [3, 5]
    assert output.selected_log_probs.shape == (1,)
    assert output.token_values.shape == (1,)
    assert output.action_log_probs.shape == (1, 8)
    assert output.selected_full_log_probs.shape == (1,)
    assert torch.allclose(
        output.action_log_probs,
        torch.full((1, 8), -math.log(8.0)),
    )
