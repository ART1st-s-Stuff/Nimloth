from __future__ import annotations

import math

import pytest

from nimloth.agent import (
    AgentRuntime,
    AgentPrompt,
    NimlothPromptTemplate,
    PolicyDecision,
    PolicyTokenTrace,
)
from nimloth.environment.navigation import NAVIGATION_ACTION_SPACE


class _RecordingPolicy:
    prompt_mode = "response"

    def __init__(self) -> None:
        self.prompts: list[AgentPrompt] = []

    def reset_episode(self) -> None:
        self.prompts.clear()

    def select_action(self, prompt: AgentPrompt) -> PolicyDecision:
        self.prompts.append(prompt)
        action_index = len(self.prompts) - 1
        return PolicyDecision(
            action_index=action_index,
            action_log_probs=tuple([-math.log(8.0)] * 8),
            response=(
                f"<think>Reason {len(self.prompts)}.</think><|latent_state|>"
                f"<|action_start|><|action_({action_index})|><|action_end|>"
            ),
        )

    def generate_state_prefix(self, prompt: AgentPrompt) -> str:
        self.prompts.append(prompt)
        return "<think>Terminal reasoning.</think><|latent_state|><|action_start|>"


def _template() -> NimlothPromptTemplate:
    return NimlothPromptTemplate(latent_token_count=1, action_count=8)


def test_navigation_agent_runs_real_history_through_one_policy() -> None:
    policy = _RecordingPolicy()
    agent = AgentRuntime(
        policy=policy,
        action_space=NAVIGATION_ACTION_SPACE,
        prompt_template=_template(),
    )
    agent.reset(system_prompt="system")

    agent.observe(text="first <image>", image="image-0")
    first = agent.act()
    agent.observe(text="second <image>", image="image-1")
    second = agent.act()

    assert first.action_key == "moveahead"
    assert second.action_key == "moveback"
    second_images = [
        part["image"]
        for message in policy.prompts[1].bound_messages()
        if isinstance(message["content"], list)
        for part in message["content"]
        if part["type"] == "image"
    ]
    assert second_images == ["image-0", "image-1"]
    assert second.prompt_messages == tuple(
        agent.policy_prompt_for_step(1).messages
    )


def test_navigation_agent_serializes_only_completed_turns() -> None:
    agent = AgentRuntime(
        policy=_RecordingPolicy(),
        action_space=NAVIGATION_ACTION_SPACE,
        prompt_template=_template(),
    )
    agent.reset(system_prompt="system")
    agent.observe(text="first <image>", image="image-0")
    action = agent.act()
    agent.observe(text="final <image>", image="image-1")

    messages = agent.completed_prompt().unbound_messages()
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert messages[-1]["content"] == action.response


def test_navigation_agent_generates_real_terminal_state_prefix() -> None:
    agent = AgentRuntime(
        policy=_RecordingPolicy(),
        action_space=NAVIGATION_ACTION_SPACE,
        prompt_template=_template(),
    )
    agent.reset(system_prompt="system")
    agent.observe(text="first <image>", image="image-0")
    agent.act()
    agent.observe(text="terminal <image>", image="image-1")

    prefix = agent.terminal_state_prefix()

    assert prefix.endswith("<|action_start|>")


def test_navigation_agent_keeps_policy_generated_response_in_history() -> None:
    class GeneratedPolicy(_RecordingPolicy):
        prompt_mode = "response"

        def select_action(self, prompt: AgentPrompt) -> PolicyDecision:
            self.prompts.append(prompt)
            return PolicyDecision(
                action_index=3,
                action_log_probs=tuple([-math.log(8.0)] * 8),
                response=(
                    "<think>Generated reasoning.</think><|latent_state|>"
                    "<|action_start|><|action_(3)|><|action_end|>"
                ),
                token_trace=PolicyTokenTrace(
                    token_ids=(10, 11, 12),
                    old_log_probs=(-0.2, -math.log(8.0), None),
                    loss_mask=(True, True, False),
                    token_roles=("reasoning", "action", "injected"),
                    action_token_ids=(8, 9, 10, 11, 12, 13, 14, 15),
                    reasoning_text="Generated reasoning.",
                    finish_reason="stop",
                ),
            )

    policy = GeneratedPolicy()
    agent = AgentRuntime(
        policy=policy,
        action_space=NAVIGATION_ACTION_SPACE,
        prompt_template=_template(),
    )
    agent.reset(system_prompt="system")
    agent.observe(text="first <image>", image="image-0")
    action = agent.act()
    agent.observe(text="second <image>", image="image-1")
    agent.act()

    assert policy.prompts[0].messages[-1]["content"] == "<think>"
    assert policy.prompts[1].messages[2]["content"] == action.response


def test_policy_decision_binds_action_and_old_log_prob_to_trace() -> None:
    trace = PolicyTokenTrace(
        token_ids=(10, 11),
        old_log_probs=(-0.3, None),
        loss_mask=(True, False),
        token_roles=("action", "injected"),
        action_token_ids=tuple(range(10, 18)),
    )

    with pytest.raises(ValueError, match="action log-prob does not match"):
        PolicyDecision(
            action_index=0,
            action_log_probs=tuple([-math.log(8.0)] * 8),
            token_trace=trace,
        )

    with pytest.raises(ValueError, match="action does not match"):
        PolicyDecision(
            action_index=1,
            action_log_probs=tuple([-math.log(8.0)] * 8),
            token_trace=PolicyTokenTrace(
                token_ids=(10, 11),
                old_log_probs=(-math.log(8.0), None),
                loss_mask=(True, False),
                token_roles=("action", "injected"),
                action_token_ids=tuple(range(10, 18)),
            ),
        )
