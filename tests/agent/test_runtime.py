from __future__ import annotations

import math

from nimloth.agent import Agent, PolicyDecision
from nimloth.environment.navigation import NAVIGATION_ACTION_SPACE


class _RecordingPolicy:
    def __init__(self) -> None:
        self.prompts: list[list[dict]] = []

    def select_action(self, messages: list[dict]) -> PolicyDecision:
        self.prompts.append(messages)
        return PolicyDecision(
            action_index=len(self.prompts) - 1,
            action_log_probs=tuple([-math.log(8.0)] * 8),
        )


def test_navigation_agent_runs_real_history_through_one_policy() -> None:
    policy = _RecordingPolicy()
    agent = Agent(policy=policy, action_space=NAVIGATION_ACTION_SPACE)
    agent.reset(system_prompt="system")

    agent.observe(text="first <image>", image="image-0")
    first = agent.act()
    agent.observe(text="second <image>", image="image-1")
    second = agent.act()

    assert first.action_key == "moveahead"
    assert second.action_key == "moveback"
    second_images = [
        part["image"]
        for message in policy.prompts[1]
        if isinstance(message["content"], list)
        for part in message["content"]
        if part["type"] == "image"
    ]
    assert second_images == ["image-0", "image-1"]
    assert second.prompt_messages == tuple(
        agent.policy_messages_for_step(1, bind_images=False)
    )


def test_navigation_agent_serializes_only_completed_turns() -> None:
    agent = Agent(
        policy=_RecordingPolicy(),
        action_space=NAVIGATION_ACTION_SPACE,
    )
    agent.reset(system_prompt="system")
    agent.observe(text="first <image>", image="image-0")
    action = agent.act()
    agent.observe(text="final <image>", image="image-1")

    messages = agent.completed_messages()
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert messages[-1]["content"] == action.response
