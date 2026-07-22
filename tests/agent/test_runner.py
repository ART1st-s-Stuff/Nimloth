"""公共 Agent runner 与 rollout 适配器的真实调用链测试。"""

from __future__ import annotations

import math

from nimloth.agent import (
    AgentRuntime,
    AgentPrompt,
    EpisodeRunner,
    NimlothPromptTemplate,
    PolicyDecision,
)
from nimloth.environment import EnvironmentObservation, EnvironmentStep
from nimloth.environment.navigation import NAVIGATION_ACTION_SPACE
from nimloth.rollout import (
    RolloutTrajectory,
    trajectory_from_agent_episode,
    validate_rollout_trajectory,
)


class _SequencePolicy:
    def __init__(self, action_indices: tuple[int, ...]) -> None:
        self._action_indices = iter(action_indices)
        self.prompts: list[AgentPrompt] = []

    def reset_episode(self) -> None:
        self.prompts.clear()

    def select_action(self, prompt: AgentPrompt) -> PolicyDecision:
        self.prompts.append(prompt)
        return PolicyDecision(
            action_index=next(self._action_indices),
            action_log_probs=tuple([-math.log(8.0)] * 8),
        )


class _FakeNavigationSession:
    def __init__(self) -> None:
        self.closed = False
        self.responses: list[str] = []

    @property
    def action_space(self):
        return NAVIGATION_ACTION_SPACE

    @property
    def system_prompt(self) -> str:
        # 环境专属指令由 session 提供，Agent 模板不认识动作名称。
        return "Environment rule: moveahead means walk forward."

    def reset(self, *, seed: int) -> EnvironmentObservation:
        assert seed == 17
        return EnvironmentObservation("instruction <image>", "image-0")

    def step(self, *, action_index: int, response: str) -> EnvironmentStep:
        self.responses.append(response)
        step_index = len(self.responses)
        return EnvironmentStep(
            observation=EnvironmentObservation(
                f"feedback {step_index} <image>",
                f"image-{step_index}",
            ),
            reward=float(step_index),
            done=step_index == 2,
            success=step_index == 2,
        )

    def close(self) -> None:
        self.closed = True


def _run_episode():
    policy = _SequencePolicy((0, 4))
    agent = AgentRuntime(
        policy=policy,
        action_space=NAVIGATION_ACTION_SPACE,
        prompt_template=NimlothPromptTemplate(
            latent_token_count=1,
            action_count=len(NAVIGATION_ACTION_SPACE),
        ),
    )
    session = _FakeNavigationSession()
    episode = EpisodeRunner(agent).run(session, seed=17, max_steps=3)
    return episode, policy, session


def test_episode_runner_uses_environment_prompt_and_closes_session() -> None:
    episode, policy, session = _run_episode()

    assert session.closed
    assert episode.done and episode.success
    assert episode.reward == 3.0
    assert episode.action_space_id == "navigation"
    assert episode.action_space_version == 1
    assert episode.transcript.observation_texts == (
        "instruction <image>",
        "feedback 1 <image>",
        "feedback 2 <image>",
    )
    assert policy.prompts[0].messages[0] == {
        "role": "system",
        "content": "Environment rule: moveahead means walk forward.",
    }
    assert session.responses == [
        episode.actions[0].response,
        episode.actions[1].response,
    ]


def test_agent_episode_is_the_only_input_needed_to_build_rollout() -> None:
    episode, _, _ = _run_episode()
    trajectory = trajectory_from_agent_episode(
        episode,
        record_id="episode-17",
        image_paths=["saved-0.png", "saved-1.png", "saved-2.png"],
        instruction="walk forward",
        split="train",
        sampling_temperature=0.7,
        sampling_top_p=0.95,
    )

    validate_rollout_trajectory(trajectory)
    restored = RolloutTrajectory.from_record(trajectory.to_record())
    validate_rollout_trajectory(restored)

    assert restored.prompt_template_spec == episode.prompt_template
    assert restored.instruction == "walk forward"
    assert restored.action_names == ["moveahead", "rotateright"]
    assert restored.policy_messages == [
        action.policy_prompt.unbound_messages() for action in episode.actions
    ]


def test_legacy_navigation_instruction_key_is_migrated() -> None:
    episode, _, _ = _run_episode()
    trajectory = trajectory_from_agent_episode(
        episode,
        record_id="legacy-instruction",
        image_paths=["saved-0.png", "saved-1.png", "saved-2.png"],
        instruction="walk forward",
        split="train",
        sampling_temperature=1.0,
        sampling_top_p=1.0,
    )
    record = trajectory.to_record()
    record.pop("instruction")

    assert RolloutTrajectory.from_record(record).instruction == "walk forward"
