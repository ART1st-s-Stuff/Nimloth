"""真正执行 Agent 与 environment 交互的 episode runner。"""

from __future__ import annotations

from dataclasses import dataclass

from nimloth.agent.runtime import AgentAction, AgentRuntime
from nimloth.agent.template import PromptTemplateSpec
from nimloth.agent.transcript import AgentTranscript
from nimloth.environment.common.session import (
    EnvironmentObservation,
    EnvironmentSession,
)


@dataclass(frozen=True)
class AgentEpisode:
    """一次内存中的完整 Agent episode；持久化由 rollout 模块负责。"""

    system_prompt: str
    observations: tuple[EnvironmentObservation, ...]
    actions: tuple[AgentAction, ...]
    rewards: tuple[float, ...]
    success: bool
    done: bool
    prompt_template: PromptTemplateSpec
    action_space_id: str
    action_space_version: int

    def __post_init__(self) -> None:
        if len(self.observations) != len(self.actions) + 1:
            raise ValueError(
                "AgentEpisode requires one final observation after all actions"
            )
        if len(self.rewards) != len(self.actions):
            raise ValueError("AgentEpisode reward/action count mismatch")

    @property
    def reward(self) -> float:
        return sum(self.rewards)

    @property
    def transcript(self) -> AgentTranscript:
        """把执行结果转换为模型无关 transcript。"""

        return AgentTranscript(
            system_prompt=self.system_prompt,
            observation_texts=tuple(
                observation.text for observation in self.observations
            ),
            observation_images=tuple(
                observation.image for observation in self.observations
            ),
            action_indices=tuple(
                action.action_index for action in self.actions
            ),
            assistant_responses=tuple(
                action.response for action in self.actions
            ),
        )


class EpisodeRunner:
    """在一个 environment session 中运行一个 Agent。"""

    def __init__(self, agent: AgentRuntime) -> None:
        self._agent = agent

    def run(
        self,
        session: EnvironmentSession,
        *,
        seed: int,
        max_steps: int,
    ) -> AgentEpisode:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if session.action_space != self._agent.action_space:
            raise ValueError(
                "Agent action space does not match environment session: "
                f"{self._agent.action_space.identifier}@"
                f"{self._agent.action_space.version} != "
                f"{session.action_space.identifier}@{session.action_space.version}"
            )

        observations: list[EnvironmentObservation] = []
        actions: list[AgentAction] = []
        rewards: list[float] = []
        success = False
        done = False
        try:
            observation = session.reset(seed=seed)
            self._agent.reset(system_prompt=session.system_prompt)
            for _ in range(max_steps):
                observations.append(observation)
                self._agent.observe(text=observation.text, image=observation.image)
                action = self._agent.act()
                actions.append(action)

                result = session.step(
                    action_index=action.action_index,
                    response=action.response,
                )
                rewards.append(result.reward)
                success = success or result.success
                done = result.done
                observation = result.observation
                if done:
                    break

            # 最后一帧用于 s_{t+1}，不会产生新的动作。
            observations.append(observation)
            self._agent.observe(text=observation.text, image=observation.image)
            return AgentEpisode(
                system_prompt=session.system_prompt,
                observations=tuple(observations),
                actions=tuple(actions),
                rewards=tuple(rewards),
                success=success,
                done=done,
                prompt_template=self._agent.prompt_template_spec,
                action_space_id=self._agent.action_space.identifier,
                action_space_version=self._agent.action_space.version,
            )
        finally:
            session.close()
