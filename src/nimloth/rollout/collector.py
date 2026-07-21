"""连接公共 Agent runner 与具体 environment session 的在线采集器。"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any

from nimloth.agent import Agent, EpisodeRunner, NimlothAgentPrompt
from nimloth.backbone.qwen25vl.policy import (
    QwenAgentPolicy,
    validate_agent_policy_protocol,
)
from nimloth.environment.navigation import (
    NAVIGATION_ACTION_SPACE,
    VAGENNavigationSession,
    instruction_from_observation,
)
from nimloth.rollout.schema import RolloutTrajectory
from nimloth.rollout.storage import save_trajectories


class VAGENNavigationRolloutCollector:
    """使用当前 Qwen policy 直接采集 VAGEN navigation trajectory。"""

    def __init__(
        self,
        qwen_model: Any,
        processor: Any,
        env_url: str,
        device: Any,
        *,
        seed_offset: int = 0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        eval_sets: tuple[str, ...] = ("base", "common_sense"),
        split: str = "eval",
    ) -> None:
        if not eval_sets:
            raise ValueError("rollout collector requires at least one eval_set")
        if split == "train" and any(
            not name.endswith("_train") for name in eval_sets
        ):
            raise ValueError(
                "training rollout requires *_train datasets; "
                f"got eval_sets={eval_sets}"
            )
        self._model = qwen_model
        self._processor = processor
        self._env_url = env_url.rstrip("/")
        self._device = device
        self._episode_counter = seed_offset
        self._temperature = temperature
        self._top_p = top_p
        self._eval_sets = eval_sets
        self._split = split
        self._client: Any | None = None
        self._policy: QwenAgentPolicy | None = None
        self._latent_token_count = 1
        if qwen_model is not None and processor is not None and device is not None:
            self.bind_policy(qwen_model, processor, device)

    def bind_policy(self, qwen_model: Any, processor: Any, device: Any) -> None:
        """绑定 trainer 已加载的 Qwen，避免采集器重复加载模型。"""

        validate_agent_policy_protocol(qwen_model.config)
        self._model = qwen_model
        self._processor = processor
        self._device = device
        self._latent_token_count = int(
            getattr(qwen_model.config, "nimloth_latent_token_count", 1)
        )
        self._policy = QwenAgentPolicy(
            model=qwen_model,
            processor=processor,
            device=device,
            temperature=self._temperature,
            top_p=self._top_p,
            latent_token_count=self._latent_token_count,
        )

    @property
    def client(self) -> Any:
        """首次采集时再连接 VAGEN，离线训练无需安装其运行依赖。"""

        if self._client is None:
            from vagen.server.client import BatchEnvClient

            self._client = BatchEnvClient(base_url=self._env_url, timeout=600)
        return self._client

    def collect(
        self,
        *,
        num_episodes: int,
        max_steps_per_episode: int = 20,
        output_dir: Path | None = None,
    ) -> list[RolloutTrajectory]:
        """采集若干 episode；失败 episode 会记录原因并整体丢弃。"""

        if self._policy is None:
            raise RuntimeError("rollout collector has no bound Agent policy")
        if num_episodes < 0:
            raise ValueError("num_episodes must be non-negative")

        target_dir = output_dir or Path(".")
        image_dir = target_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        self._log(
            rl_collect="start",
            num_episodes=num_episodes,
            output=str(target_dir),
        )

        trajectories: list[RolloutTrajectory] = []
        for episode_index in range(num_episodes):
            seed = self._episode_counter
            self._episode_counter += 1
            episode_id = f"rl_{seed:06d}"
            eval_set = self._eval_sets[episode_index % len(self._eval_sets)]
            started_at = time.monotonic()
            self._log(rl_ep=episode_index, id=episode_id, eval_set=eval_set)

            agent = Agent(
                policy=self._policy,
                action_space=NAVIGATION_ACTION_SPACE,
                prompt=NimlothAgentPrompt(
                    latent_token_count=self._latent_token_count,
                    action_count=len(NAVIGATION_ACTION_SPACE),
                ),
            )
            session = VAGENNavigationSession(
                client=self.client,
                episode_id=episode_id,
                eval_set=eval_set,
            )
            try:
                episode = EpisodeRunner(agent).run(
                    session,
                    seed=seed,
                    max_steps=max_steps_per_episode,
                )
                if not episode.actions:
                    raise RuntimeError("environment episode produced no actions")
                image_paths = self._save_images(
                    episode_id,
                    episode.observations,
                    image_dir,
                )
                action_keys = [action.action_key for action in episode.actions]
                observation_texts = [
                    observation.text for observation in episode.observations
                ]
                trajectory = RolloutTrajectory(
                    record_id=episode_id,
                    image_paths=image_paths,
                    action_indices=[
                        action.action_index for action in episode.actions
                    ],
                    action_names=action_keys,
                    action_log_probs=[
                        list(action.action_log_probs) for action in episode.actions
                    ],
                    nav_instruction=instruction_from_observation(
                        observation_texts[0]
                    ),
                    success=episode.success
                    or any(reward >= 10.0 for reward in episode.rewards),
                    reward=episode.reward,
                    split=self._split,
                    messages=agent.completed_messages(bind_images=False),
                    system_prompt=episode.system_prompt,
                    observation_texts=observation_texts,
                    policy_messages=[
                        [dict(message) for message in action.prompt_messages]
                        for action in episode.actions
                    ],
                    prompt_version=agent.prompt_version,
                    latent_token_count=self._latent_token_count,
                    sampling_temperature=self._temperature,
                    sampling_top_p=self._top_p,
                    action_space_id=NAVIGATION_ACTION_SPACE.identifier,
                    action_space_version=NAVIGATION_ACTION_SPACE.version,
                )
                trajectories.append(trajectory)
                self._log(
                    rl_ep=episode_index,
                    done=True,
                    steps=len(episode.actions),
                    success=trajectory.success,
                    reward=round(trajectory.reward, 2),
                    elapsed_s=round(time.monotonic() - started_at, 1),
                )
            except Exception as error:
                traceback.print_exc()
                self._log(
                    rl_ep=episode_index,
                    warning="discarding failed trajectory",
                    error=str(error),
                )

        jsonl_path = save_trajectories(trajectories, target_dir)
        self._log(
            rl_collect="done",
            trajectories=len(trajectories),
            jsonl_path=str(jsonl_path),
        )
        return trajectories

    @staticmethod
    def _save_images(
        episode_id: str,
        observations: tuple[Any, ...],
        image_dir: Path,
    ) -> list[str]:
        """按 step 顺序保存 observation，包含最后的下一状态图片。"""

        paths: list[str] = []
        for step_index, observation in enumerate(observations):
            path = image_dir / f"{episode_id}_step{step_index:02d}.png"
            observation.image.save(path)
            paths.append(str(path))
        return paths

    @staticmethod
    def _log(**payload: Any) -> None:
        print(json.dumps(payload, ensure_ascii=False), flush=True)
