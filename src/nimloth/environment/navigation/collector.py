"""使用 AgentRuntime 和 VAGEN navigation environment 采集 rollout。"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any

from nimloth.agent import (
    AgentPolicy,
    AgentRuntime,
    EpisodeRunner,
    create_prompt_template,
)
from nimloth.config.agent import AgentConfig
from nimloth.environment.navigation.action_space import NAVIGATION_ACTION_SPACE
from nimloth.environment.navigation.vagen import (
    NAVIGATION_REQUEST_TIMEOUT_SECONDS,
    VAGENNavigationSession,
    instruction_from_observation,
)
from nimloth.rollout.from_agent import trajectory_from_agent_episode
from nimloth.rollout.schema import RolloutTrajectory
from nimloth.rollout.storage import load_trajectories, save_trajectories


class VAGENNavigationRolloutCollector:
    """使用当前 Agent policy 采集 VAGEN navigation trajectory。"""

    def __init__(
        self,
        policy: AgentPolicy | None,
        env_url: str,
        *,
        seed_offset: int = 0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        eval_sets: tuple[str, ...] = ("base", "common_sense"),
        split: str = "eval",
        agent_config: AgentConfig | None = None,
        latent_token_count: int = 1,
        seed_per_eval_set: bool = False,
    ) -> None:
        if not eval_sets:
            raise ValueError("rollout collector requires at least one eval_set")
        if split == "train" and any(not name.endswith("_train") for name in eval_sets):
            raise ValueError(
                "training rollout requires *_train datasets; "
                f"got eval_sets={eval_sets}"
            )
        self._env_url = env_url.rstrip("/")
        self._episode_counter = seed_offset
        self._seed_per_eval_set = bool(seed_per_eval_set)
        self._eval_set_counters = {
            eval_set: int(seed_offset) for eval_set in eval_sets
        }
        self._temperature = temperature
        self._top_p = top_p
        self._eval_sets = eval_sets
        self._split = split
        self._agent_config = agent_config or AgentConfig()
        self._client: Any | None = None
        self._policy = policy
        self._latent_token_count = int(latent_token_count)

    def _next_episode_identity(self, episode_index: int) -> tuple[str, str, int]:
        eval_set = self._eval_sets[episode_index % len(self._eval_sets)]
        if self._seed_per_eval_set:
            seed = self._eval_set_counters[eval_set]
            self._eval_set_counters[eval_set] += 1
            episode_id = f"rl_{eval_set}_{seed:06d}"
        else:
            seed = self._episode_counter
            self._episode_counter += 1
            episode_id = f"rl_{seed:06d}"
        return episode_id, eval_set, seed

    def bind_policy(
        self,
        policy: AgentPolicy,
        *,
        latent_token_count: int,
    ) -> None:
        """绑定在线行为 policy；collector 不接触神经网络模型结构。"""

        self._policy = policy
        self._latent_token_count = int(latent_token_count)

    @property
    def client(self) -> Any:
        if self._client is None:
            from vagen.server.client import BatchEnvClient

            self._client = BatchEnvClient(
                base_url=self._env_url,
                timeout=NAVIGATION_REQUEST_TIMEOUT_SECONDS,
            )
        return self._client

    def collect(
        self,
        *,
        num_episodes: int,
        max_steps_per_episode: int = 20,
        output_dir: Path | None = None,
        resume_existing: bool = False,
    ) -> list[RolloutTrajectory]:
        if self._policy is None:
            raise RuntimeError("rollout collector has no bound Agent")
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

        trajectories = (
            self._load_resume_prefix(target_dir, num_episodes=num_episodes)
            if resume_existing
            else []
        )
        if trajectories:
            self._log(
                rl_collect="resume",
                trajectories=len(trajectories),
                next_episode=len(trajectories),
            )
        for episode_index in range(len(trajectories), num_episodes):
            episode_id, eval_set, seed = self._next_episode_identity(episode_index)
            started_at = time.monotonic()
            self._log(rl_ep=episode_index, id=episode_id, eval_set=eval_set)

            runtime = AgentRuntime(
                policy=self._policy,
                action_space=NAVIGATION_ACTION_SPACE,
                prompt_template=create_prompt_template(
                    self._agent_config.prompt_spec(
                        latent_token_count=self._latent_token_count,
                    ),
                    action_count=len(NAVIGATION_ACTION_SPACE),
                ),
            )
            session = VAGENNavigationSession(
                client=self.client,
                episode_id=episode_id,
                eval_set=eval_set,
            )
            try:
                episode = EpisodeRunner(runtime).run(
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
                observation_texts = [
                    observation.text for observation in episode.observations
                ]
                trajectory = trajectory_from_agent_episode(
                    episode,
                    record_id=episode_id,
                    image_paths=image_paths,
                    instruction=instruction_from_observation(observation_texts[0]),
                    split=self._split,
                    sampling_temperature=self._temperature,
                    sampling_top_p=self._top_p,
                    terminal_state=runtime.terminal_state(),
                )
                trajectories.append(trajectory)
                # 每个真实episode完成后立即原子重写短JSONL前缀，确保抢占不会丢失
                # 已完成样本，也不会留下半行JSON。
                save_trajectories(trajectories, target_dir)
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
                if resume_existing:
                    raise

        jsonl_path = save_trajectories(trajectories, target_dir)
        self._log(
            rl_collect="done",
            trajectories=len(trajectories),
            jsonl_path=str(jsonl_path),
        )
        return trajectories

    def _load_resume_prefix(
        self,
        target_dir: Path,
        *,
        num_episodes: int,
    ) -> list[RolloutTrajectory]:
        jsonl_path = target_dir / "trajectories.jsonl"
        if not jsonl_path.exists():
            return []
        trajectories = load_trajectories(jsonl_path)
        if len(trajectories) > num_episodes:
            raise ValueError(
                "resume trajectory count exceeds requested episodes: "
                f"{len(trajectories)} > {num_episodes}"
            )
        for episode_index, trajectory in enumerate(trajectories):
            expected_id, _eval_set, _seed = self._next_episode_identity(
                episode_index
            )
            if trajectory.record_id != expected_id:
                raise ValueError(
                    "resume trajectories must be the contiguous requested seed "
                    f"prefix: index={episode_index}, expected={expected_id!r}, "
                    f"actual={trajectory.record_id!r}"
                )
            if trajectory.split != self._split:
                raise ValueError(
                    "resume trajectory split does not match collector: "
                    f"{trajectory.split!r} != {self._split!r}"
                )
        return trajectories

    @staticmethod
    def _save_images(
        episode_id: str,
        observations: tuple[Any, ...],
        image_dir: Path,
    ) -> list[str]:
        paths: list[str] = []
        for step_index, observation in enumerate(observations):
            path = image_dir / f"{episode_id}_step{step_index:02d}.png"
            observation.image.save(path)
            paths.append(str(path))
        return paths

    @staticmethod
    def _log(**payload: Any) -> None:
        print(json.dumps(payload, ensure_ascii=False), flush=True)


__all__ = ["VAGENNavigationRolloutCollector"]
