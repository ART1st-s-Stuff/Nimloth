"""使用 AgentRuntime 和 VAGEN navigation environment 采集 rollout。"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nimloth.agent import (
    AgentAction,
    AgentEpisode,
    AgentPolicy,
    AgentRuntime,
    EpisodeRunner,
    PolicyState,
    create_prompt_template,
)
from nimloth.config.agent import AgentConfig
from nimloth.environment.navigation.action_space import NAVIGATION_ACTION_SPACE
from nimloth.environment.navigation.vagen import (
    NAVIGATION_REQUEST_TIMEOUT_SECONDS,
    VAGENNavigationSession,
    instruction_from_observation,
    navigation_environment_config,
    observation_image,
    observation_text,
    vagen_eval_nimloth_observation_text,
    vagen_eval_nimloth_system_prompt,
)
from nimloth.environment.common.session import EnvironmentObservation
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
        navigation_profile: str = "current",
        max_episode_attempts: int = 1,
    ) -> None:
        if not eval_sets:
            raise ValueError("rollout collector requires at least one eval_set")
        if split == "train" and any(not name.endswith("_train") for name in eval_sets):
            raise ValueError(
                "training rollout requires *_train datasets; "
                f"got eval_sets={eval_sets}"
            )
        if max_episode_attempts < 1:
            raise ValueError("max_episode_attempts must be positive")
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
        self._navigation_profile = navigation_profile
        self._max_episode_attempts = int(max_episode_attempts)

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
        max_episode_attempts: int | None = None,
        output_dir: Path | None = None,
        resume_existing: bool = False,
    ) -> list[RolloutTrajectory]:
        if self._policy is None:
            raise RuntimeError("rollout collector has no bound Agent")
        if num_episodes < 0:
            raise ValueError("num_episodes must be non-negative")
        episode_attempts = (
            self._max_episode_attempts
            if max_episode_attempts is None
            else int(max_episode_attempts)
        )
        if episode_attempts < 1:
            raise ValueError("max_episode_attempts must be positive")

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
            self._log(
                rl_ep=episode_index,
                id=episode_id,
                eval_set=eval_set,
                max_attempts=episode_attempts,
            )

            for attempt in range(1, episode_attempts + 1):
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
                    navigation_profile=self._navigation_profile,
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
                        attempt=attempt,
                        steps=len(episode.actions),
                        success=trajectory.success,
                        reward=round(trajectory.reward, 2),
                        elapsed_s=round(time.monotonic() - started_at, 1),
                    )
                    break
                except Exception as error:
                    traceback.print_exc()
                    self._log(
                        rl_ep=episode_index,
                        warning="trajectory attempt failed",
                        attempt=attempt,
                        max_attempts=episode_attempts,
                        error=str(error),
                    )
                    if attempt == episode_attempts:
                        raise RuntimeError(
                            "trajectory failed after bounded retries: "
                            f"id={episode_id}, eval_set={eval_set}, seed={seed}, "
                            f"attempts={episode_attempts}"
                        ) from error
                    self._log(
                        rl_ep=episode_index,
                        retrying=True,
                        id=episode_id,
                        eval_set=eval_set,
                        seed=seed,
                        next_attempt=attempt + 1,
                    )

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


@dataclass
class _BatchedEpisodeState:
    index: int
    episode_id: str
    eval_set: str
    seed: int
    system_prompt: str
    runtime: AgentRuntime
    observations: list[EnvironmentObservation] = field(default_factory=list)
    actions: list[AgentAction] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    success: bool = False
    done: bool = False


class VAGENBatchedNavigationRolloutCollector(VAGENNavigationRolloutCollector):
    """VAGEN active-env batching with identity-aligned PlannerPolicyHead calls."""

    def __init__(self, *args: Any, client: Any | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if client is not None:
            self._client = client

    def _batch_observation(
        self,
        raw_observation: Any,
        info: Any,
        *,
        initial: bool,
    ) -> EnvironmentObservation:
        return EnvironmentObservation(
            text=(
                vagen_eval_nimloth_observation_text(
                    raw_observation,
                    initial=initial,
                )
                if self._navigation_profile == "vagen_eval"
                else observation_text(raw_observation)
            ),
            image=observation_image(raw_observation),
            info=dict(info) if isinstance(info, dict) else {},
        )

    def _persist_completed_prefix(
        self,
        completed: dict[int, Any],
        target_dir: Path,
    ) -> list[Any]:
        prefix = []
        for index in range(len(completed)):
            trajectory = completed.get(index)
            if trajectory is None:
                break
            prefix.append(trajectory)
        if prefix:
            save_trajectories(prefix, target_dir)
        return prefix

    def collect(
        self,
        *,
        num_episodes: int,
        max_steps_per_episode: int = 20,
        max_episode_attempts: int | None = None,
        output_dir: Path | None = None,
        resume_existing: bool = False,
    ) -> list[RolloutTrajectory]:
        if self._policy is None:
            raise RuntimeError("rollout collector has no bound Agent")
        if num_episodes < 0:
            raise ValueError("num_episodes must be non-negative")
        if max_steps_per_episode < 1:
            raise ValueError("max_steps_per_episode must be positive")
        attempts = (
            self._max_episode_attempts
            if max_episode_attempts is None
            else int(max_episode_attempts)
        )
        if attempts != 1:
            raise ValueError(
                "batched rollout retries require a future per-request replay "
                "protocol; max_episode_attempts must currently be 1"
            )
        if resume_existing:
            raise ValueError(
                "batched rollout resume requires active-env identity replay and "
                "is not implemented"
            )
        target_dir = output_dir or Path(".")
        image_dir = target_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        if num_episodes == 0:
            save_trajectories([], target_dir)
            return []

        identities = tuple(
            (index, *self._next_episode_identity(index))
            for index in range(num_episodes)
        )
        env_ids = [episode_id for _index, episode_id, _eval_set, _seed in identities]
        open_env_ids: set[str] = set()
        completed: dict[int, RolloutTrajectory] = {}
        states: dict[str, _BatchedEpisodeState] = {}
        try:
            self.client.create_environments_batch(
                {
                    episode_id: navigation_environment_config(
                        eval_set,
                        profile=self._navigation_profile,
                    )
                    for _index, episode_id, eval_set, _seed in identities
                }
            )
            open_env_ids.update(env_ids)
            system_prompts = self.client.get_system_prompts_batch(env_ids)
            reset_rows = self.client.reset_batch(
                {
                    episode_id: seed
                    for _index, episode_id, _eval_set, seed in identities
                }
            )
            if set(system_prompts) != set(env_ids) or set(reset_rows) != set(env_ids):
                raise RuntimeError(
                    "VAGEN batch reset did not return every requested environment"
                )
            for index, episode_id, eval_set, seed in identities:
                system_prompt = str(system_prompts.get(episode_id, ""))
                if self._navigation_profile == "vagen_eval":
                    system_prompt = vagen_eval_nimloth_system_prompt()
                if not system_prompt:
                    raise RuntimeError(
                        f"environment {episode_id} returned an empty system prompt"
                    )
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
                runtime.reset(system_prompt=system_prompt)
                raw_observation, info = reset_rows[episode_id]
                observation = self._batch_observation(
                    raw_observation,
                    info,
                    initial=True,
                )
                runtime.observe(text=observation.text, image=observation.image)
                states[episode_id] = _BatchedEpisodeState(
                    index=index,
                    episode_id=episode_id,
                    eval_set=eval_set,
                    seed=seed,
                    system_prompt=system_prompt,
                    runtime=runtime,
                    observations=[observation],
                )

            for _step in range(max_steps_per_episode):
                active = tuple(
                    state
                    for state in states.values()
                    if not state.done and len(state.actions) < max_steps_per_episode
                )
                if not active:
                    break
                prompts = tuple(
                    state.runtime.pending_policy_prompt() for state in active
                )
                select_actions = getattr(self._policy, "select_actions", None)
                if select_actions is None:
                    raise RuntimeError("batched collector policy has no select_actions")
                decisions = tuple(select_actions(prompts))
                if len(decisions) != len(active):
                    raise RuntimeError(
                        "batched planner decisions do not align with active envs"
                    )
                actions = tuple(
                    state.runtime.record_policy_decision(prompt, decision)
                    for state, prompt, decision in zip(
                        active,
                        prompts,
                        decisions,
                        strict=True,
                    )
                )
                step_rows = self.client.step_batch(
                    {
                        state.episode_id: action.response
                        for state, action in zip(active, actions, strict=True)
                    }
                )
                if set(step_rows) != {state.episode_id for state in active}:
                    raise RuntimeError(
                        "VAGEN batch step did not return every active environment"
                    )
                finished: list[_BatchedEpisodeState] = []
                for state, action in zip(active, actions, strict=True):
                    raw_observation, reward, done, info = step_rows[state.episode_id]
                    info_dict = dict(info) if isinstance(info, dict) else {}
                    adjusted_reward = float(reward)
                    if not info_dict.get("last_action_success", True):
                        adjusted_reward -= 0.1
                    success = bool(info_dict.get("task_success", False)) or (
                        adjusted_reward >= 10.0
                    )
                    observation = self._batch_observation(
                        raw_observation,
                        info_dict,
                        initial=False,
                    )
                    state.actions.append(action)
                    state.rewards.append(adjusted_reward)
                    state.observations.append(observation)
                    state.success = state.success or success
                    state.done = bool(done)
                    state.runtime.observe(
                        text=observation.text,
                        image=observation.image,
                    )
                    if state.done or len(state.actions) >= max_steps_per_episode:
                        finished.append(state)

                if finished:
                    terminal_prompts = tuple(
                        state.runtime.terminal_policy_prompt() for state in finished
                    )
                    generate_states = getattr(self._policy, "generate_states", None)
                    if generate_states is None:
                        raise RuntimeError(
                            "batched collector policy has no generate_states"
                        )
                    terminal_states = tuple(generate_states(terminal_prompts))
                    if len(terminal_states) != len(finished):
                        raise RuntimeError(
                            "batched terminal states do not align with finished envs"
                        )
                    closed_now: list[str] = []
                    for state, terminal_state in zip(
                        finished,
                        terminal_states,
                        strict=True,
                    ):
                        if not state.actions:
                            raise RuntimeError(
                                f"environment {state.episode_id} produced no actions"
                            )
                        terminal_state = AgentRuntime.validate_terminal_state(
                            terminal_state
                        )
                        episode = AgentEpisode(
                            system_prompt=state.system_prompt,
                            observations=tuple(state.observations),
                            actions=tuple(state.actions),
                            rewards=tuple(state.rewards),
                            success=state.success,
                            done=state.done,
                            prompt_template=state.runtime.prompt_template_spec,
                            action_space_id=NAVIGATION_ACTION_SPACE.identifier,
                            action_space_version=NAVIGATION_ACTION_SPACE.version,
                        )
                        image_paths = self._save_images(
                            state.episode_id,
                            episode.observations,
                            image_dir,
                        )
                        completed[state.index] = trajectory_from_agent_episode(
                            episode,
                            record_id=state.episode_id,
                            image_paths=image_paths,
                            instruction=instruction_from_observation(
                                episode.observations[0].text
                            ),
                            split=self._split,
                            sampling_temperature=self._temperature,
                            sampling_top_p=self._top_p,
                            terminal_state=terminal_state,
                        )
                        closed_now.append(state.episode_id)
                    self.client.close_batch(closed_now)
                    open_env_ids.difference_update(closed_now)
                    self._persist_completed_prefix(completed, target_dir)

            if len(completed) != num_episodes:
                raise RuntimeError(
                    "batched rollout did not finalize every requested episode: "
                    f"{len(completed)} != {num_episodes}"
                )
            trajectories = [completed[index] for index in range(num_episodes)]
            save_trajectories(trajectories, target_dir)
            return trajectories
        finally:
            if open_env_ids:
                self.client.close_batch(sorted(open_env_ids))


__all__ = [
    "VAGENBatchedNavigationRolloutCollector",
    "VAGENNavigationRolloutCollector",
]
