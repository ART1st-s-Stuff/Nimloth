"""Online rollout collection: Qwen policy interacting with VAGEN environments.

The rollout collector runs the Qwen policy in the VAGEN navigation environment,
collecting trajectories that include per-frame images, taken actions, and sparse rewards.
Each trajectory is later encoded into WM latent states by the trainer.
"""

from __future__ import annotations

import gzip
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from nimloth.agent import (
    PROMPT_VERSION,
    AgentTranscript,
    NavigationAgent,
    NimlothAgentPrompt,
    instruction_from_observation,
    navigation_action_name,
    validate_action_log_probs,
)
from nimloth.backbone.qwen25vl.policy import QwenNavigationPolicy


def validate_rl_policy_protocol(model_config: Any) -> None:
    """Fail fast unless the policy matches the implemented k=1 inject runtime."""

    latent_count = int(getattr(model_config, "nimloth_latent_token_count", 1))
    query_mode = getattr(model_config, "nimloth_latent_query_mode", None)
    if latent_count != 1 or query_mode != "inject":
        raise ValueError(
            "RL action/encoding runtime currently requires a k=1 inject checkpoint; "
            f"got latent_token_count={latent_count}, latent_query_mode={query_mode!r}"
        )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RolloutTrajectory:
    """One episode collected by the Qwen policy in the environment."""

    record_id: str
    image_paths: list[str] = field(default_factory=list)
    """image_paths[t] = observation *before* taking action t."""
    action_indices: list[int] = field(default_factory=list)
    """action_indices[t] = action taken at step t (0..7)."""
    action_names: list[str] = field(default_factory=list)
    """action_names[t] = canonical readable action name for action_indices[t]."""
    action_log_probs: list[list[float]] = field(default_factory=list)
    """action_log_probs[t] = [log_prob(a0), ..., log_prob(a7)] at step t (log-softmax)."""
    nav_instruction: str = ""
    """Navigation instruction from env server."""
    success: bool = False
    reward: float = 0.0
    split: str = "train"
    messages: list[dict[str, Any]] = field(default_factory=list)
    """Full conversation history (system, user, assistant turns)."""
    system_prompt: str = ""
    observation_texts: list[str] = field(default_factory=list)
    policy_messages: list[list[dict[str, Any]]] = field(default_factory=list)
    prompt_version: str = PROMPT_VERSION
    latent_token_count: int = 1
    sampling_temperature: float = 1.0
    sampling_top_p: float = 1.0

    @property
    def num_steps(self) -> int:
        return len(self.action_indices)

    def build_policy_messages(
        self,
        step_index: int,
        *,
        bind_images: bool,
    ) -> list[dict[str, Any]]:
        """Rebuild one policy query from the shared Agent transcript contract."""

        transcript = AgentTranscript(
            system_prompt=self.system_prompt,
            observation_texts=tuple(self.observation_texts),
            observation_images=tuple(self.image_paths),
            action_indices=tuple(self.action_indices),
        )
        prompt = NimlothAgentPrompt(latent_token_count=self.latent_token_count)
        return prompt.build_policy_messages(
            transcript.policy_prefix(step_index),
            bind_images=bind_images,
        )

    def build_completed_messages(self, *, bind_images: bool) -> list[dict[str, Any]]:
        """Rebuild completed action turns from the shared Agent contract."""

        transcript = AgentTranscript(
            system_prompt=self.system_prompt,
            observation_texts=tuple(self.observation_texts),
            observation_images=tuple(self.image_paths),
            action_indices=tuple(self.action_indices),
        )
        prompt = NimlothAgentPrompt(latent_token_count=self.latent_token_count)
        return prompt.build_supervised_messages(transcript, bind_images=bind_images)

    def to_record(self) -> dict[str, Any]:
        """Serialize to the Nimloth JSONL record format."""
        return {
            "id": self.record_id,
            "split": self.split,
            "success": self.success,
            "reward": self.reward,
            "messages": self.messages,
            "image_paths": self.image_paths,
            "action_indices": self.action_indices,
            "action_names": self.action_names,
            # JSON has no infinity literal.  ``null`` represents an action
            # removed by greedy/top-p sampling and round-trips back to -inf.
            "action_log_probs": [
                [None if value == float("-inf") else value for value in row]
                for row in self.action_log_probs
            ],
            "nav_instruction": self.nav_instruction,
            "system_prompt": self.system_prompt,
            "observation_texts": self.observation_texts,
            "policy_messages": self.policy_messages,
            "prompt_version": self.prompt_version,
            "latent_token_count": self.latent_token_count,
            "sampling_temperature": self.sampling_temperature,
            "sampling_top_p": self.sampling_top_p,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "RolloutTrajectory":
        return cls(
            record_id=str(record.get("id", "")),
            image_paths=list(record.get("image_paths", [])),
            action_indices=list(record.get("action_indices", [])),
            action_names=list(record.get("action_names", [])),
            action_log_probs=[
                [float("-inf") if value is None else float(value) for value in row]
                for row in record.get("action_log_probs", [])
            ],
            nav_instruction=str(record.get("nav_instruction", "")),
            success=bool(record.get("success", False)),
            reward=float(record.get("reward", 0.0)),
            split=str(record.get("split", "train")),
            messages=list(record.get("messages", [])),
            system_prompt=str(record.get("system_prompt", "")),
            observation_texts=list(record.get("observation_texts", [])),
            policy_messages=list(record.get("policy_messages", [])),
            prompt_version=str(record.get("prompt_version", "")),
            latent_token_count=int(record.get("latent_token_count", 1)),
            sampling_temperature=float(record.get("sampling_temperature", 1.0)),
            sampling_top_p=float(record.get("sampling_top_p", 1.0)),
        )


def validate_rollout_trajectory(trajectory: RolloutTrajectory) -> None:
    """Validate one structured Agent trajectory before writing or training."""

    prefix = f"trajectory {trajectory.record_id}"
    if len(trajectory.image_paths) != trajectory.num_steps + 1:
        raise ValueError(
            f"{prefix}: images={len(trajectory.image_paths)} "
            f"but actions={trajectory.num_steps}"
        )
    if len(trajectory.observation_texts) != trajectory.num_steps + 1:
        raise ValueError(
            f"{prefix}: observations={len(trajectory.observation_texts)} "
            f"but actions={trajectory.num_steps}"
        )
    if len(trajectory.action_names) != trajectory.num_steps:
        raise ValueError(
            f"{prefix}: action_names={len(trajectory.action_names)} "
            f"but actions={trajectory.num_steps}"
        )
    expected_names = [
        navigation_action_name(index) for index in trajectory.action_indices
    ]
    if trajectory.action_names != expected_names:
        raise ValueError(f"{prefix}: action names do not match action indices")
    if len(trajectory.action_log_probs) != trajectory.num_steps:
        raise ValueError(
            f"{prefix}: action_log_probs={len(trajectory.action_log_probs)} "
            f"but actions={trajectory.num_steps}"
        )
    for step, (action_index, log_probs) in enumerate(
        zip(
            trajectory.action_indices,
            trajectory.action_log_probs,
            strict=True,
        )
    ):
        try:
            validate_action_log_probs(action_index, log_probs)
        except ValueError as error:
            raise ValueError(
                f"{prefix} step {step} has invalid action probabilities: {error}"
            ) from error
    if len(trajectory.policy_messages) != trajectory.num_steps:
        raise ValueError(
            f"{prefix}: policy_messages={len(trajectory.policy_messages)} "
            f"but actions={trajectory.num_steps}"
        )
    if not trajectory.system_prompt:
        raise ValueError(f"{prefix} has no system prompt")
    if trajectory.prompt_version != PROMPT_VERSION:
        raise ValueError(
            f"{prefix} uses unsupported prompt version {trajectory.prompt_version!r}"
        )
    if not trajectory.nav_instruction:
        raise ValueError(f"{prefix} has no navigation instruction")
    if trajectory.sampling_temperature < 0.0:
        raise ValueError(f"{prefix} has a negative sampling temperature")
    if not 0.0 < trajectory.sampling_top_p <= 1.0:
        raise ValueError(f"{prefix} has sampling_top_p outside (0, 1]")
    for step, policy_messages in enumerate(trajectory.policy_messages):
        expected_messages = trajectory.build_policy_messages(
            step,
            bind_images=False,
        )
        if policy_messages != expected_messages:
            raise ValueError(
                f"{prefix} step {step} policy prompt does not match the "
                "shared Agent template"
            )
    expected_completed = trajectory.build_completed_messages(bind_images=False)
    if trajectory.messages != expected_completed:
        raise ValueError(
            f"{prefix} completed messages do not match the shared Agent template"
        )


# ---------------------------------------------------------------------------
# Collector interface
# ---------------------------------------------------------------------------


class RolloutCollector(Protocol):
    """Interface for collecting trajectories from an environment."""

    def collect(
        self,
        *,
        num_episodes: int,
        max_steps_per_episode: int = 20,
        output_dir: Path | None = None,
    ) -> list[RolloutTrajectory]:
        """Run ``num_episodes`` episodes and return collected trajectories."""
        ...


# ---------------------------------------------------------------------------
# VAGEN-backed collector (delegates to VAGEN's trainer.val_only rollout)
# ---------------------------------------------------------------------------


class VAGENRolloutCollector:
    """Collect trajectories by running VAGEN in validation-only mode.

    Legacy placeholder — use ``EnvRolloutCollector`` for direct env interaction.
    """

    def __init__(
        self,
        vagen_config_path: Path,
        vagen_checkpoint_dir: Path,
        output_root: Path,
    ) -> None:
        self._vagen_config_path = vagen_config_path
        self._vagen_checkpoint_dir = vagen_checkpoint_dir
        self._output_root = output_root

    def collect(self, *, num_episodes, max_steps_per_episode=20, output_dir=None):
        raise NotImplementedError(
            "VAGENRolloutCollector is not implemented. Use EnvRolloutCollector with --env-url."
        )


# ---------------------------------------------------------------------------
# Env-backed collector (direct env server interaction using trainer's Qwen)
# ---------------------------------------------------------------------------

class EnvRolloutCollector:
    """Collect trajectories by running Qwen policy against the VAGEN env server.

    Reuses the trainer's Qwen model (no subprocess/model-reloading).
    Each ``collect()`` call creates envs on the server, runs Qwen-based
    the configured action sampling, and returns ``RolloutTrajectory`` objects.
    """

    def __init__(
        self,
        qwen_model,
        processor,
        env_url: str,
        device,
        seed_offset: int = 0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        eval_sets: tuple[str, ...] = ("base", "common_sense"),
        split: str = "eval",
    ) -> None:
        if not eval_sets:
            raise ValueError("EnvRolloutCollector requires at least one eval_set")
        if split == "train" and any(not name.endswith("_train") for name in eval_sets):
            raise ValueError(
                "training rollout requires *_train datasets; "
                f"got eval_sets={eval_sets}"
            )
        self._model = qwen_model
        self._processor = processor
        self._env_url = env_url.rstrip("/")
        self._device = device
        self._ep_counter = seed_offset
        self._client = None  # lazy init
        self._temperature = temperature
        self._top_p = top_p
        self._eval_sets = eval_sets
        self._split = split
        self._agent_policy: QwenNavigationPolicy | None = None
        self._latent_token_count = 1
        if qwen_model is not None and processor is not None and device is not None:
            self.bind_policy(qwen_model, processor, device)

    def bind_policy(self, qwen_model, processor, device) -> None:
        """Attach the loaded Qwen policy used by the shared Agent runtime."""

        self._model = qwen_model
        self._processor = processor
        self._device = device
        self._latent_token_count = int(
            getattr(qwen_model.config, "nimloth_latent_token_count", 1)
        )
        self._agent_policy = QwenNavigationPolicy(
            model=qwen_model,
            processor=processor,
            device=device,
            temperature=self._temperature,
            top_p=self._top_p,
            latent_token_count=self._latent_token_count,
        )

    @property
    def client(self):
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

        if self._agent_policy is None:
            raise RuntimeError("EnvRolloutCollector has no bound Agent policy")
        out_dir = output_dir or Path(".")
        out_dir.mkdir(parents=True, exist_ok=True)
        img_dir = out_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        print(json.dumps({"rl_collect": "start", "num_episodes": num_episodes,
                          "output": str(out_dir)}), flush=True)

        # --- lazy-init client -------------------------------------------------
        if self._client is None:
            print(json.dumps({"rl_collect": "init_client", "url": self._env_url}), flush=True)
            try:
                from vagen.server.client import BatchEnvClient
                self._client = BatchEnvClient(base_url=self._env_url, timeout=600)
                print(json.dumps({"rl_collect": "client_created"}), flush=True)
            except Exception:
                import traceback
                traceback.print_exc()
                print(json.dumps({"rl_collect": "client_init_failed"}), flush=True)
                raise

        trajectories: list[RolloutTrajectory] = []

        for ep_i in range(num_episodes):
            seed = self._ep_counter
            ep_id = f"rl_{seed:06d}"
            self._ep_counter += 1
            t0 = time.time()
            eval_set = self._eval_sets[ep_i % len(self._eval_sets)]

            print(json.dumps({"rl_ep": ep_i, "id": ep_id, "eval_set": eval_set}), flush=True)

            env_config = {
                "env_name": "navigation",
                "env_config": {
                    "render_mode": "vision",
                    "prompt_format": "nimloth",
                    "use_state_reward": False,
                    "eval_set": eval_set,
                    "max_actions_per_step": 1,
                    "max_action_penalty": -0.1,
                    "format_reward": 0.0,  # Nimloth: no format reward (we control the format)
                    "success_threshold": 1.5,
                    "step_length": 0.5,
                    "grounding_reward_weight": 0.5,
                    "worldmodeling_reward_weight": 0.5,
                    "gpu_device": 0,
                },
            }

            # --- create env on server ---
            print(json.dumps({"rl_ep": ep_i, "step": "create_env"}), flush=True)
            try:
                self._client.create_environments_batch({ep_id: env_config})
                print(json.dumps({"rl_ep": ep_i, "step": "create_env_done"}), flush=True)
            except Exception:
                import traceback
                traceback.print_exc()
                print(json.dumps({"rl_ep": ep_i, "step": "create_env_failed",
                                  "error": str(traceback.format_exc())}), flush=True)
                continue

            # --- get system prompt ---
            print(json.dumps({"rl_ep": ep_i, "step": "get_prompt"}), flush=True)
            try:
                prompts = self._client.get_system_prompts_batch([ep_id])
                system_prompt = str(prompts.get(ep_id, ""))
                if not system_prompt:
                    raise RuntimeError(f"environment {ep_id} returned an empty system prompt")
                print(json.dumps({"rl_ep": ep_i, "step": "get_prompt_done"}), flush=True)
            except Exception:
                import traceback
                traceback.print_exc()
                try:
                    self._client.close_batch([ep_id])
                except Exception:
                    pass
                continue

            # --- reset ---
            print(json.dumps({"rl_ep": ep_i, "step": "reset", "seed": seed}), flush=True)
            try:
                results = self._client.reset_batch({ep_id: seed})
                obs, info = results[ep_id]
                print(json.dumps({"rl_ep": ep_i, "step": "reset_done"}), flush=True)
            except Exception:
                import traceback
                traceback.print_exc()
                print(json.dumps({"rl_ep": ep_i, "step": "reset_failed",
                                  "error": str(traceback.format_exc())}), flush=True)
                try:
                    self._client.close_batch([ep_id])
                except Exception:
                    pass
                continue

            action_names: list[str] = []
            action_indices: list[int] = []
            action_log_probs: list[list[float]] = []
            image_paths: list[str] = []
            observation_texts: list[str] = []
            policy_messages: list[list[dict[str, Any]]] = []
            done = False
            step_rewards: list[float] = []
            success = False
            trajectory_failed = False
            agent = NavigationAgent(
                policy=self._agent_policy,
                prompt=NimlothAgentPrompt(
                    latent_token_count=self._latent_token_count
                ),
            )
            agent.reset(system_prompt=system_prompt)

            for step in range(max_steps_per_episode):
                print(json.dumps({
                    "rl_ep": ep_i,
                    "step": f"action_{step}",
                    "history_len": len(action_names),
                }), flush=True)

                # --- image ---
                try:
                    img = _obs_to_pil(obs)
                    observation_text = _obs_to_text(obs)
                except Exception:
                    import traceback
                    traceback.print_exc()
                    trajectory_failed = True
                    break

                # --- save image ---
                img_path = img_dir / f"{ep_id}_step{step:02d}.png"
                img.save(str(img_path))
                image_paths.append(str(img_path))

                # --- qwen action selection ---
                try:
                    agent.observe(text=observation_text, image=img)
                    agent_action = agent.act()
                    print(json.dumps({"rl_ep": ep_i, "action_selected": agent_action.action_name,
                                      "action_idx": agent_action.action_index}), flush=True)
                except Exception:
                    import traceback
                    traceback.print_exc()
                    trajectory_failed = True
                    break

                # --- env step ---
                try:
                    step_results = self._client.step_batch({ep_id: agent_action.response})
                    obs, r, done, info = step_results[ep_id]
                    # Apply failure penalty if action didn't execute
                    action_ok = (
                        info.get("last_action_success", True)
                        if isinstance(info, dict)
                        else True
                    )
                    if not action_ok:
                        r = float(r) - 0.1  # failure_penalty
                    step_rewards.append(float(r))
                    success = success or bool(
                        info.get("task_success", False) if isinstance(info, dict) else False
                    )
                    print(json.dumps({"rl_ep": ep_i, "env_step_done": True, "done": done,
                                      "step_reward": r, "action_ok": action_ok}), flush=True)
                except Exception:
                    import traceback
                    traceback.print_exc()
                    trajectory_failed = True
                    break

                observation_texts.append(observation_text)
                action_names.append(agent_action.action_name)
                action_indices.append(agent_action.action_index)
                action_log_probs.append(list(agent_action.action_log_probs))
                policy_messages.append([dict(message) for message in agent_action.prompt_messages])

                if done:
                    break

            # Save final observation (so image_paths has len = num_steps + 1)
            try:
                final_img = _obs_to_pil(obs)
                final_observation_text = _obs_to_text(obs)
                img_path = img_dir / f"{ep_id}_step{len(action_names):02d}.png"
                final_img.save(str(img_path))
                image_paths.append(str(img_path))
                observation_texts.append(final_observation_text)
                agent.observe(text=final_observation_text, image=final_img)
            except Exception:
                trajectory_failed = True

            # --- compute success from per-step rewards ---
            reward = sum(step_rewards)
            success = success or any(r >= 10.0 for r in step_rewards)

            # --- close env ---
            try:
                self._client.close_batch([ep_id])
            except Exception:
                pass

            if (
                trajectory_failed
                or not action_names
                or len(image_paths) != len(action_names) + 1
                or len(observation_texts) != len(action_names) + 1
                or len(policy_messages) != len(action_names)
            ):
                print(json.dumps({
                    "rl_ep": ep_i,
                    "warning": "discarding incomplete trajectory",
                    "num_actions": len(action_names),
                    "num_images": len(image_paths),
                    "num_observations": len(observation_texts),
                    "trajectory_failed": trajectory_failed,
                }), flush=True)
                continue

            trajectories.append(RolloutTrajectory(
                record_id=ep_id,
                image_paths=image_paths,
                action_indices=action_indices,
                action_names=list(action_names),
                action_log_probs=action_log_probs,
                nav_instruction=instruction_from_observation(observation_texts[0]),
                success=success,
                reward=reward,
                split=self._split,
                messages=agent.completed_messages(bind_images=False),
                system_prompt=system_prompt,
                observation_texts=observation_texts,
                policy_messages=policy_messages,
                prompt_version=agent.prompt_version,
                latent_token_count=self._latent_token_count,
                sampling_temperature=self._temperature,
                sampling_top_p=self._top_p,
            ))

            elapsed = time.time() - t0
            print(json.dumps({
                "rl_ep": ep_i, "done": True,
                "steps": len(action_names),
                "success": success,
                "reward": round(reward, 2),
                "elapsed_s": round(elapsed, 1),
            }), flush=True)

        jsonl_path = save_trajectories(trajectories, out_dir)
        print(json.dumps({
            "rl_collect": "done",
            "trajectories": len(trajectories),
            "jsonl_path": str(jsonl_path),
        }), flush=True)
        return trajectories


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _obs_to_text(obs: Any) -> str:
    """Return the environment-authored observation prompt with its image slot."""

    if isinstance(obs, dict):
        text = obs.get("obs_str")
        if isinstance(text, str) and text.strip():
            if "<image>" not in text:
                raise ValueError("VAGEN obs_str has no <image> placeholder")
            return text
    raise ValueError("environment observation has no non-empty obs_str")


def _obs_to_pil(obs) -> "Image.Image":
    """Convert env server observation to PIL Image.

    Handles VAGEN env server's multi_modal_data format where images are
    stored as ``{"multi_modal_data": {"image": [PIL.Image, ...]}, ...}``.
    """
    from PIL import Image

    if isinstance(obs, Image.Image):
        return obs

    if isinstance(obs, dict):
        # Standard direct image keys
        for key in ("image", "rgb", "pixels"):
            if key in obs:
                val = obs[key]
                if isinstance(val, Image.Image):
                    return val
                if hasattr(val, "shape"):
                    return Image.fromarray(val)
                if isinstance(val, dict) and "__pil_image__" in val:
                    from vagen.server.serial import deserialize_pil_image
                    return deserialize_pil_image(val)

        # VAGEN env server: multi_modal_data
        if "multi_modal_data" in obs:
            mm_data = obs["multi_modal_data"]
            # mm_data is a dict of lists, e.g. {"image": [PIL.Image], ...}
            for key in ("image", "images", "rgb", "pixels"):
                if key in mm_data:
                    values = mm_data[key]
                    if values and len(values) > 0:
                        val = values[0]
                        if isinstance(val, Image.Image):
                            return val
                        if hasattr(val, "shape"):  # numpy array
                            return Image.fromarray(val)
                        if isinstance(val, dict) and "__pil_image__" in val:
                            from vagen.server.serial import deserialize_pil_image
                            return deserialize_pil_image(val)

            # Try first available key
            for key, values in mm_data.items():
                if values and len(values) > 0:
                    val = values[0]
                    if isinstance(val, Image.Image):
                        return val
                    if hasattr(val, "shape"):
                        return Image.fromarray(val)

        raise ValueError(f"Cannot extract image from obs dict with keys {list(obs.keys())}")

    if hasattr(obs, "shape"):
        return Image.fromarray(obs)

    raise ValueError(f"Unknown obs type: {type(obs)}")


# ---------------------------------------------------------------------------
# Legacy placeholder
# ---------------------------------------------------------------------------


def _run_vagen_rollout(
    config_path: Path, checkpoint_dir: Path, output_dir: Path,
    num_episodes: int, max_steps: int,
) -> None:
    raise NotImplementedError(
        "Use EnvRolloutCollector with --env-url instead of VAGEN subprocess."
    )


# ---------------------------------------------------------------------------
# JSONL-backed collector (reads pre-collected trajectories from disk)
# ---------------------------------------------------------------------------


class JSONLRolloutCollector:
    """Read trajectories from pre-existing JSONL files/directories.

    用于外部 rollout（如 Slurm 上的 rollout_env.py）生成 JSONL 后，RL trainer
    从 JSONL 消费轨迹的离线场景。

    支持：
    - 指定一个或多个 JSONL 文件或目录（目录下递归搜索 ``*.jsonl`` 和 ``*.jsonl.gz``）
    - 按 iteration 循环读取（数据轮转，不会终止训练）
    - 分布式环境下所有 rank 调用 ``collect()`` 得到相同结果（确定性轮转）

    数据轮转策略：
    - 首次调用 ``collect()`` 时加载所有 JSONL 文件中的所有轨迹并 shuffle
    - 每次调用返回 ``num_episodes`` 条，内部指针前进
    - 指针到达末尾时自动回到开头（loop=True）
    - 所有 rank 同时调用、相同调用次数 → 得到相同轨迹序列
    """

    def __init__(self, sources: list[Path] | None = None, loop: bool = True) -> None:
        self._sources: list[Path] = list(sources) if sources else []
        self._loop = loop
        self._all_trajectories: list[RolloutTrajectory] | None = None
        self._cursor: int = 0
        self._call_count: int = 0  # 外部 collect 调用次数（用于分布式调试）

    def _load_all(self) -> list[RolloutTrajectory]:
        """首次调用时加载所有 JSONL 源文件中的轨迹并 shuffle。"""
        all_trajs: list[RolloutTrajectory] = []
        files = self._expand_sources()
        if not files:
            raise FileNotFoundError(
                f"JSONLRolloutCollector: 未找到任何 JSONL 文件，sources={self._sources}"
            )
        for fpath in files:
            try:
                loaded = load_trajectories(fpath)
                all_trajs.extend(loaded)
            except Exception as e:
                print(json.dumps({"jsonl_load_warning": str(fpath), "error": str(e)}),
                      flush=True)
        if not all_trajs:
            raise ValueError(
                f"JSONLRolloutCollector: 从 {len(files)} 个 JSONL 文件中未读到任何有效轨迹"
            )
        # shuffle 一次保证数据不按原始顺序；分布式下所有 rank 读同一 shuffle 结果
        import random
        rng = random.Random(42)
        rng.shuffle(all_trajs)
        return all_trajs

    def _expand_sources(self) -> list[Path]:
        """展开 sources 中的目录 → 所有 .jsonl / .jsonl.gz 文件。"""
        files: list[Path] = []
        for src in self._sources:
            if src.is_dir():
                for pat in ("**/*.jsonl", "**/*.jsonl.gz"):
                    files.extend(sorted(src.glob(pat)))
            elif src.exists():
                files.append(src)
        return files

    @property
    def total_trajectories(self) -> int:
        if self._all_trajectories is None:
            self._all_trajectories = self._load_all()
        return len(self._all_trajectories)

    def collect(
        self,
        *,
        num_episodes: int,
        max_steps_per_episode: int = 20,
        output_dir: Path | None = None,
    ) -> list[RolloutTrajectory]:
        """返回 ``num_episodes`` 条轨迹，从已加载的源数据中轮转读取。

        所有 rank 调用时都会拿到相同的轨迹序列（确定性），保证 FSDP 训练一致性。
        """
        self._call_count += 1
        if self._all_trajectories is None:
            self._all_trajectories = self._load_all()

        total = len(self._all_trajectories)
        if total == 0:
            return []

        result: list[RolloutTrajectory] = []
        needed = num_episodes
        while needed > 0:
            remaining = total - self._cursor
            take = min(needed, remaining)
            if take > 0:
                result.extend(self._all_trajectories[self._cursor:self._cursor + take])
                self._cursor += take
                needed -= take
            if self._cursor >= total:
                if self._loop:
                    self._cursor = 0
                else:
                    break  # 不循环，剩余不足
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def save_trajectories(trajectories: list[RolloutTrajectory], output_dir: Path) -> Path:
    """Write trajectories to a Nimloth JSONL file, one record per line."""
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "trajectories.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for traj in trajectories:
            validate_rollout_trajectory(traj)
            f.write(
                json.dumps(traj.to_record(), ensure_ascii=False, allow_nan=False) + "\n"
            )
    return jsonl_path


def load_trajectories(jsonl_path: Path) -> list[RolloutTrajectory]:
    """Read trajectories from a Nimloth JSONL or JSONL.GZ file."""
    trajectories: list[RolloutTrajectory] = []
    opener = gzip.open if jsonl_path.suffix == ".gz" else Path.open
    with opener(jsonl_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trajectories.append(RolloutTrajectory.from_record(json.loads(line)))
    return trajectories
