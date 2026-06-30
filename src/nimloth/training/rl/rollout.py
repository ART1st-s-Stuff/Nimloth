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
    """action_names[t] = VAGEN text name of action at step t."""
    action_log_probs: list[list[float]] = field(default_factory=list)
    """action_log_probs[t] = [log_prob(a0), ..., log_prob(a7)] at step t (log-softmax)."""
    nav_instruction: str = ""
    """Navigation instruction from env server."""
    success: bool = False
    reward: float = 0.0
    split: str = "train"
    messages: list[dict[str, Any]] = field(default_factory=list)
    """Full conversation history (system, user, assistant turns)."""

    @property
    def num_steps(self) -> int:
        return len(self.action_indices)

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
            "action_log_probs": self.action_log_probs,
            "nav_instruction": self.nav_instruction,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "RolloutTrajectory":
        return cls(
            record_id=str(record.get("id", "")),
            image_paths=list(record.get("image_paths", [])),
            action_indices=list(record.get("action_indices", [])),
            action_names=list(record.get("action_names", [])),
            action_log_probs=list(record.get("action_log_probs", [])),
            nav_instruction=str(record.get("nav_instruction", "")),
            success=bool(record.get("success", False)),
            reward=float(record.get("reward", 0.0)),
            split=str(record.get("split", "train")),
            messages=list(record.get("messages", [])),
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

# Map VAGEN text action names → numeric indices (aligned with ACTION_NAMES order)
ACTION_NAME_MAP: dict[str, int] = {
    "moveahead": 0,
    "moveback": 1,
    "moveright": 2,
    "moveleft": 3,
    "rotateright": 4,
    "rotateleft": 5,
    "lookup": 6,
    "lookdown": 7,
}
ACTION_NAMES: list[str] = [
    "moveahead", "moveback", "moveright", "moveleft",
    "rotateright", "rotateleft", "lookup", "lookdown",
]
ACTION_NAME_TO_IDX: dict[str, int] = {name: idx for idx, name in enumerate(ACTION_NAMES)}

_NAV_SYSTEM_TEXT = (
    "You are a home robot and perform navigation tasks according to instructions.\n"
    "Actions you can take: moveahead, moveback, moveright, moveleft, "
    "rotateright, rotateleft, lookup, lookdown.\n"
    "Rewards: Format correct: +0.5. Achieve the human instruction: +10.0.\n"
    "Look at the image carefully and navigate to complete the instruction."
)


class EnvRolloutCollector:
    """Collect trajectories by running Qwen policy against the VAGEN env server.

    Reuses the trainer's Qwen model (no subprocess/model-reloading).
    Each ``collect()`` call creates envs on the server, runs Qwen-based
    greedy action selection, and returns ``RolloutTrajectory`` objects.
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
    ) -> None:
        self._model = qwen_model
        self._processor = processor
        self._env_url = env_url.rstrip("/")
        self._device = device
        self._ep_counter = seed_offset
        self._client = None  # lazy init
        self._temperature = temperature
        self._top_p = top_p

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
            ep_id = f"rl_{self._ep_counter:06d}"
            self._ep_counter += 1
            seed = self._ep_counter * 13 + 7
            t0 = time.time()
            eval_set = "base" if (ep_i % 2 == 0) else "common_sense"

            print(json.dumps({"rl_ep": ep_i, "id": ep_id, "eval_set": eval_set}), flush=True)

            env_config = {
                "env_name": "navigation",
                "env_config": {
                    "render_mode": "vision",
                    "prompt_format": "wm",
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
                nav_instruction = prompts.get(ep_id, "Navigate to the target object.")
                print(json.dumps({"rl_ep": ep_i, "step": "get_prompt_done"}), flush=True)
            except Exception:
                import traceback
                traceback.print_exc()
                nav_instruction = "Navigate to the target object."

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
            action_log_probs_list: list[list[float]] = []
            image_paths: list[str] = []
            done = False
            step_rewards: list[float] = []
            success = False

            for step in range(max_steps_per_episode):
                print(json.dumps({"rl_ep": ep_i, "step": f"action_{step}", "history_len": len(action_names)}), flush=True)

                # --- image ---
                try:
                    img = _obs_to_pil(obs)
                except Exception:
                    import traceback
                    traceback.print_exc()
                    break

                # --- save image ---
                img_path = img_dir / f"{ep_id}_step{step:02d}.png"
                img.save(str(img_path))
                image_paths.append(str(img_path))

                # --- qwen action selection ---
                try:
                    action_name, action_idx, log_probs_list = _select_action_nimloth(
                        self._model, self._processor, img,
                        nav_instruction, action_names,
                        temperature=self._temperature,
                        top_p=self._top_p,
                    )
                    print(json.dumps({"rl_ep": ep_i, "action_selected": action_name,
                                      "action_idx": action_idx}), flush=True)
                except Exception:
                    import traceback
                    traceback.print_exc()
                    action_name, action_idx, log_probs_list = "moveahead", 0, [0.0] * 8

                # --- env step ---
                # Build VAGEN wm-format response so parse_worldmodeling succeeds.
                vagen_response = (
                    f"<think><reasoning>Navigating toward target.</reasoning>"
                    f"<prediction>Moving.</prediction></think>"
                    f"<answer>{action_name}</answer>"
                )
                try:
                    step_results = self._client.step_batch({ep_id: vagen_response})
                    obs, r, done, info = step_results[ep_id]
                    # Apply failure penalty if action didn't execute
                    action_ok = info.get("last_action_success", True) if isinstance(info, dict) else True
                    if not action_ok:
                        r = float(r) - 0.1  # failure_penalty
                    step_rewards.append(float(r))
                    print(json.dumps({"rl_ep": ep_i, "env_step_done": True, "done": done,
                                      "step_reward": r, "action_ok": action_ok}), flush=True)
                except Exception:
                    import traceback
                    traceback.print_exc()
                    break

                action_names.append(action_name)
                action_indices.append(action_idx)
                # Lazy init list for log_probs
                if not hasattr(self, '_ep_log_probs'):
                    self._ep_log_probs: list[list[float]] = []
                self._ep_log_probs.append(log_probs_list)

                if done:
                    break

            # Save final observation (so image_paths has len = num_steps + 1)
            try:
                final_img = _obs_to_pil(obs)
                img_path = img_dir / f"{ep_id}_step{len(action_names):02d}.png"
                final_img.save(str(img_path))
                image_paths.append(str(img_path))
            except Exception:
                pass

            # --- compute success from per-step rewards ---
            reward = sum(step_rewards)
            success = any(r >= 10.0 for r in step_rewards)

            # --- close env ---
            try:
                self._client.close_batch([ep_id])
            except Exception:
                pass

            ep_log_probs = list(getattr(self, '_ep_log_probs', []))
            messages = _build_vagen_messages(nav_instruction, len(action_names), action_names)
            trajectories.append(RolloutTrajectory(
                record_id=ep_id,
                image_paths=image_paths,
                action_indices=action_indices,
                action_names=list(action_names),
                action_log_probs=ep_log_probs,
                nav_instruction=nav_instruction,
                success=success,
                reward=reward,
                split="train",
                messages=messages,
            ))
            self._ep_log_probs = []

            elapsed = time.time() - t0
            print(json.dumps({
                "rl_ep": ep_i, "done": True,
                "steps": len(action_names),
                "success": success,
                "reward": round(reward, 2),
                "elapsed_s": round(elapsed, 1),
            }), flush=True)

        print(json.dumps({"rl_collect": "done", "trajectories": len(trajectories)}), flush=True)
        return trajectories


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _select_action_nimloth(model, processor, image, nav_instruction: str,
                           action_history: list[str],
                           temperature: float = 1.0,
                           top_p: float = 1.0) -> tuple[str, int, list[float]]:
    """Sampled action selection using Nimloth action tokens.

    The SFT2 model was trained with ``<|action_(0)|>`` … ``<|action_(7)|>``
    special tokens.  We build a Nimloth-format prompt (ending with
    ``<|action_start|>``), run Qwen forward, and extract the logits at
    ``<|action_start|>``.  Sampling with temperature + nucleus (top-p).

    Returns (action_name, action_index, action_log_probs) where
    action_log_probs is the log-softmax over all 8 actions.
    """
    import torch
    from nimloth.latent.extraction import (
        LatentActionTokens,
        extract_action_prior,
        special_token_ids,
    )

    tokens = LatentActionTokens()
    token_ids = special_token_ids(processor.tokenizer, tokens)
    action_token_ids = [token_ids[t] for t in tokens.action_tokens]

    # Build Nimloth-format messages.
    # The assistant response includes <|latent_state|> and <|action_start|>.
    # The model will predict the next token (one of <|action_(N)|>).
    num_images = 1 + len(action_history)
    messages: list[dict] = [
        {"role": "system", "content": [{"type": "text", "text": _NAV_SYSTEM_TEXT}]},
    ]

    # Initial observation
    messages.append({"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": f"Observe the scene. {nav_instruction}"},
    ]})

    # History turns: user shows image, assistant says what it did + latent + action_start
    for act_name in action_history:
        messages.append({"role": "assistant", "content": [
            {"type": "text", "text": (
                f"<think>Navigating.</think>"
                f"<|latent_state|><|action_start|><|action_({ACTION_NAME_TO_IDX[act_name]})|><|action_end|>"
            )},
        ]})
        messages.append({"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"Observe the scene after {act_name}. {nav_instruction}"},
        ]})

    # Current turn: we want the model to predict the next action
    messages.append({"role": "assistant", "content": [
        {"type": "text", "text": "<think>What should I do next?</think><|latent_state|><|action_start|>"},
    ]})

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image] * num_images, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    # Locate the <|action_start|> token. Its logits predict the next token
    # (one of <|action_(0)|>…<|action_(7)|> in the training distribution).
    input_ids = inputs["input_ids"][0]
    as_positions = (input_ids == token_ids[tokens.action_start]).nonzero(as_tuple=True)[0]
    if as_positions.numel() == 0:
        raise RuntimeError("<|action_start|> token not found in prompt")
    action_start_pos = int(as_positions[-1].item())  # use the last one
    logits = outputs.logits[0, action_start_pos, :]
    action_logits = logits[action_token_ids]
    action_log_probs = torch.log_softmax(action_logits.float(), dim=-1)

    # Sample with temperature
    if temperature > 0:
        scaled_logits = action_logits.float() / temperature
        if top_p < 1.0:
            # Nucleus (top-p) sampling
            sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
            cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            keep = cum_probs <= top_p
            keep[0] = True  # Always keep top token
            keep_mask = torch.zeros_like(scaled_logits, dtype=torch.bool)
            keep_mask[sorted_indices[keep]] = True
            scaled_logits[~keep_mask] = float("-inf")
        probs = torch.softmax(scaled_logits, dim=-1)
        chosen_idx = int(torch.multinomial(probs, 1).item())
    else:
        chosen_idx = int(action_logits.argmax().item())

    best_name = ACTION_NAMES[chosen_idx]
    return best_name, chosen_idx, action_log_probs.cpu().tolist()


def _build_vagen_messages(nav_instruction: str, num_steps: int,
                          action_names: list[str]) -> list[dict]:
    """Build conversation messages for the trajectory record."""
    messages: list[dict] = [
        {"role": "system", "content": _NAV_SYSTEM_TEXT},
    ]
    messages.append({"role": "user", "content": (
        f"[Initial Observation]:\n{nav_instruction}\nDecide your next action(s)."
    )})
    for i, act_name in enumerate(action_names):
        messages.append({"role": "assistant",
                         "content": f"<think>Reasoning.</think><answer>{act_name}</answer>"})
        if i + 1 < num_steps:
            messages.append({"role": "user", "content": (
                f"After your answer, the extracted valid action is {act_name}.\n"
                f"The environment feedback is: Last action executed successfully.\n"
                f"After that, the observation is:\n{nav_instruction}\n"
                f"Decide your next action(s)."
            )})
    return messages


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
            f.write(json.dumps(traj.to_record(), ensure_ascii=False) + "\n")
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
