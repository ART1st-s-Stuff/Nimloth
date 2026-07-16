"""Online rollout collection: Qwen policy interacting with VAGEN environments.

The rollout collector runs the Qwen policy in the VAGEN navigation environment,
collecting trajectories that include per-frame images, taken actions, and sparse rewards.
Each trajectory is later encoded into WM latent states by the trainer.
"""

from __future__ import annotations

import gzip
import json
import math
import time

import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


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
        eval_sets: tuple[str, ...] = ("base", "common_sense"),
        split: str = "eval",
        history_window: int = 4,
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
        self._base_seed_offset = int(seed_offset)
        self._ep_counter = int(seed_offset)
        self._client = None  # lazy init
        self._temperature = temperature
        self._top_p = top_p
        self._eval_sets = eval_sets
        self._split = split
        if history_window < 0:
            raise ValueError(f"history_window must be >= 0, got {history_window}")
        self._history_window = int(history_window)

    @property
    def client(self):
        if self._client is None:
            from vagen.server.client import BatchEnvClient
            self._client = BatchEnvClient(base_url=self._env_url, timeout=600)
        return self._client

    def _environment_config(self, eval_set: str) -> dict[str, Any]:
        return {
            "env_name": "navigation",
            "env_config": {
                "render_mode": "vision",
                "prompt_format": "wm",
                "use_state_reward": False,
                "eval_set": eval_set,
                "max_actions_per_step": 1,
                "max_action_penalty": -0.1,
                "format_reward": 0.0,
                "success_threshold": 1.5,
                "step_length": 0.5,
                "grounding_reward_weight": 0.5,
                "worldmodeling_reward_weight": 0.5,
                "gpu_device": 0,
            },
        }

    def _sampling_seed(self, episode_seed: int, step: int) -> int:
        # Sampling is a pure function of env seed + step, so checkpoint resume
        # does not depend on a process-local RNG state.
        return int((self._base_seed_offset + 1) * 1_000_003 + episode_seed * 997 + step)

    def set_resume_iteration(
        self,
        *,
        start_iteration: int,
        envs_per_iteration: int,
        validation_enabled: bool,
        validation_interval: int,
        validation_envs: int,
    ) -> None:
        """Restore the deterministic environment-seed cursor for resume."""

        completed = max(0, int(start_iteration) - 1)
        consumed = completed * int(envs_per_iteration)
        if validation_enabled and validation_interval > 0:
            consumed += (completed // int(validation_interval)) * int(validation_envs)
        self._ep_counter = self._base_seed_offset + consumed

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
            seed = self._ep_counter
            ep_id = f"rl_{seed:06d}"
            self._ep_counter += 1
            t0 = time.time()
            eval_set = self._eval_sets[ep_i % len(self._eval_sets)]

            print(json.dumps({"rl_ep": ep_i, "id": ep_id, "eval_set": eval_set}), flush=True)

            env_config = self._environment_config(eval_set)

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
                nav_instruction = str(prompts.get(ep_id, "")).strip()
                if not nav_instruction:
                    raise RuntimeError(f"environment returned no instruction for {ep_id}")
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
            done = False
            step_rewards: list[float] = []
            success = False
            episode_valid = True

            for step in range(max_steps_per_episode):
                print(json.dumps({"rl_ep": ep_i, "step": f"action_{step}", "history_len": len(action_names)}), flush=True)

                # --- image ---
                try:
                    img = _obs_to_pil(obs)
                except Exception:
                    import traceback
                    traceback.print_exc()
                    episode_valid = False
                    break

                # --- save image ---
                img_path = img_dir / f"{ep_id}_step{step:02d}.png"
                img.save(str(img_path))
                image_paths.append(str(img_path))

                # --- qwen action selection ---
                try:
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(self._sampling_seed(seed, step))
                    action_name, action_idx, log_probs_list = _select_action_nimloth(
                        self._model, self._processor, image_paths,
                        nav_instruction, action_names,
                        temperature=self._temperature,
                        top_p=self._top_p,
                        history_window=self._history_window,
                        generator=generator,
                    )
                    print(json.dumps({"rl_ep": ep_i, "action_selected": action_name,
                                      "action_idx": action_idx}), flush=True)
                except Exception:
                    import traceback
                    traceback.print_exc()
                    episode_valid = False
                    break

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
                    episode_valid = False
                    break

                action_names.append(action_name)
                action_indices.append(action_idx)
                action_log_probs.append(log_probs_list)

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

            if not episode_valid or not action_names or len(image_paths) != len(action_names) + 1:
                print(json.dumps({
                    "rl_ep": ep_i,
                    "warning": "discarding incomplete trajectory",
                    "num_actions": len(action_names),
                    "num_images": len(image_paths),
                }), flush=True)
                continue

            messages = _build_vagen_messages(nav_instruction, len(action_names), action_names)
            trajectory = RolloutTrajectory(
                record_id=ep_id,
                image_paths=image_paths,
                action_indices=action_indices,
                action_names=list(action_names),
                action_log_probs=action_log_probs,
                nav_instruction=nav_instruction,
                success=success,
                reward=reward,
                split=self._split,
                messages=messages,
            )
            validate_rollout_trajectory(trajectory)
            trajectories.append(trajectory)

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


def _window_policy_history(
    image_history: list[Any],
    action_history: list[str],
    history_window: int,
) -> tuple[list[Any], list[str]]:
    if len(image_history) != len(action_history) + 1:
        raise ValueError(
            "policy history requires one more image than actions: "
            f"images={len(image_history)}, actions={len(action_history)}"
        )
    if history_window < 0:
        raise ValueError(f"history_window must be >= 0, got {history_window}")
    if len(action_history) > history_window:
        action_history = action_history[-history_window:] if history_window else []
        image_history = image_history[-(history_window + 1):]
    return list(image_history), list(action_history)


def build_nimloth_policy_messages(
    image_history: list[Any],
    nav_instruction: str,
    action_history: list[str],
    *,
    history_window: int,
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Build the canonical k=1/inject action prompt with real history images."""

    image_history, action_history = _window_policy_history(
        image_history, action_history, history_window
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": _NAV_SYSTEM_TEXT}]},
        {"role": "user", "content": [
            {"type": "image", "image": image_history[0]},
            {"type": "text", "text": f"Observe the scene. {nav_instruction}"},
        ]},
    ]
    for index, action_name in enumerate(action_history):
        if action_name not in ACTION_NAME_TO_IDX:
            raise ValueError(f"unknown navigation action in history: {action_name!r}")
        messages.append({"role": "assistant", "content": [
            {"type": "text", "text": (
                "<think>Navigating.</think><|latent_state|><|action_start|>"
                f"<|action_({ACTION_NAME_TO_IDX[action_name]})|><|action_end|>"
            )},
        ]})
        messages.append({"role": "user", "content": [
            {"type": "image", "image": image_history[index + 1]},
            {"type": "text", "text": (
                f"Observe the scene after {action_name}. {nav_instruction}"
            )},
        ]})
    messages.append({"role": "assistant", "content": [
        {"type": "text", "text": (
            "<think>What should I do next?</think>"
            "<|latent_state|><|action_start|>"
        )},
    ]})
    return messages, image_history


def compute_nimloth_action_distribution(
    model,
    processor,
    image_history: list[Any],
    nav_instruction: str,
    action_history: list[str],
    *,
    history_window: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw logits and untempered log-probs for the eight action tokens."""

    from nimloth.latent.extraction import LatentActionTokens, special_token_ids

    tokens = LatentActionTokens()
    token_ids = special_token_ids(processor.tokenizer, tokens)
    action_token_ids = [token_ids[token] for token in tokens.action_tokens]
    messages, images = build_nimloth_policy_messages(
        image_history,
        nav_instruction,
        action_history,
        history_window=history_window,
    )
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=images, return_tensors="pt", padding=True
    )
    try:
        device = next(model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("policy model has no parameters") from exc
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=False, return_dict=True)

    input_ids = inputs["input_ids"][0]
    positions = (input_ids == token_ids[tokens.action_start]).nonzero(as_tuple=True)[0]
    if positions.numel() == 0:
        raise RuntimeError("<|action_start|> token not found in prompt")
    action_start_pos = int(positions[-1].item())
    action_ids = torch.tensor(action_token_ids, device=outputs.logits.device)
    action_logits = outputs.logits[0, action_start_pos, action_ids].float()
    if action_logits.shape != (len(ACTION_NAMES),):
        raise RuntimeError(f"unexpected action logits shape: {tuple(action_logits.shape)}")
    if not torch.isfinite(action_logits).all():
        raise RuntimeError("action logits contain non-finite values")
    return action_logits, torch.log_softmax(action_logits, dim=-1)


def sample_action_from_logits(
    action_logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None = None,
) -> tuple[int, list[float]]:
    """Sample one action and return temperature-scaled full log-probs.

    Top-p only restricts sampling.  PPO log-probs use the full temperature-scaled
    distribution, matching VAGEN/VERL's recompute-log-prob convention.
    """

    if action_logits.shape != (len(ACTION_NAMES),):
        raise ValueError(f"expected 8 action logits, got {tuple(action_logits.shape)}")
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if not 0 < top_p <= 1:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    logits = action_logits.detach().float().cpu()
    if temperature == 0:
        chosen_idx = int(logits.argmax().item())
        log_probs = torch.log_softmax(logits, dim=-1)
        return chosen_idx, [float(value) for value in log_probs.tolist()]

    scaled = logits / temperature
    log_probs = torch.log_softmax(scaled, dim=-1)
    sampling_logits = scaled.clone()
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(sampling_logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
        keep = cumulative_before < top_p
        keep[0] = True
        keep_mask = torch.zeros_like(sampling_logits, dtype=torch.bool)
        keep_mask[sorted_indices[keep]] = True
        sampling_logits[~keep_mask] = float("-inf")
    probabilities = torch.softmax(sampling_logits, dim=-1)
    chosen_idx = int(torch.multinomial(probabilities, 1, generator=generator).item())
    return chosen_idx, [float(value) for value in log_probs.tolist()]


def _select_action_nimloth(
    model,
    processor,
    image_history: list[Any],
    nav_instruction: str,
    action_history: list[str],
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    history_window: int = 4,
    generator: torch.Generator | None = None,
) -> tuple[str, int, list[float]]:
    """Compute the current policy distribution and sample one action."""

    action_logits, _ = compute_nimloth_action_distribution(
        model,
        processor,
        image_history,
        nav_instruction,
        action_history,
        history_window=history_window,
    )
    chosen_idx, log_probs = sample_action_from_logits(
        action_logits,
        temperature=temperature,
        top_p=top_p,
        generator=generator,
    )
    return ACTION_NAMES[chosen_idx], chosen_idx, log_probs


def validate_rollout_trajectory(trajectory: RolloutTrajectory) -> None:
    """Reject incomplete or numerically invalid policy trajectories."""

    steps = trajectory.num_steps
    if steps <= 0:
        raise ValueError(f"trajectory {trajectory.record_id!r} has no actions")
    if len(trajectory.image_paths) != steps + 1:
        raise ValueError(
            f"trajectory {trajectory.record_id!r}: images={len(trajectory.image_paths)} "
            f"but actions={steps}"
        )
    if len(trajectory.action_names) != steps:
        raise ValueError(
            f"trajectory {trajectory.record_id!r}: action_names="
            f"{len(trajectory.action_names)} but actions={steps}"
        )
    if len(trajectory.action_log_probs) != steps:
        raise ValueError(
            f"trajectory {trajectory.record_id!r}: action_log_probs="
            f"{len(trajectory.action_log_probs)} but actions={steps}"
        )
    for step, (action_idx, log_probs) in enumerate(
        zip(trajectory.action_indices, trajectory.action_log_probs)
    ):
        if not 0 <= int(action_idx) < len(ACTION_NAMES):
            raise ValueError(f"trajectory {trajectory.record_id!r}: invalid action {action_idx}")
        if len(log_probs) != len(ACTION_NAMES):
            raise ValueError(
                f"trajectory {trajectory.record_id!r}: step {step} has "
                f"{len(log_probs)} action log-probs"
            )
        if not all(math.isfinite(float(value)) for value in log_probs):
            raise ValueError(
                f"trajectory {trajectory.record_id!r}: step {step} has non-finite log-probs"
            )
    if not trajectory.nav_instruction.strip():
        raise ValueError(f"trajectory {trajectory.record_id!r} has no instruction")


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
