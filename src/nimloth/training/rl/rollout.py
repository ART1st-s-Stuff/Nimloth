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
from typing import TYPE_CHECKING, Any, Callable, Protocol

import torch

if TYPE_CHECKING:
    from PIL.Image import Image


@dataclass(frozen=True)
class RLPolicyProtocol:
    """Latent-query protocol read from an SFT/HF checkpoint."""

    latent_token_count: int
    latent_query_mode: str


def validate_rl_policy_protocol(model_config: Any) -> RLPolicyProtocol:
    """Return the supported metadata-driven RL protocol or fail fast.

    RL supplies latent query slots as inputs.  Autoregressively generated query
    tokens have different policy-probability ownership and are intentionally not
    accepted by this runtime.
    """

    latent_count = int(getattr(model_config, "nimloth_latent_token_count", 1))
    query_mode = getattr(model_config, "nimloth_latent_query_mode", None)
    if latent_count < 1:
        raise ValueError(f"RL latent_token_count must be positive, got {latent_count}")
    if query_mode != "inject":
        raise ValueError(
            "RL action/encoding runtime currently supports inject query mode only; "
            f"got latent_token_count={latent_count}, latent_query_mode={query_mode!r}"
        )
    return RLPolicyProtocol(latent_count, query_mode)


def qwen_hidden_size_from_config(model_config: Any) -> int:
    """Read Qwen text hidden size without assuming one Transformers layout."""

    hidden_size = getattr(model_config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(getattr(model_config, "text_config", None), "hidden_size", None)
    if hidden_size is None or int(hidden_size) < 1:
        raise ValueError("Qwen checkpoint config does not expose a positive hidden_size")
    return int(hidden_size)


def build_injected_query_prefix(
    latent_token_count: int,
    *,
    include_action_start: bool,
) -> str:
    """Build the canonical injected latent block used by RL prompts."""

    from nimloth.latent.extraction import LatentActionTokens, latent_state_block

    tokens = LatentActionTokens()
    block = latent_state_block(latent_token_count, tokens)
    return block + (tokens.action_start if include_action_start else "")


@dataclass(frozen=True)
class WMActionDecision:
    action_index: int
    state_source: str
    fast_path_step: int


class WMValueFastPathController:
    """Greedy WM/value policy with periodic Qwen-GT state re-synchronization."""

    def __init__(
        self,
        *,
        encode_state: Callable[[Any], torch.Tensor],
        predictor: Any,
        value_head: Any,
        horizon: int,
    ) -> None:
        if horizon < 1:
            raise ValueError(f"fast_path_horizon must be positive, got {horizon}")
        self._encode_state = encode_state
        self._predictor = predictor
        self._value_head = value_head
        self._horizon = int(horizon)
        self.reset()

    def reset(self) -> None:
        self._state: torch.Tensor | None = None
        self._step = 0

    @property
    def needs_sync(self) -> bool:
        return self._state is None

    def start_segment(self, qwen_gt_state: torch.Tensor) -> None:
        if qwen_gt_state.ndim != 2 or qwen_gt_state.shape[0] != 1:
            raise ValueError(
                "qwen_gt_state must have shape (1, emb_dim), got "
                f"{tuple(qwen_gt_state.shape)}"
            )
        self._state = qwen_gt_state.detach()
        self._step = 0

    @torch.no_grad()
    def select_action(self, observation: Any) -> WMActionDecision:
        if self._state is None:
            self._state = self._encode_state(observation).detach()
            self._step = 0
            state_source = "qwen_gt"
        else:
            state_source = "wm_predicted"
        values = self._value_head(self._state).float()
        if values.ndim != 2 or values.shape[0] != 1:
            raise ValueError(
                "WMValueFastPathController expects value_head output shape (1, num_actions), "
                f"got {tuple(values.shape)}"
            )
        return WMActionDecision(
            action_index=int(values.argmax(dim=-1).item()),
            state_source=state_source,
            fast_path_step=self._step,
        )

    @torch.no_grad()
    def advance(self, action_index: int, *, done: bool) -> None:
        if self._state is None:
            raise RuntimeError("select_action() must be called before advance()")
        if done or self._step + 1 >= self._horizon:
            self.reset()
            return
        action = torch.tensor(
            [int(action_index)], dtype=torch.long, device=self._state.device
        )
        self._state = self._predictor.predict_next_emb(self._state, action).detach()
        self._step += 1


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
    action_log_probs: list[list[float | None] | None] = field(default_factory=list)
    """Qwen behavior log-probs, or None when the behavior policy is WM/value."""
    policy_sources: list[str] = field(default_factory=list)
    """Per-step behavior policy ownership: qwen or wm_value."""
    state_sources: list[str] = field(default_factory=list)
    """Per-step state source: qwen_gt or wm_predicted."""
    fast_path_steps: list[int] = field(default_factory=list)
    """Zero-based position inside the current fast-path segment."""
    rollout_policy: str = "qwen"
    fast_path_horizon: int = 0
    latent_token_count: int = 1
    latent_query_mode: str = "inject"
    action_temperature: float = 1.0
    action_top_p: float = 1.0
    action_log_prob_semantics: str | None = None
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
            "policy_sources": self.policy_sources,
            "state_sources": self.state_sources,
            "fast_path_steps": self.fast_path_steps,
            "rollout_policy": self.rollout_policy,
            "fast_path_horizon": self.fast_path_horizon,
            "latent_token_count": self.latent_token_count,
            "latent_query_mode": self.latent_query_mode,
            "action_temperature": self.action_temperature,
            "action_top_p": self.action_top_p,
            "action_log_prob_semantics": self.action_log_prob_semantics,
            "nav_instruction": self.nav_instruction,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "RolloutTrajectory":
        action_indices = list(record.get("action_indices", []))
        policy_sources = list(record.get("policy_sources", []))
        if not policy_sources and action_indices:
            policy_sources = ["qwen"] * len(action_indices)
        state_sources = list(record.get("state_sources", []))
        if not state_sources and action_indices:
            state_sources = ["qwen_gt"] * len(action_indices)
        fast_path_steps = list(record.get("fast_path_steps", []))
        if not fast_path_steps and action_indices:
            fast_path_steps = [0] * len(action_indices)
        return cls(
            record_id=str(record.get("id", "")),
            image_paths=list(record.get("image_paths", [])),
            action_indices=action_indices,
            action_names=list(record.get("action_names", [])),
            action_log_probs=list(record.get("action_log_probs", [])),
            policy_sources=policy_sources,
            state_sources=state_sources,
            fast_path_steps=fast_path_steps,
            rollout_policy=str(record.get("rollout_policy", "qwen")),
            fast_path_horizon=int(record.get("fast_path_horizon", 0)),
            latent_token_count=int(record.get("latent_token_count", 1)),
            latent_query_mode=str(record.get("latent_query_mode", "inject")),
            action_temperature=float(record.get("action_temperature", 1.0)),
            action_top_p=float(record.get("action_top_p", 1.0)),
            action_log_prob_semantics=record.get("action_log_prob_semantics"),
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
        rollout_policy: str = "qwen",
        state_proj: Any = None,
        wm_predictor: Any = None,
        value_head: Any = None,
        fast_path_horizon: int = 2,
        latent_token_count: int = 1,
        latent_query_mode: str = "inject",
    ) -> None:
        if not eval_sets:
            raise ValueError("EnvRolloutCollector requires at least one eval_set")
        if split == "train" and any(not name.endswith("_train") for name in eval_sets):
            raise ValueError(
                "training rollout requires *_train datasets; "
                f"got eval_sets={eval_sets}"
            )
        if rollout_policy not in ("qwen", "wm_value", "qwen_wm"):
            raise ValueError(
                "rollout_policy must be qwen, wm_value, or qwen_wm, "
                f"got {rollout_policy!r}"
            )
        if latent_token_count < 1 or latent_query_mode != "inject":
            raise ValueError(
                "EnvRolloutCollector requires a positive-k inject protocol, got "
                f"k={latent_token_count}, mode={latent_query_mode!r}"
            )
        if rollout_policy in ("wm_value", "qwen_wm"):
            missing = [
                name
                for name, module in (
                    ("state_proj", state_proj),
                    ("wm_predictor", wm_predictor),
                    ("value_head", value_head),
                )
                if module is None
            ]
            if missing:
                raise ValueError(f"WM rollout requires {', '.join(missing)}")
            if fast_path_horizon < 1:
                raise ValueError("WM fast_path_horizon must be positive")
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
        self._rollout_policy = rollout_policy
        self._state_proj = state_proj
        self._wm_predictor = wm_predictor
        self._value_head = value_head
        self._fast_path_horizon = int(fast_path_horizon)
        self._latent_token_count = int(latent_token_count)
        self._latent_query_mode = latent_query_mode

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
            action_log_probs: list[list[float | None] | None] = []
            policy_sources: list[str] = []
            state_sources: list[str] = []
            fast_path_steps: list[int] = []
            image_paths: list[str] = []
            observation_images: list[Any] = []
            done = False
            step_rewards: list[float] = []
            success = False
            wm_controller: WMValueFastPathController | None = None
            if self._rollout_policy in ("wm_value", "qwen_wm"):
                wm_controller = WMValueFastPathController(
                    encode_state=lambda image: _encode_wm_state_nimloth(
                        self._model,
                        self._processor,
                        image,
                        nav_instruction,
                        action_names,
                        self._state_proj,
                        self._device,
                        latent_token_count=self._latent_token_count,
                        observation_history=observation_images,
                    ),
                    predictor=self._wm_predictor,
                    value_head=self._value_head,
                    horizon=self._fast_path_horizon,
                )

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
                observation_images.append(img.copy())

                # --- behavior-policy action selection ---
                try:
                    qwen_step = self._rollout_policy == "qwen" or (
                        self._rollout_policy == "qwen_wm"
                        and wm_controller is not None
                        and wm_controller.needs_sync
                    )
                    if qwen_step:
                        (
                            action_name,
                            action_idx,
                            log_probs_list,
                            qwen_wm_state,
                        ) = _select_action_nimloth(
                            self._model,
                            self._processor,
                            img,
                            nav_instruction,
                            action_names,
                            temperature=self._temperature,
                            top_p=self._top_p,
                            latent_token_count=self._latent_token_count,
                            observation_history=observation_images,
                            state_proj=(
                                self._state_proj
                                if self._rollout_policy == "qwen_wm"
                                else None
                            ),
                        )
                        if self._rollout_policy == "qwen_wm":
                            if qwen_wm_state is None or wm_controller is None:
                                raise RuntimeError(
                                    "hybrid Qwen step did not produce a WM state"
                                )
                            wm_controller.start_segment(qwen_wm_state)
                        policy_source = "qwen"
                        state_source = "qwen_gt"
                        fast_path_step = 0
                    else:
                        assert wm_controller is not None
                        decision = wm_controller.select_action(img)
                        action_idx = decision.action_index
                        action_name = ACTION_NAMES[action_idx]
                        log_probs_list = None
                        policy_source = "wm_value"
                        state_source = decision.state_source
                        fast_path_step = decision.fast_path_step
                    print(json.dumps({
                        "rl_ep": ep_i,
                        "action_selected": action_name,
                        "action_idx": action_idx,
                        "policy_source": policy_source,
                        "state_source": state_source,
                        "fast_path_step": fast_path_step,
                    }), flush=True)
                except Exception:
                    import traceback
                    traceback.print_exc()
                    # A policy failure invalidates behavior ownership; discard the
                    # incomplete trajectory instead of fabricating an action/log-prob.
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
                    break

                action_names.append(action_name)
                action_indices.append(action_idx)
                action_log_probs.append(log_probs_list)
                policy_sources.append(policy_source)
                state_sources.append(state_source)
                fast_path_steps.append(fast_path_step)
                if wm_controller is not None:
                    wm_controller.advance(action_idx, done=done)

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

            if not action_names or len(image_paths) != len(action_names) + 1:
                print(json.dumps({
                    "rl_ep": ep_i,
                    "warning": "discarding incomplete trajectory",
                    "num_actions": len(action_names),
                    "num_images": len(image_paths),
                }), flush=True)
                continue

            messages = _build_vagen_messages(nav_instruction, len(action_names), action_names)
            trajectories.append(RolloutTrajectory(
                record_id=ep_id,
                image_paths=image_paths,
                action_indices=action_indices,
                action_names=list(action_names),
                action_log_probs=action_log_probs,
                policy_sources=policy_sources,
                state_sources=state_sources,
                fast_path_steps=fast_path_steps,
                rollout_policy=self._rollout_policy,
                fast_path_horizon=(
                    self._fast_path_horizon
                    if self._rollout_policy in ("wm_value", "qwen_wm")
                    else 0
                ),
                latent_token_count=self._latent_token_count,
                latent_query_mode=self._latent_query_mode,
                action_temperature=self._temperature,
                action_top_p=self._top_p,
                action_log_prob_semantics=(
                    "sampling_distribution_v1"
                    if self._rollout_policy in ("qwen", "qwen_wm")
                    else None
                ),
                nav_instruction=nav_instruction,
                success=success,
                reward=reward,
                split=self._split,
                messages=messages,
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


def _obs_to_pil(obs) -> "Image":
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


def _build_nimloth_policy_messages(
    image: Any,
    nav_instruction: str,
    action_history: list[str],
    *,
    latent_token_count: int,
    include_action_start: bool,
    observation_history: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Build the canonical injected-query prompt for Qwen action/state reads."""

    observations = list(observation_history or [image] * (len(action_history) + 1))
    if len(observations) != len(action_history) + 1:
        raise ValueError(
            "observation_history must contain one more item than action_history, got "
            f"{len(observations)} observations and {len(action_history)} actions"
        )
    query_prefix = build_injected_query_prefix(
        latent_token_count, include_action_start=include_action_start
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": _NAV_SYSTEM_TEXT}]},
        {"role": "user", "content": [
            {"type": "image", "image": observations[0]},
            {"type": "text", "text": f"Observe the scene. {nav_instruction}"},
        ]},
    ]
    for step, act_name in enumerate(action_history):
        messages.append({"role": "assistant", "content": [
            {"type": "text", "text": (
                f"<think>Navigating.</think>{build_injected_query_prefix(latent_token_count, include_action_start=True)}"
                f"<|action_({ACTION_NAME_TO_IDX[act_name]})|><|action_end|>"
            )},
        ]})
        messages.append({"role": "user", "content": [
            {"type": "image", "image": observations[step + 1]},
            {"type": "text", "text": f"Observe the scene after {act_name}. {nav_instruction}"},
        ]})
    messages.append({"role": "assistant", "content": [
        {"type": "text", "text": f"<think>What should I do next?</think>{query_prefix}"},
    ]})
    return messages


def _encode_wm_state_nimloth(
    model: Any,
    processor: Any,
    image: Any,
    nav_instruction: str,
    action_history: list[str],
    state_proj: Any,
    device: Any,
    *,
    latent_token_count: int,
    observation_history: list[Any] | None = None,
) -> torch.Tensor:
    """Encode one real observation into a metadata-driven k-query WM state."""

    from nimloth.latent.extraction import (
        LatentActionTokens,
        extract_latent_state_block,
        find_last_latent_state_block,
        last_hidden_state,
        special_token_ids,
    )

    tokens = LatentActionTokens()
    token_ids = special_token_ids(
        processor.tokenizer, tokens, latent_token_count=latent_token_count
    )
    messages = _build_nimloth_policy_messages(
        image,
        nav_instruction,
        action_history,
        latent_token_count=latent_token_count,
        include_action_start=False,
        observation_history=observation_history,
    )
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = list(observation_history or [image] * (1 + len(action_history)))
    inputs = processor(
        text=[text], images=images, return_tensors="pt", padding=True
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    hidden = last_hidden_state(outputs)
    block = find_last_latent_state_block(
        inputs["input_ids"][0],
        token_ids,
        tokens,
        latent_token_count=latent_token_count,
    )
    latent = extract_latent_state_block(hidden[0:1], block).unsqueeze(0)
    return state_proj(latent).float().detach()


def action_sampling_logits(
    action_logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    """Apply the exact behavior-policy sampling transform to action logits."""

    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if temperature <= 0:
        transformed = torch.full_like(action_logits.float(), float("-inf"))
        transformed[action_logits.argmax()] = 0.0
        return transformed
    transformed = action_logits.float() / temperature
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(transformed, descending=True)
        cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        keep = cum_probs <= top_p
        keep[0] = True
        keep_mask = torch.zeros_like(transformed, dtype=torch.bool)
        keep_mask[sorted_indices[keep]] = True
        transformed = transformed.masked_fill(~keep_mask, float("-inf"))
    return transformed


def serialize_action_log_probs(log_probs: torch.Tensor) -> list[float | None]:
    """Encode zero-probability actions as JSON null rather than non-standard Infinity."""

    return [
        float(value) if torch.isfinite(value) else None
        for value in log_probs.detach().cpu()
    ]


def _select_action_nimloth(model, processor, image, nav_instruction: str,
                           action_history: list[str],
                           temperature: float = 1.0,
                           top_p: float = 1.0,
                           *,
                           latent_token_count: int = 1,
                           observation_history: list[Any] | None = None,
                           state_proj: Any = None,
                           ) -> tuple[str, int, list[float | None], torch.Tensor | None]:
    """Sampled action selection using Nimloth action tokens.

    The SFT2 model was trained with ``<|action_(0)|>`` … ``<|action_(7)|>``
    special tokens.  We build a Nimloth-format prompt (ending with
    ``<|action_start|>``), run Qwen forward, and extract the logits at
    ``<|action_start|>``.  Sampling with temperature + nucleus (top-p).

    Returns action name/index, behavior log-probs, and optionally the projected
    k-query state from the same Qwen forward.
    """
    import torch
    from nimloth.latent.extraction import (
        LatentActionTokens,
        extract_latent_state_block,
        find_last_latent_state_block,
        last_hidden_state,
        special_token_ids,
    )

    tokens = LatentActionTokens()
    token_ids = special_token_ids(
        processor.tokenizer, tokens, latent_token_count=latent_token_count
    )
    action_token_ids = [token_ids[t] for t in tokens.action_tokens]

    num_images = 1 + len(action_history)
    messages = _build_nimloth_policy_messages(
        image,
        nav_instruction,
        action_history,
        latent_token_count=latent_token_count,
        include_action_start=True,
        observation_history=observation_history,
    )

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = list(observation_history or [image] * num_images)
    inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)
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
    behavior_logits = action_sampling_logits(
        action_logits, temperature=temperature, top_p=top_p
    )
    action_log_probs = torch.log_softmax(behavior_logits, dim=-1)

    if temperature > 0:
        chosen_idx = int(torch.multinomial(torch.softmax(behavior_logits, dim=-1), 1).item())
    else:
        chosen_idx = int(action_logits.argmax().item())

    qwen_wm_state = None
    if state_proj is not None:
        latent_block = find_last_latent_state_block(
            input_ids,
            token_ids,
            tokens,
            latent_token_count=latent_token_count,
        )
        hidden = last_hidden_state(outputs)
        latent = extract_latent_state_block(hidden[0:1], latent_block).unsqueeze(0)
        qwen_wm_state = state_proj(latent).float().detach()

    serialized_log_probs = serialize_action_log_probs(action_log_probs)
    best_name = ACTION_NAMES[chosen_idx]
    return best_name, chosen_idx, serialized_log_probs, qwen_wm_state


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
