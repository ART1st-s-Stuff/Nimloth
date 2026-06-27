"""Navigation environment manager wrapping VAGEN's NavigationService.

Provides batch environment lifecycle (create / reset / step / close) over
AI2-THOR navigation tasks, plus trajectory recording in Nimloth WM format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nimloth.environment.common.base import BaseEnvManager
from nimloth.environment.common.types import EnvConfig, StepResult, TrajectoryRecording


class NavigationEnvManager(BaseEnvManager):
    """Manage a fleet of AI2-THOR navigation environments.

    Thin wrapper around VAGEN's :class:`~vagen.env.navigation.service.NavigationService`
    that adds Nimloth-format trajectory recording and action-index mapping.

    Usage::

        mgr = NavigationEnvManager(
            output_dir="/tmp/nav_rollouts",
            gpu_devices=[0, 1],
        )
        mgr.reset([EnvConfig(env_name="navigation", env_config={...}, seed=42)])
        while mgr.active_env_ids():
            actions = policy(mgr.active_obs())   # list[str]
            mgr.step(actions)
        trajectories = mgr.get_trajectories()
    """

    def __init__(
        self,
        *,
        output_dir: str | Path,
        gpu_devices: list[int] | None = None,
        max_workers: int = 4,
    ) -> None:
        import sys

        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._gpu_devices = gpu_devices or [0]
        self._max_workers = max_workers

        # Lazy-import VAGEN so the module is importable even without ai2thor
        # (only fails when actually used).
        from vagen.env.navigation.service import NavigationService
        from vagen.env.navigation.service_config import NavigationServiceConfig

        svc_cfg = NavigationServiceConfig(
            max_workers=self._max_workers,
            devices=self._gpu_devices,
        )
        self._service: NavigationService = NavigationService(svc_cfg)

        # Per-env bookkeeping
        self._env_configs: dict[int, EnvConfig] = {}
        self._done: dict[int, bool] = {}
        self._active_order: list[int] = []  # env_ids in the order reset() gives
        self._step_counts: dict[int, int] = {}

        # Recording
        self._recordings: dict[int, TrajectoryRecording] = {}
        self._image_counter: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, env_configs: list[EnvConfig]) -> list[dict[str, Any]]:
        """Create environments for *env_configs* and return initial observations.

        Existing environments that are no longer needed are closed first.
        """
        self._close_all()
        self._env_configs.clear()
        self._done.clear()
        self._active_order.clear()
        self._step_counts.clear()
        self._recordings.clear()

        if not env_configs:
            return []

        ids2configs: dict[str, Any] = {}
        for i, cfg in enumerate(env_configs):
            self._env_configs[i] = cfg
            self._done[i] = False
            self._step_counts[i] = 0
            self._active_order.append(i)
            ids2configs[str(i)] = {
                "env_name": cfg.env_name,
                "env_config": dict(cfg.env_config),
                "seed": cfg.seed,
            }

        self._service.create_environments_batch(ids2configs)
        seeds = {str(i): cfg.seed for i, cfg in self._env_configs.items()}
        obs_dict = self._service.reset_batch(seeds)

        # Build initial-observation list in active-order
        results: list[dict[str, Any]] = []
        for i in self._active_order:
            obs_i, info_i = obs_dict.get(str(i), ({}, {}))
            results.append({"obs": obs_i, "info": info_i})
        return results

    def step(self, actions: list[str]) -> list[StepResult]:
        """Execute one action per active environment."""
        active = self.active_env_ids()
        if len(actions) != len(active):
            raise ValueError(
                f"Expected {len(active)} actions, got {len(actions)}"
            )

        ids2actions = {str(env_id): action for env_id, action in zip(active, actions)}
        step_results_raw = self._service.step_batch(ids2actions)

        results: list[StepResult] = []
        for env_id in active:
            obs, reward, done, info = step_results_raw[str(env_id)]
            self._step_counts[env_id] = self._step_counts.get(env_id, 0) + 1
            if done:
                self._done[env_id] = True

            # Extract PIL image from the multi_modal_data dict.
            pil_image = self._extract_image(obs)

            result = StepResult(
                env_id=env_id,
                obs_str=obs.get("obs_str", ""),
                image=pil_image,
                reward=float(reward),
                done=bool(done),
                info=dict(info),
            )
            results.append(result)

            # Auto-record if tracking is active for this env
            if env_id in self._recordings:
                self.record_step(env_id, result)

        return results

    def active_env_ids(self) -> list[int]:
        return [i for i in self._active_order if not self._done.get(i, True)]

    def is_done(self, env_id: int) -> bool:
        return self._done.get(env_id, True)

    def close(self) -> None:
        self._close_all()

    def _close_all(self) -> None:
        try:
            self._service.close_batch(None)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def start_recording(self, env_id: int, system_prompt: str = "") -> None:
        cfg = self._env_configs.get(env_id)
        rec = TrajectoryRecording(
            env_id=env_id,
            env_name=cfg.env_name if cfg else "",
            eval_set=cfg.env_config.get("eval_set", "") if cfg else "",
            seed=cfg.seed if cfg else 0,
            instruction=cfg.env_config.get("instruction", "") if cfg else "",
        )
        if system_prompt:
            rec.messages.append({"role": "system", "content": system_prompt})
        self._recordings[env_id] = rec

    def record_step(self, env_id: int, result: StepResult) -> None:
        rec = self._recordings.get(env_id)
        if rec is None:
            return

        # Save image to disk
        img_path = self._output_dir / f"img_{self._image_counter:06d}.png"
        self._image_counter += 1
        if result.image is not None:
            result.image.save(img_path)
        img_path_str = str(img_path)

        # Map action name → index using VAGEN's lookup
        action_name = result.info.get("actions", [None])[0] if isinstance(result.info.get("actions"), list) else result.info.get("action")
        action_idx = _action_name_to_index(action_name)

        # --- image_paths alignment ---
        # image_paths[t] is the observation *before* taking action_indices[t].
        # The first image was already saved in the initial observation (handled
        # by _save_initial_image). Each step adds the *next* image.
        rec.image_paths.append(img_path_str)

        # --- messages ---
        # User turn: the observation text (with <image> placeholder for the image)
        rec.messages.append({
            "role": "user",
            "content": _build_user_message(result.obs_str),
        })
        # Assistant turn: the action taken
        action_token = _action_idx_to_token(action_idx)
        rec.messages.append({
            "role": "assistant",
            "content": f"<|latent_state|><|action_start|>{action_token}<|action_end|>",
        })

        if action_idx >= 0:
            rec.action_indices.append(action_idx)
        rec.action_names.append(action_name or "unknown")
        rec.reward += result.reward
        rec.success = rec.success or result.info.get("task_success", False) or result.info.get("metrics", {}).get("traj_metrics", {}).get("success", False)
        rec.done = result.done
        rec.num_steps += 1

    def save_initial_image(self, env_id: int, obs: dict[str, Any]) -> None:
        """Save the first observation image for *env_id* and add to recording.

        Call this right after :meth:`reset` for each environment before the
        first :meth:`step`.
        """
        rec = self._recordings.get(env_id)
        if rec is None:
            return
        pil_image = self._extract_image(obs)
        img_path = self._output_dir / f"img_{self._image_counter:06d}.png"
        self._image_counter += 1
        if pil_image is not None:
            pil_image.save(img_path)
        rec.image_paths.append(str(img_path))
        rec.messages.append({
            "role": "user",
            "content": _build_user_message(obs.get("obs_str", "")),
        })

    def get_trajectories(self) -> list[TrajectoryRecording]:
        """Return recordings of all finished environments."""
        return [
            rec for env_id, rec in self._recordings.items()
            if self._done.get(env_id, False)
        ]

    def clear_recordings(self) -> None:
        self._recordings.clear()
        self._image_counter = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_image(obs: dict[str, Any]) -> Any | None:
        """Pull the first PIL image from a VAGEN observation dict."""
        mm = obs.get("multi_modal_data", {})
        if not mm:
            return None
        # mm is {image_placeholder: [PIL.Image, ...]}
        for images in mm.values():
            if isinstance(images, list) and images:
                return images[0]
        return None


# ---------------------------------------------------------------------------
# Action mapping helpers (inlined to avoid circular imports with wm.dataset)
# ---------------------------------------------------------------------------

_ACTION_NAME_TO_IDX: dict[str, int] = {
    "move_forward": 0,
    "moveahead": 0,
    "move_backward": 1,
    "moveback": 1,
    "move_right": 2,
    "moveright": 2,
    "move_left": 3,
    "moveleft": 3,
    "turn_right": 4,
    "rotateright": 4,
    "turn_left": 5,
    "rotateleft": 5,
    "look_up": 6,
    "lookup": 6,
    "look_down": 7,
    "lookdown": 7,
}

_IDX_TO_TOKEN: dict[int, str] = {
    i: f"<|action_({i})|>" for i in range(8)
}


def _action_name_to_index(name: str | None) -> int:
    if name is None:
        return -1
    return _ACTION_NAME_TO_IDX.get(name.lower(), -1)


def _action_idx_to_token(idx: int) -> str:
    return _IDX_TO_TOKEN.get(idx, f"<|action_({idx})|>")


def _build_user_message(obs_str: str) -> str:
    """Build a user message from the observation string.

    The VAGEN observation already contains ``<image>`` placeholders, so we
    can use the obs_str directly.
    """
    return obs_str
