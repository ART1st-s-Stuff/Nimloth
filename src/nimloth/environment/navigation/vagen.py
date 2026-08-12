"""VAGEN navigation server 到 Nimloth environment session 的适配。"""

from __future__ import annotations

import re
from typing import Any

from PIL import Image

from nimloth.environment.common.action_space import DiscreteActionSpace
from nimloth.environment.common.session import EnvironmentObservation, EnvironmentStep
from nimloth.environment.navigation.action_space import NAVIGATION_ACTION_SPACE


_INSTRUCTION_RE = re.compile(r"Human Instruction:\s*(.+?)(?:\n|$)")
NAVIGATION_REQUEST_TIMEOUT_SECONDS = 300


def instruction_from_observation(observation_text: str) -> str:
    """从 VAGEN 初始 observation 中提取 navigation instruction。"""

    match = _INSTRUCTION_RE.search(observation_text)
    return match.group(1).strip() if match else ""


def observation_text(raw_observation: Any) -> str:
    """读取 environment 原样提供、且包含图片占位符的文本。"""

    if isinstance(raw_observation, dict):
        text = raw_observation.get("obs_str")
        if isinstance(text, str) and text.strip():
            if "<image>" not in text:
                raise ValueError("VAGEN obs_str has no <image> placeholder")
            return text
    raise ValueError("environment observation has no non-empty obs_str")


def observation_image(raw_observation: Any) -> Image.Image:
    """把 VAGEN observation 中的第一张图转换为 RGB PIL image。"""

    if isinstance(raw_observation, Image.Image):
        return validate_navigation_image(raw_observation)
    if hasattr(raw_observation, "shape"):
        return validate_navigation_image(Image.fromarray(raw_observation))
    if not isinstance(raw_observation, dict):
        raise ValueError(f"unknown observation type: {type(raw_observation)}")

    for key in ("image", "rgb", "pixels"):
        if key in raw_observation:
            return validate_navigation_image(_image_value(raw_observation[key]))

    for container_key in ("multi_modal_input", "multi_modal_data"):
        multi_modal_data = raw_observation.get(container_key, {})
        if not isinstance(multi_modal_data, dict):
            continue
        preferred_keys = ("<image>", "image", "images", "rgb", "pixels")
        values_by_priority = [
            multi_modal_data[key]
            for key in preferred_keys
            if key in multi_modal_data
        ]
        values_by_priority.extend(
            value
            for key, value in multi_modal_data.items()
            if key not in preferred_keys
        )
        for values in values_by_priority:
            if isinstance(values, list) and values:
                return validate_navigation_image(_image_value(values[0]))
    raise ValueError(
        f"cannot extract image from observation keys {list(raw_observation.keys())}"
    )


def validate_navigation_image(image: Image.Image) -> Image.Image:
    """拒绝AI2-THOR/Vulkan故障产生的纯色伪观测。"""

    rgb = image.convert("RGB")
    extrema = rgb.getextrema()
    dynamic_range = max(high - low for low, high in extrema)
    if dynamic_range == 0:
        raise RuntimeError(
            "navigation observation is a uniform image; "
            f"AI2-THOR/Vulkan rendering is invalid: extrema={extrema}"
        )
    return rgb


def navigation_image_dynamic_range(image: Image.Image) -> int:
    """返回RGB通道内最大的像素动态范围，供启动门禁记录。"""

    extrema = image.convert("RGB").getextrema()
    return max(high - low for low, high in extrema)


def vagen_eval_nimloth_system_prompt(
    *,
    latent_token_count: int = 16,
) -> str:
    """Rebuild the source-eval wording on the upstream Nimloth protocol."""

    from vagen.envs.navigation.utils.prompt import system_prompt

    return system_prompt(
        format_name="nimloth",
        max_actions_per_step=1,
        action_sep="|",
        example_count=0,
        latent_token_count=latent_token_count,
    )


def vagen_eval_nimloth_observation_text(
    raw_observation: Any,
    *,
    initial: bool,
    latent_token_count: int = 16,
) -> str:
    """Match the SFT1-converted source-eval observation wording exactly."""

    text = observation_text(raw_observation)
    if not initial:
        return text.replace("After your answer,", "After your action,").replace(
            "Decide your next action.", "Decide your next action(s)."
        )
    instruction = instruction_from_observation(text)
    if not instruction:
        raise ValueError("VAGEN initial observation has no navigation instruction")
    from vagen.envs.navigation.utils.prompt import get_format_instruction

    return (
        "[Initial Observation]:\n<image>\n"
        f"Human Instruction: {instruction}\n"
        "Decide your next action(s).\n"
        + get_format_instruction(
            "nimloth",
            max_actions_per_step=1,
            action_sep="|",
            latent_token_count=latent_token_count,
        )
    )


def _image_value(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if hasattr(value, "shape"):
        return Image.fromarray(value).convert("RGB")
    if isinstance(value, dict) and "__pil_image__" in value:
        import base64
        import io

        return Image.open(
            io.BytesIO(base64.b64decode(value["__pil_image__"]))
        ).convert("RGB")
    raise ValueError(f"unsupported environment image value: {type(value)}")


def navigation_environment_config(
    eval_set: str,
    *,
    profile: str = "current",
    latent_token_count: int = 16,
) -> dict[str, Any]:
    """构造当前 Nimloth navigation rollout 使用的 VAGEN 配置。"""

    if profile not in {"current", "vagen_eval"}:
        raise ValueError(f"unknown navigation profile: {profile!r}")
    if (
        isinstance(latent_token_count, bool)
        or not isinstance(latent_token_count, int)
        or latent_token_count < 1
    ):
        raise ValueError("latent_token_count must be a positive int")
    config = {
        "prompt_format": "nimloth",
        "latent_token_count": latent_token_count,
        "eval_set": eval_set,
        "max_actions_per_step": 1,
        "action_sep": "|",
        "example_count": 0,
        "format_reward": 0.0,
        "per_turn_format_reward": 0.0,
        "success_reward": 10.0,
        "success_threshold": 1.5,
        "step_length": 0.5,
        "gpu_device": 0,
    }
    if profile == "vagen_eval":
        config.update(
            {
                "format_reward": 0.0,
                "per_turn_format_reward": 0.01,
                "success_reward": 1.0,
                "success_threshold": 1.0,
                "step_length": 0.3,
            }
        )
    return config


class VAGENNavigationSession:
    """一个 VAGEN navigation episode 的资源和生命周期。"""

    def __init__(
        self,
        *,
        client: Any,
        episode_id: str,
        eval_set: str,
        failure_penalty: float = 0.1,
        navigation_profile: str = "current",
        latent_token_count: int = 16,
    ) -> None:
        self._client = client
        self._episode_id = episode_id
        self._eval_set = eval_set
        self._failure_penalty = failure_penalty
        self._navigation_profile = navigation_profile
        self._latent_token_count = int(latent_token_count)
        self._system_prompt = ""
        self._created = False

    @property
    def action_space(self) -> DiscreteActionSpace:
        return NAVIGATION_ACTION_SPACE

    @property
    def system_prompt(self) -> str:
        if not self._system_prompt:
            raise RuntimeError("environment session has not been reset")
        return self._system_prompt

    def reset(self, *, seed: int) -> EnvironmentObservation:
        self._client.create_environments_batch(
            {
                self._episode_id: navigation_environment_config(
                    self._eval_set,
                    profile=self._navigation_profile,
                    latent_token_count=self._latent_token_count,
                )
            }
        )
        self._created = True
        raw_observation, info = self._client.reset_batch(
            {self._episode_id: seed}
        )[self._episode_id]
        prompts = self._client.get_system_prompts_batch([self._episode_id])
        self._system_prompt = str(prompts.get(self._episode_id, ""))
        if self._navigation_profile == "vagen_eval":
            self._system_prompt = vagen_eval_nimloth_system_prompt(
                latent_token_count=self._latent_token_count
            )
        if not self._system_prompt:
            raise RuntimeError(
                f"environment {self._episode_id} returned an empty system prompt"
            )
        return EnvironmentObservation(
            text=(
                vagen_eval_nimloth_observation_text(
                    raw_observation,
                    initial=True,
                    latent_token_count=self._latent_token_count,
                )
                if self._navigation_profile == "vagen_eval"
                else observation_text(raw_observation)
            ),
            image=observation_image(raw_observation),
            info=dict(info) if isinstance(info, dict) else {},
        )

    def step(self, *, action_index: int, response: str) -> EnvironmentStep:
        self.action_space.validate_index(action_index)
        raw_observation, reward, done, info = self._client.step_batch(
            {self._episode_id: response}
        )[self._episode_id]
        info_dict = dict(info) if isinstance(info, dict) else {}
        adjusted_reward = float(reward)
        if not info_dict.get("last_action_success", True):
            adjusted_reward -= self._failure_penalty
        # VAGEN navigation 的稀疏成功奖励为 10；旧服务可能不返回 task_success。
        success = bool(info_dict.get("task_success", False)) or adjusted_reward >= 10.0
        return EnvironmentStep(
            observation=EnvironmentObservation(
                text=(
                    vagen_eval_nimloth_observation_text(
                        raw_observation,
                        initial=False,
                        latent_token_count=self._latent_token_count,
                    )
                    if self._navigation_profile == "vagen_eval"
                    else observation_text(raw_observation)
                ),
                image=observation_image(raw_observation),
                info=info_dict,
            ),
            reward=adjusted_reward,
            done=bool(done),
            success=success,
            info=info_dict,
        )

    def close(self) -> None:
        if not self._created:
            return
        try:
            self._client.close_batch([self._episode_id])
        finally:
            self._created = False
