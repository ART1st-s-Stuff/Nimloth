"""Exact VAGEN-navigation protocol used by Nimloth rollout and SFT data.

The environment owns the system prompt and every observation string.  This
module only performs the same source-eval -> Nimloth token conversion used by
SFT preprocessing; it must not invent instructions, feedback, or rewards.
"""

from __future__ import annotations

import re
from typing import Any

from nimloth.latent.extraction import latent_state_block


ACTION_NAMES: tuple[str, ...] = (
    "move_forward",
    "move_backward",
    "move_right",
    "move_left",
    "turn_right",
    "turn_left",
    "look_up",
    "look_down",
)
ACTION_NAME_TO_IDX = {name: index for index, name in enumerate(ACTION_NAMES)}

# These are the aliases emitted by the pinned VAGEN source_eval_mode prompt.
_SOURCE_ACTION_TO_INDEX: dict[str, int] = {
    "move_forward": 0,
    "move_backward": 1,
    "move_right": 2,
    "move_left": 3,
    "turn_right": 4,
    "turn_left": 5,
    "look_up": 6,
    "look_down": 7,
}


def source_eval_text_to_nimloth(text: str, *, latent_token_count: int) -> str:
    """Apply ``convert_rollouts.rewrite_prompt_instruction`` exactly."""

    action_block = (
        latent_state_block(latent_token_count)
        + "<|action_start|><|action_(idx)|><|action_end|>"
    )
    legend = (
        "where idx is one of: 0=move_forward, 1=move_backward, 2=move_right, "
        "3=move_left, 4=turn_right, 5=turn_left, 6=look_up, 7=look_down."
    )
    instruction = (
        "Respond in this format:\n"
        f"<think>...</think>{action_block}\n{legend}"
    )
    converted = str(text)
    replacements = (
        (
            "You can optionally think first, then give your action. Respond in this format:\n"
            "<think>...</think><action>some_action</action>",
            "You can optionally think first, then give your action. " + instruction,
        ),
        (
            "Respond in this format:\n<think>...</think><action>some_action</action>",
            instruction,
        ),
        (
            "<think>...</think><action>some_action</action>",
            f"<think>...</think>{action_block}",
        ),
        (
            "<action>{action_example}</action>",
            "<|action_start|><|action_(idx)|><|action_end|>",
        ),
    )
    for source, target in replacements:
        converted = converted.replace(source, target)
    return re.sub(r"<action>\s*([^<]+?)\s*</action>", action_block, converted, flags=re.DOTALL)


def nimloth_assistant_response(thought: str, action_index: int, *, latent_token_count: int) -> str:
    """Create the inject-mode assistant response stored in SFT/RL history."""

    thought = thought.strip()
    if re.fullmatch(r"<think>.*?</think>", thought, flags=re.DOTALL) is None:
        raise ValueError("assistant thought must be one complete <think>...</think> block")
    if not 0 <= int(action_index) < len(ACTION_NAMES):
        raise ValueError(f"invalid navigation action index: {action_index}")
    return (
        f"{thought}{latent_state_block(latent_token_count)}"
        f"<|action_start|><|action_({int(action_index)})|><|action_end|>"
    )


def thought_from_assistant_response(response: str) -> str:
    """Extract the exact generated thought from an inject-mode response."""

    match = re.match(r"\s*(<think>.*?</think>)", str(response), flags=re.DOTALL)
    if match is None:
        raise ValueError("assistant response does not start with a complete thought block")
    return match.group(1)


def vagen_env_response(thought: str, action_index: int) -> str:
    """Convert one Nimloth decision back to the VAGEN source action syntax."""

    thought = thought.strip()
    if re.fullmatch(r"<think>.*?</think>", thought, flags=re.DOTALL) is None:
        raise ValueError("environment response requires one complete thought block")
    if not 0 <= int(action_index) < len(ACTION_NAMES):
        raise ValueError(f"invalid navigation action index: {action_index}")
    return f"{thought}<action>{ACTION_NAMES[int(action_index)]}</action>"


def normalize_vagen_policy_image(
    image: Any,
    *,
    max_pixels: int = 2048 * 2048,
    min_pixels: int = 512 * 512,
) -> Any:
    """Match the pinned VAGEN rollout manager's policy-image normalization.

    VAGEN calls ``verl.utils.dataset.rl_dataset.process_image`` before both
    policy inference and validation-image persistence.  Navigation emits raw
    255×255 frames, while SFT persisted512×512 frames. At the SFT2
    ``max_pixels=100352`` setting, omission yields81 image tokens instead of121.
    """

    import math
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise ValueError(f"navigation policy image must be PIL.Image, got {type(image)}")
    if min_pixels <= 0 or max_pixels < min_pixels:
        raise ValueError(
            f"invalid VAGEN image pixel bounds: min={min_pixels}, max={max_pixels}"
        )
    area = image.width * image.height
    if area > max_pixels:
        factor = math.sqrt(max_pixels / area)
        image = image.resize((int(image.width * factor), int(image.height * factor)))
    area = image.width * image.height
    if area < min_pixels:
        factor = math.sqrt(min_pixels / area)
        image = image.resize((int(image.width * factor), int(image.height * factor)))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def observation_text_and_image(
    observation: Any, *, latent_token_count: int
) -> tuple[str, Any]:
    """Extract the policy-visible VAGEN observation without dropping text.

    The pinned navigation environment returns ``obs_str`` plus exactly one
    image under ``multi_modal_data['<image>']``.  Accepting image-only inputs here
    would recreate the P0 bug, so malformed observations fail closed.
    """

    if not isinstance(observation, dict):
        raise ValueError(
            "navigation observation must be a dict containing obs_str and multi_modal_data"
        )
    obs_str = observation.get("obs_str")
    if not isinstance(obs_str, str) or not obs_str.strip():
        raise ValueError("navigation observation is missing non-empty obs_str")
    multi_modal = observation.get("multi_modal_data")
    if not isinstance(multi_modal, dict):
        raise ValueError("navigation observation is missing multi_modal_data")
    images = multi_modal.get("<image>")
    if not isinstance(images, list) or len(images) != 1:
        raise ValueError(
            "navigation observation must contain exactly one multi_modal_data['<image>']"
        )
    converted = source_eval_text_to_nimloth(
        obs_str, latent_token_count=latent_token_count
    )
    if converted.count("<image>") != 1:
        raise ValueError(
            "navigation obs_str must contain exactly one <image> placeholder; "
            f"got {converted.count('<image>')}"
        )
    return converted, normalize_vagen_policy_image(images[0])


def extract_human_instruction(observation_text: str) -> str:
    """Return the task text embedded by VAGEN in the initial observation."""

    match = re.search(
        r"Human Instruction:\s*(.*?)\s*\n(?:Decide your next action\(s\)\.|<image>)",
        observation_text,
        flags=re.DOTALL,
    )
    if match is None or not match.group(1).strip():
        raise ValueError("initial observation does not contain a Human Instruction")
    return match.group(1).strip()


def trajectory_messages(
    system_prompt: str,
    observation_texts: list[str],
    assistant_responses: list[str],
    *,
    history_window: int,
) -> list[dict[str, str]]:
    """Reconstruct the same alternating history consumed during SFT.

    ``observation_texts`` must contain the current observation, hence exactly
    one more entry than completed assistant responses.  ``history_window`` is
    measured in completed turns, matching VAGEN's rollout manager. A value
    large enough for the episode preserves the full SFT teacher-forcing prefix;
    the source VAGEN collector generated with ``window_size=5``.
    """

    if not str(system_prompt).strip():
        raise ValueError("system prompt must be non-empty")
    if len(observation_texts) != len(assistant_responses) + 1:
        raise ValueError(
            "policy history requires one more observation than assistant response: "
            f"observations={len(observation_texts)}, responses={len(assistant_responses)}"
        )
    if history_window < 0:
        raise ValueError(f"history_window must be >= 0, got {history_window}")
    start = max(0, len(assistant_responses) - history_window)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": str(system_prompt)},
        {"role": "user", "content": observation_texts[start]},
    ]
    for index in range(start, len(assistant_responses)):
        messages.append({"role": "assistant", "content": assistant_responses[index]})
        messages.append({"role": "user", "content": observation_texts[index + 1]})
    return messages


def task_succeeded(info: Any) -> bool:
    """Read success from VAGEN's explicit metrics instead of reward thresholds."""

    if not isinstance(info, dict):
        return False
    if "task_success" in info:
        return bool(info["task_success"])
    metrics = info.get("metrics")
    if isinstance(metrics, dict):
        trajectory = metrics.get("traj_metrics")
        if isinstance(trajectory, dict) and "success" in trajectory:
            return bool(trajectory["success"])
    return False
