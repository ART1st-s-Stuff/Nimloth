"""使用 Qwen 把结构化 rollout 编码为训练可消费的 latent transition。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from nimloth.agent import PROMPT_VERSION
from nimloth.backbone.qwen25vl.batch import build_qwen_batch
from nimloth.latent.extraction import (
    LatentActionTokens,
    extract_latent_state,
    find_last_latent_state_index,
    last_hidden_state,
)
from nimloth.rollout.schema import RolloutTrajectory, validate_rollout_trajectory
from nimloth.rollout.transitions import discounted_action_value_targets


@dataclass(frozen=True)
class EncodedRolloutTransition:
    """一条 rollout transition 的 Qwen state 与 policy provenance。"""

    qwen_hidden_current: torch.Tensor
    qwen_hidden_next: torch.Tensor
    action_index: int
    value_target: float
    old_log_prob: float
    policy_messages: list[dict[str, Any]]
    policy_image_paths: list[str]
    sampling_temperature: float
    sampling_top_p: float
    latent_token_count: int


def encode_trajectory_states(
    trajectory: RolloutTrajectory,
    qwen_model: torch.nn.Module,
    processor: Any,
    token_id_map: dict[str, int],
    device: torch.device,
) -> list[torch.Tensor]:
    """逐个 policy state 编码，返回包含最终状态的 CPU hidden 列表。"""

    if trajectory.prompt_version != PROMPT_VERSION:
        raise ValueError(
            f"trajectory {trajectory.record_id!r} uses prompt_version "
            f"{trajectory.prompt_version!r}; expected {PROMPT_VERSION!r}"
        )
    if not trajectory.system_prompt or not trajectory.observation_texts:
        raise ValueError(
            f"trajectory {trajectory.record_id!r} has no structured Agent transcript"
        )

    states: list[torch.Tensor] = []
    tokens = LatentActionTokens()
    for step_index in range(len(trajectory.image_paths)):
        messages = trajectory.build_policy_messages(step_index, bind_images=True)
        encoding = build_qwen_batch(
            [{"messages": messages}],
            processor,
            max_length=999999,
            latent_token_count=trajectory.latent_token_count,
        )
        model_inputs = {key: value.to(device) for key, value in encoding.items()}
        with torch.no_grad():
            output = qwen_model(
                **model_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = last_hidden_state(output)
        latent_index = find_last_latent_state_index(
            encoding["input_ids"][0],
            token_id_map,
            tokens,
        )
        latent = extract_latent_state(hidden[0:1], latent_index)
        states.append(latent.squeeze(0).detach().cpu())
    return states


def encode_rollout_transitions(
    trajectories: list[RolloutTrajectory],
    qwen_model: torch.nn.Module,
    processor: Any,
    token_id_map: dict[str, int],
    device: torch.device,
    *,
    gamma: float = 0.99,
) -> list[EncodedRolloutTransition]:
    """校验 trajectory，并展开为带 Qwen latent state 的 transition。"""

    transitions: list[EncodedRolloutTransition] = []
    for trajectory in trajectories:
        validate_rollout_trajectory(trajectory)
        hidden_states = encode_trajectory_states(
            trajectory,
            qwen_model,
            processor,
            token_id_map,
            device,
        )
        if len(hidden_states) < 2:
            continue
        value_targets = discounted_action_value_targets(
            trajectory.to_record(),
            gamma=gamma,
        )
        for step_index in range(trajectory.num_steps):
            action_index = trajectory.action_indices[step_index]
            transitions.append(
                EncodedRolloutTransition(
                    qwen_hidden_current=hidden_states[step_index],
                    qwen_hidden_next=hidden_states[step_index + 1],
                    action_index=action_index,
                    value_target=float(value_targets[step_index]),
                    old_log_prob=float(
                        trajectory.action_log_probs[step_index][action_index]
                    ),
                    policy_messages=trajectory.build_policy_messages(
                        step_index,
                        bind_images=False,
                    ),
                    policy_image_paths=trajectory.image_paths[: step_index + 1],
                    sampling_temperature=trajectory.sampling_temperature,
                    sampling_top_p=trajectory.sampling_top_p,
                    latent_token_count=trajectory.latent_token_count,
                )
            )
    return transitions
