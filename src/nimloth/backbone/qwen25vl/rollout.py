"""使用 Qwen 把结构化 rollout 编码为 latent transition。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from nimloth.agent import NIMLOTH_PROMPT_TEMPLATE_ID, PROMPT_VERSION
from nimloth.backbone.qwen25vl.batch import build_qwen_batch
from nimloth.latent.extraction import (
    LatentActionTokens,
    extract_latent_state,
    find_last_latent_state_index,
    last_hidden_state,
)
from nimloth.rollout.schema import RolloutTrajectory
from nimloth.rollout.encoding import EncodedTrajectory, EncodedTransition
from nimloth.rollout.transitions import discounted_action_value_targets
from nimloth.rollout.validation import validate_rollout_trajectory
from nimloth.util.module import evaluating


EncodedRolloutTransition = EncodedTransition


@dataclass(frozen=True)
class QwenRolloutEncoder:
    """使用一个 Qwen backbone 将 trajectory 编码成 RL transition。"""

    model: torch.nn.Module
    processor: Any
    token_id_map: dict[str, int]
    device: torch.device

    def __call__(
        self,
        trajectories: Sequence[RolloutTrajectory],
        *,
        gamma: float,
    ) -> list[EncodedTrajectory]:
        return encode_rollout_trajectories(
            list(trajectories),
            self.model,
            self.processor,
            self.token_id_map,
            self.device,
            gamma=gamma,
        )


def encode_trajectory_states(
    trajectory: RolloutTrajectory,
    qwen_model: torch.nn.Module,
    processor: Any,
    token_id_map: dict[str, int],
    device: torch.device,
) -> list[torch.Tensor]:
    """逐个 policy state 编码，返回包含最终状态的 CPU hidden 列表。"""

    prompt_spec = trajectory.resolved_prompt_template_spec()
    if (
        prompt_spec.identifier != NIMLOTH_PROMPT_TEMPLATE_ID
        or prompt_spec.version != PROMPT_VERSION
    ):
        raise ValueError(
            f"trajectory {trajectory.record_id!r} uses unsupported Qwen "
            f"prompt template {prompt_spec.identifier}@{prompt_spec.version}"
        )
    if not trajectory.system_prompt or not trajectory.observation_texts:
        raise ValueError(
            f"trajectory {trajectory.record_id!r} has no structured Agent transcript"
        )

    states: list[torch.Tensor] = []
    latent_token_count = trajectory.resolved_latent_token_count()
    tokens = LatentActionTokens()
    # State 编码和行为 policy 使用相同的确定性 eval-mode Qwen。
    with evaluating(qwen_model), torch.no_grad():
        for step_index in range(len(trajectory.image_paths)):
            messages = trajectory.build_policy_messages(step_index, bind_images=True)
            encoding = build_qwen_batch(
                [{"messages": messages}],
                processor,
                max_length=999999,
                latent_token_count=latent_token_count,
            )
            model_inputs = {key: value.to(device) for key, value in encoding.items()}
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


def encode_rollout_trajectories(
    trajectories: list[RolloutTrajectory],
    qwen_model: torch.nn.Module,
    processor: Any,
    token_id_map: dict[str, int],
    device: torch.device,
    *,
    gamma: float = 0.99,
) -> list[EncodedTrajectory]:
    """校验 trajectory，并保留其连续 Qwen latent transition。"""

    encoded_trajectories: list[EncodedTrajectory] = []
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
        transitions: list[EncodedTransition] = []
        for step_index in range(trajectory.num_steps):
            action_index = trajectory.action_indices[step_index]
            transitions.append(
                EncodedTransition(
                    record_id=trajectory.record_id,
                    step_index=step_index,
                    current_hidden=hidden_states[step_index],
                    next_hidden=hidden_states[step_index + 1],
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
                    latent_token_count=trajectory.resolved_latent_token_count(),
                )
            )
        if transitions:
            encoded_trajectories.append(
                EncodedTrajectory(
                    record_id=trajectory.record_id,
                    transitions=tuple(transitions),
                )
            )
    return encoded_trajectories
