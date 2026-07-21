"""RL 阶段对保存 policy prompt 进行 PPO replay 的适配。"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from nimloth.agent import bind_image_placeholders
from nimloth.backbone.qwen25vl.policy import batch_action_log_probs
from nimloth.rollout.encoding import EncodedRolloutTransition


def compute_current_policy_log_probs(
    transitions: Sequence[EncodedRolloutTransition],
    model: torch.nn.Module,
    processor: Any,
    token_id_map: dict[str, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """用当前 Qwen 重放 rollout 时的精确 prompt 与采样变换。"""

    if not transitions:
        raise ValueError("PPO policy batch must not be empty")
    latent_token_counts = {
        transition.latent_token_count for transition in transitions
    }
    if len(latent_token_counts) != 1:
        raise ValueError("one PPO batch cannot mix latent token counts")
    bound_messages = [
        bind_image_placeholders(
            transition.policy_messages,
            transition.policy_image_paths,
        )
        for transition in transitions
    ]
    return batch_action_log_probs(
        model=model,
        processor=processor,
        token_id_map=token_id_map,
        messages=bound_messages,
        taken_action_indices=[
            transition.action_index for transition in transitions
        ],
        temperatures=[
            transition.sampling_temperature for transition in transitions
        ],
        top_ps=[transition.sampling_top_p for transition in transitions],
        device=device,
        latent_token_count=latent_token_counts.pop(),
    )
