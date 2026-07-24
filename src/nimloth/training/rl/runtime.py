"""RL 单批算法所需的模型执行契约。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nimloth.agent import ActionLogProbReplay, Agent, AgentPrompt
from nimloth.backbone import BackboneInputBuilder
from nimloth.util.module import evaluating


@dataclass(frozen=True)
class RLModelRuntime:
    """执行 RL 的 Backbone 表征与可选 actor 概率重放。"""

    agent: Agent
    input_builder: BackboneInputBuilder
    representation_to_backbone: bool
    policy_replay: ActionLogProbReplay | None

    def encode_state_sequence(
        self,
        prompts: tuple[AgentPrompt, ...],
        *,
        batch_size: int,
        state_steps: int,
    ) -> torch.Tensor:
        """按配置保留或截断表征目标到 Backbone 的计算图。"""

        expected = batch_size * state_steps
        if len(prompts) != expected:
            raise ValueError(
                f"RL state prompt count must be {expected}, got {len(prompts)}"
            )
        backbone_batch = self.input_builder.build(
            [prompt.unbound_messages() for prompt in prompts],
            [prompt.images for prompt in prompts],
            include_labels=False,
        )
        if self.representation_to_backbone:
            hidden = self.agent.backbone(
                backbone_batch,
                include_lm_loss=False,
            ).hidden
        else:
            # 冻结表征模式只截断 WM/value/SIGReg 到 Backbone 的梯度；PPO replay
            # 仍可按 actor 配置独立训练同一个 Backbone。
            with evaluating(self.agent.backbone), torch.no_grad():
                hidden = self.agent.backbone(
                    backbone_batch,
                    include_lm_loss=False,
                ).hidden.detach()
        if hidden.ndim not in (2, 3) or hidden.shape[0] != expected:
            raise ValueError(
                "RL Backbone hidden must have shape (B*T, D) or (B*T, k, D), "
                f"got {tuple(hidden.shape)} for B*T={expected}"
            )
        # Qwen 单 latent token 返回 (B*T,D)，多 token 返回 (B*T,k,D)。
        # 时间轴只在这里恢复，token 轴继续由 StateProjector 解释。
        return hidden.reshape(batch_size, state_steps, *hidden.shape[1:])


__all__ = ["RLModelRuntime"]
