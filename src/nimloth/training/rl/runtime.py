"""RL 单批算法所需的模型执行契约。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nimloth.agent import ActionLogProbReplay, Agent, AgentPrompt
from nimloth.backbone import BackboneInputBuilder, DINOGridTargets
from nimloth.util.module import evaluating


@dataclass(frozen=True)
class RLModelRuntime:
    """执行 RL 的 Backbone 表征与可选 actor 概率重放。"""

    agent: Agent
    input_builder: BackboneInputBuilder
    state_source: str
    representation_to_backbone: bool
    policy_replay: ActionLogProbReplay | None
    dino_grid_targets: DINOGridTargets | None = None

    def encode_state_prompts(
        self,
        prompts: tuple[AgentPrompt, ...],
    ) -> torch.Tensor:
        """Recompute complete real prefixes while retaining the Qwen graph."""

        if self.state_source != "recompute":
            raise ValueError("encode_state_prompts requires state_source=recompute")
        if not prompts:
            raise ValueError("state prompt batch must not be empty")
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
            with evaluating(self.agent.backbone), torch.no_grad():
                hidden = self.agent.backbone(
                    backbone_batch,
                    include_lm_loss=False,
                ).hidden.detach()
        if hidden.ndim not in (2, 3) or hidden.shape[0] != len(prompts):
            raise ValueError(
                "RL Backbone output must have shape (B,D) or (B,K,D), "
                f"got {tuple(hidden.shape)} for B={len(prompts)}"
            )
        return hidden

    def encode_state_sequence(
        self,
        prompts: tuple[AgentPrompt, ...],
        *,
        batch_size: int,
        state_steps: int,
    ) -> torch.Tensor:
        """按配置保留或截断表征目标到 Backbone 的计算图。"""

        if self.state_source != "recompute":
            raise ValueError("encode_state_sequence requires state_source=recompute")
        expected = batch_size * state_steps
        if len(prompts) != expected:
            raise ValueError(
                f"RL state prompt count must be {expected}, got {len(prompts)}"
            )
        step_outputs: list[torch.Tensor] = []
        for step in range(state_steps):
            step_prompts = prompts[step::state_steps]
            hidden = self.encode_state_prompts(step_prompts)
            step_outputs.append(hidden)
        return torch.stack(step_outputs, dim=1)

    def validate_rollout_state_hiddens(
        self,
        hidden_states: torch.Tensor,
        *,
        batch_size: int,
        state_steps: int,
    ) -> None:
        """校验 rollout hidden 与当前 StateProjector 的输入形状一致。"""

        if self.state_source != "rollout":
            raise ValueError(
                "validate_rollout_state_hiddens requires state_source=rollout"
            )
        if self.representation_to_backbone:
            raise ValueError(
                "rollout-cached Qwen states require "
                "gradient.representation_to_backbone=false"
            )
        state_proj = self.agent.wm.state_proj
        projector = getattr(state_proj, "module", state_proj)
        expected_shape = (
            batch_size,
            state_steps,
            int(projector.latent_token_count),
            int(projector.qwen_hidden_dim),
        )
        if tuple(hidden_states.shape) != expected_shape:
            raise ValueError(
                "cached Qwen states must have shape "
                f"{expected_shape}, got {tuple(hidden_states.shape)}"
            )


__all__ = ["RLModelRuntime"]
