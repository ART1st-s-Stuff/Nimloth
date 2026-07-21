"""SFT2 单个 batch 的模型前向编排。"""

from __future__ import annotations

from dataclasses import dataclass

from nimloth.agent import Agent, AgentBatch, AgentTarget
from nimloth.training.sft2.objective import SFT2Objective, SFT2StepOutput


@dataclass(frozen=True)
class SFT2Algorithm:
    """组织当前 Agent、target state 与 SFT2 objective。

    本类只表达算法顺序；它不知道 processor、Qwen、cache、EMA 实现、DDP、
    optimizer 或 checkpoint。
    """

    agent: Agent
    target: AgentTarget
    objective: SFT2Objective

    def training_step(
        self,
        batch: AgentBatch,
        *,
        wm_weight: float,
    ) -> SFT2StepOutput:
        model_output = self.agent(
            batch.current,
            batch.action_indices,
            include_lm_loss=True,
        )
        target_states = self.target(batch.next)
        return self.objective(
            model_output=model_output,
            target_states=target_states,
            batch=batch,
            wm_weight=wm_weight,
            training=True,
        )

    def evaluation_step(self, batch: AgentBatch) -> SFT2StepOutput:
        model_output = self.agent(
            batch.current,
            batch.action_indices,
            include_lm_loss=False,
        )
        target_states = self.target(batch.next)
        return self.objective(
            model_output=model_output,
            target_states=target_states,
            batch=batch,
            wm_weight=1.0,
            training=False,
        )

    def unwrapped(self) -> "SFT2Algorithm":
        agent = self.agent.unwrapped()
        return SFT2Algorithm(
            agent=agent,
            target=self.target.with_agent(agent),
            objective=self.objective,
        )


__all__ = ["SFT2Algorithm"]
